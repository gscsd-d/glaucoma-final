"""
青光眼眼底图像二分类骨架
适配环境: Python 3.12 / PyTorch / CUDA 12.8 / 单卡 T4 16GB
主指标: AUROC   次指标: AUPRC
用法:
  训练: python train.py --mode train --use_age --use_pos_weight
  推理: python train.py --mode infer --tta
"""
import argparse, copy, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
from tqdm import tqdm

# ----------------------------- 全局配置 -----------------------------
COL_IMG, COL_LABEL, COL_AGE = "image", "label", "age"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 无 NVIDIA GPU 时自动回退 CPU (AMD 本机用于调试; Linux+ROCm 下 torch.cuda.is_available() 为 True)
USE_AMP = torch.cuda.is_available() and os.environ.get("DISABLE_AMP", "0") != "1"
AMP_DEVICE = "cuda" if USE_AMP else "cpu"


def amp_ctx():
    """有 CUDA 时用 bf16 混合精度 (指数范围同 fp32, 不会溢出);
    老卡不支持 bf16 时回退 fp16; CPU 上空操作"""
    if not USE_AMP:
        return autocast("cpu", enabled=False)
    try:
        return autocast(AMP_DEVICE, dtype=torch.bfloat16, enabled=True)
    except Exception:
        return autocast(AMP_DEVICE, enabled=True)

# 候选列名 (自动识别官方标注文件里各不相同的列名, 避免 KeyError)
IMG_CANDS = ["image", "Image_name", "image_name", "imagename", "filename",
             "file_name", "file", "img", "img_name", "name", "path"]
LBL_CANDS = ["label", "true_label", "Label", "target", "glaucoma",
             "diagnosis", "class", "y", "is_glaucoma"]
AGE_CANDS = ["age", "Age", "patient_age"]


def resolve_columns(df, need_label=True, need_age=True):
    """自动识别 image/label/age 列名并写回本模块全局, 返回 (img, label, age)。"""
    global COL_IMG, COL_LABEL, COL_AGE
    lower = {str(c).lower(): c for c in df.columns}

    def pick(cands, role, required=True):
        for cand in cands:
            if cand.lower() in lower:
                return lower[cand.lower()]
        if required:
            raise KeyError(f"无法识别『{role}』列; 候选={cands}; 实际列={list(df.columns)}")
        return None

    COL_IMG = pick(IMG_CANDS, "图像(image)")
    if need_label:
        COL_LABEL = pick(LBL_CANDS, "标签(label)")
    if need_age:
        COL_AGE = pick(AGE_CANDS, "年龄(age)", required=False)
    print(f"[*] 列映射 -> image='{COL_IMG}' "
          f"label='{COL_LABEL if need_label else '-'}' "
          f"age='{COL_AGE if need_age else '-'}'")
    return COL_IMG, COL_LABEL, COL_AGE


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def read_image(path):
    """兼容中文/非ASCII路径的安全读取"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------- 数据集 -----------------------------
class RetinaDataset(Dataset):
    def __init__(self, df, img_dir, transform, use_age=False,
                 age_mean=0.0, age_std=1.0, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.use_age = use_age
        self.age_mean, self.age_std = age_mean, age_std
        self.is_test = is_test
        # 文件名大小写索引 (Linux 区分大小写: CSV 常写 .jpg 而磁盘是 .JPG)
        self._index = {}
        if self.img_dir.is_dir():
            for f in self.img_dir.iterdir():
                if f.is_file():
                    self._index[f.name.lower()] = f.name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        rel = str(row[COL_IMG])
        name = self._index.get(Path(rel).name.lower(), Path(rel).name)
        img = read_image(self.img_dir / name)
        img = self.transform(image=img)["image"]

        age = torch.tensor([0.0], dtype=torch.float32)     # 缺失 -> 0.0 即"平均年龄"
        if COL_AGE in self.df.columns:
            v = row[COL_AGE]
            if pd.notna(v) and 1.0 <= float(v) <= 110.0:   # 过滤 0 / 137 等异常值
                age = torch.tensor([(float(v) - self.age_mean) / self.age_std],
                                   dtype=torch.float32)

        if self.is_test:
            return img, age, row[COL_IMG]
        y = torch.tensor(float(row[COL_LABEL]), dtype=torch.float32)
        return img, age, y


# ----------------------------- 数据增强 -----------------------------
def _coarse_dropout(img_size):
    """兼容 albumentations 1.x / 2.x 的参数差异"""
    try:                                   # 2.x
        return A.CoarseDropout(num_holes_range=(4, 8),
                               hole_height_range=(img_size // 16, img_size // 8),
                               hole_width_range=(img_size // 16, img_size // 8),
                               p=0.3)
    except TypeError:                      # 1.x
        return A.CoarseDropout(max_holes=8, max_height=img_size // 16,
                               max_width=img_size // 16, p=0.3)


def build_transforms(img_size, train=True):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.Affine 在 albumentations 1.x/2.x 参数一致 (替代已废弃的 ShiftScaleRotate)
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05),
                     rotate=(-15, 15), p=0.5),
            # 应对多设备/多光照
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            # 模拟眼底圆形视野黑边裁剪
            _coarse_dropout(img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ----------------------------- 模型 -----------------------------
class GlaucomaModel(nn.Module):
    def __init__(self, backbone="efficientnet_b3", pretrained=True,
                 use_age=True, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat = self.backbone.num_features
        self.use_age = use_age
        in_dim = feat + (1 if use_age else 0)
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, 1),
        )

    def forward(self, x, age=None):
        f = self.backbone(x)
        if self.use_age and age is not None:
            f = torch.cat([f, age], dim=1)
        return self.head(f).squeeze(1)   # logits


# ----------------------------- 损失 -----------------------------
class FocalLoss(nn.Module):
    """专治类别极不均衡; pos_weight 可进一步加权阳性"""
    def __init__(self, alpha=0.5, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ----------------------------- 评估 -----------------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    probs, ys, losses = [], [], []
    for img, age, y in loader:
        img, age, y = img.to(DEVICE), age.to(DEVICE), y.to(DEVICE)
        with amp_ctx():
            logits = model(img, age)
            loss = criterion(logits, y)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.cpu().numpy())
        losses.append(loss.item())
    probs = np.concatenate(probs); ys = np.concatenate(ys)
    n_nan = int(np.isnan(probs).sum())
    if n_nan:
        print(f"  [WARN] 评估中有 {n_nan} 个 NaN 预测, 已用 0.5 替代")
        probs = np.nan_to_num(probs, nan=0.5)
    auc = roc_auc_score(ys, probs)
    ap = average_precision_score(ys, probs)
    return np.mean(losses), auc, ap, probs, ys


def best_threshold(ys, probs, mode="f1", beta=2.0, precision_floor=0.5):
    """按指标选阈值。
    mode:
      'f1'     -> 最大化 F1
      'fbeta'  -> 最大化 F-beta (beta>1 偏召回; 默认 beta=2)
      'recall' -> 在 precision >= precision_floor 约束下最大化召回
                  (关键: 避免退化成最低阈值"几乎全判阳性")
    """
    ys = np.asarray(ys).astype(int)
    grid = np.linspace(0.01, 0.99, 99)
    best_t, best_v = 0.5, -1.0
    for t in grid:
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (ys == 1)).sum())
        fp = int(((pred == 1) & (ys == 0)).sum())
        fn = int(((pred == 0) & (ys == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        if mode == "f1":
            v = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        elif mode == "fbeta":
            b2 = beta * beta
            denom = b2 * prec + rec
            v = (1 + b2) * prec * rec / denom if denom else 0.0
        elif mode == "recall":
            v = rec if prec >= precision_floor else -1.0
        else:
            v = rec
        if v > best_v:
            best_t, best_v = t, v
    if mode == "recall" and best_v < 0:      # 无阈值满足精度下限 -> 回退 F1
        return best_threshold(ys, probs, "f1")
    return best_t, best_v


class ModelEMA:
    """权重的指数滑动平均: 训练时维护平滑权重, 评估/保存时用, 通常泛化更好"""
    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


# ----------------------------- 训练 -----------------------------
def train(args):
    set_seed(args.seed)
    tr_df = pd.read_csv(args.train_csv)
    va_df = pd.read_csv(args.val_csv)
    resolve_columns(tr_df, need_label=True, need_age=True)

    # age 标准化
    age_mean = tr_df[COL_AGE].mean() if COL_AGE in tr_df else 0.0
    age_std = tr_df[COL_AGE].std() if COL_AGE in tr_df else 1.0
    if not age_std or np.isnan(age_std):
        age_std = 1.0

    tf_tr = build_transforms(args.img_size, True)
    tf_va = build_transforms(args.img_size, False)
    ds_tr = RetinaDataset(tr_df, args.train_dir, tf_tr, args.use_age, age_mean, age_std)
    ds_va = RetinaDataset(va_df, args.val_dir, tf_va, args.use_age, age_mean, age_std)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    # 正负比 -> pos_weight
    n_pos = int((tr_df[COL_LABEL] == 1).sum()); n_neg = len(tr_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE) if args.use_pos_weight else None
    print(f"[*] pos={n_pos} neg={n_neg} pos_weight={None if pos_weight is None else round(pos_weight.item(),3)}")

    model = GlaucomaModel(args.backbone, not args.no_pretrained, args.use_age).to(DEVICE)
    criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # bf16 混合精度不需要 GradScaler (无溢出/下溢, 无需 loss scaling)
    scaler = None

    # EMA 权重滑动平均 (评估/保存用平滑权重)
    ema = ModelEMA(model, args.ema_decay) if not args.no_ema else None
    eval_model = ema.ema if ema is not None else model

    best_auc, best_path = -1, Path(args.out) / "best.pt"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0
        pbar = tqdm(dl_tr, desc=f"Epoch {epoch+1}/{args.epochs}")
        for img, age, y in pbar:
            img, age, y = img.to(DEVICE), age.to(DEVICE), y.to(DEVICE)
            # 标签平滑 (正则化)
            if args.label_smooth > 0:
                y = y * (1 - args.label_smooth) + 0.5 * args.label_smooth
            # mixup (软标签, FocalLoss 支持软目标)
            if args.mixup > 0 and img.size(0) > 1 and random.random() < 0.5:
                lam = float(np.random.beta(args.mixup, args.mixup))
                perm = torch.randperm(img.size(0), device=img.device)
                img = lam * img + (1 - lam) * img[perm]
                age = lam * age + (1 - lam) * age[perm]
                y = lam * y + (1 - lam) * y[perm]
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                logits = model(img, age)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)   # 跳过 NaN/Inf batch
                continue
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer); scaler.update()
            else:                       # bf16 / CPU: 无需缩放
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        vloss, auc, ap, probs, ys = evaluate(eval_model, dl_va, criterion)
        print(f"[Epoch {epoch+1}] train={tot/max(len(dl_tr),1):.4f} val={vloss:.4f} "
              f"AUROC={auc:.4f} AUPRC={ap:.4f} ({time.time()-t0:.0f}s)")

        if auc > best_auc:
            best_auc = auc
            torch.save({"model": eval_model.state_dict(), "age_mean": age_mean,
                        "age_std": age_std, "args": vars(args)}, best_path)
            print(f"  [OK] saved best (AUROC={auc:.4f}) -> {best_path}")

    # 用最优权重在验证集上选阈值
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    _, auc, ap, probs, ys = evaluate(model, dl_va, criterion)
    t_f1, v_f1 = best_threshold(ys, probs, "f1")
    t_f2, v_f2 = best_threshold(ys, probs, "fbeta", beta=2.0)
    t_rec, v_rec = best_threshold(ys, probs, "recall", precision_floor=0.5)
    print(f"\n[BEST] AUROC={auc:.4f} AUPRC={ap:.4f}")
    print(f" 阈值@maxF1        = {t_f1:.2f} (F1={v_f1:.4f})")
    print(f" 阈值@F2(偏召回)    = {t_f2:.2f} (F2={v_f2:.4f})")
    print(f" 阈值@recall(P>=.5) = {t_rec:.2f} (Recall={v_rec:.4f})  <- 召回优先(有精度下限)")
    pd.DataFrame([{"threshold_f1": t_f1, "threshold_f2": t_f2,
                   "threshold_recall": t_rec,
                   "val_auroc": auc, "val_auprc": ap}]).to_csv(
        Path(args.out) / "threshold.csv", index=False)


# ----------------------------- 推理 + TTA -----------------------------
@torch.no_grad()
def predict_tta(model, img, age, tta=True):
    outs = [torch.sigmoid(model(img, age))]
    if tta:
        outs.append(torch.sigmoid(model(torch.flip(img, dims=[3]), age)))       # hflip
        outs.append(torch.sigmoid(model(torch.flip(img, dims=[2]), age)))       # vflip
        outs.append(torch.sigmoid(model(torch.flip(img, dims=[2, 3]), age)))    # both
    return torch.stack(outs, 0).mean(0)


def infer(args):
    ckpt = torch.load(Path(args.out) / "best.pt", map_location=DEVICE, weights_only=False)
    a = ckpt["args"]; age_mean, age_std = ckpt["age_mean"], ckpt["age_std"]

    model = GlaucomaModel(a["backbone"], False, a["use_age"]).to(DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()

    te_df = pd.read_csv(args.test_csv)
    resolve_columns(te_df, need_label=False, need_age=True)
    tf_te = build_transforms(a["img_size"], False)
    ds = RetinaDataset(te_df, args.test_dir, tf_te, a["use_age"],
                       age_mean, age_std, is_test=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    th_path = Path(args.out) / "threshold.csv"
    threshold = 0.5
    if args.infer_threshold is not None:
        threshold = float(args.infer_threshold)          # 手动指定优先
    elif th_path.exists():
        try:
            row = pd.read_csv(th_path)
            for col in ["threshold_recall", "threshold_f2", "threshold_f1"]:
                if col in row.columns:
                    threshold = float(row[col].iloc[0])
                    break
        except Exception as e:
            print(f"[!] 读取 threshold.csv 失败({e}), 回退 0.5")
    print(f"[*] using threshold = {threshold:.2f}")

    names, probs = [], []
    for img, age, name in tqdm(dl, desc="Infer"):
        img, age = img.to(DEVICE), age.to(DEVICE)
        with amp_ctx():
            p = predict_tta(model, img, age, tta=args.tta)
        probs.append(p.float().cpu().numpy()); names.extend(list(name))

    probs = np.concatenate(probs)
    sub = pd.DataFrame({
        "Image_name": names,
        "Probability": np.round(probs, 6),
        "Pred_Label": (probs >= threshold).astype(int),
    })
    sub.to_csv(args.submission, index=False, encoding="utf-8")
    print(f"[OK] saved {args.submission}  rows={len(sub)}  "
          f"pos_rate={sub['Pred_Label'].mean():.4f}")


# ----------------------------- 阈值分析 (eval) -----------------------------
@torch.no_grad()
def eval_mode(args):
    """加载 best.pt, 在验证集上打印 阈值-精度-召回 表, 便于挑部署阈值"""
    ckpt = torch.load(Path(args.out) / "best.pt", map_location=DEVICE, weights_only=False)
    a = ckpt["args"]
    model = GlaucomaModel(a["backbone"], False, a["use_age"]).to(DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()

    va_df = pd.read_csv(args.val_csv)
    resolve_columns(va_df, need_label=True, need_age=True)
    tf = build_transforms(a["img_size"], False)
    ds = RetinaDataset(va_df, args.val_dir, tf, a["use_age"],
                       ckpt["age_mean"], ckpt["age_std"])
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    _, auc, ap, probs, ys = evaluate(model, dl, FocalLoss())
    ys = ys.astype(int)
    print(f"[eval] AUROC={auc:.4f} AUPRC={ap:.4f}  阳性率={ys.mean() * 100:.2f}%")
    print(" 阈值 | 精度  | 召回  |  F1   |  F2   | 预测阳性率")
    for t in np.arange(0.10, 0.96, 0.05):
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (ys == 1)).sum())
        fp = int(((pred == 1) & (ys == 0)).sum())
        fn = int(((pred == 0) & (ys == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
        print(f" {t:.2f} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {f2:.3f} | {pred.mean():.3f}")
    # 保存验证集预测, 便于离线做完整 PR 分析/挑阈值
    try:
        names = va_df[COL_IMG].astype(str).tolist()
        pd.DataFrame({"Image_name": names, "Probability": probs,
                      "True_Label": ys.astype(int)}).to_csv(
            Path(args.out) / "valid_preds.csv", index=False)
        print(f"[eval] saved -> {Path(args.out) / 'valid_preds.csv'}")
    except Exception as e:
        print(f"[eval] 保存 valid_preds 失败: {e}")
    print("=> 选一个满足临床要求的阈值, 用 --infer_threshold <值> 重新推理")


# ----------------------------- CLI -----------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "infer", "eval"], default="train")
    p.add_argument("--backbone", default="efficientnet_b3")
    p.add_argument("--no_pretrained", action="store_true",
                   help="不使用预训练权重 (离线环境/CPU 快速调试)")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--label_smooth", type=float, default=0.0, help="标签平滑系数, 如 0.05")
    p.add_argument("--mixup", type=float, default=0.0, help="mixup alpha, 如 0.2 (0=关闭)")
    p.add_argument("--no_ema", action="store_true", help="关闭 EMA 权重平均(默认开启)")
    p.add_argument("--ema_decay", type=float, default=0.9998, help="EMA 衰减系数")
    p.add_argument("--use_age", action="store_true")
    p.add_argument("--use_pos_weight", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--infer_threshold", type=float, default=None,
                   help="推理阈值(覆盖 threshold.csv); 不设则用验证集自动选")
    p.add_argument("--train_dir", default="data/train")
    p.add_argument("--val_dir", default="data/val")
    p.add_argument("--test_dir", default="data/test")
    p.add_argument("--train_csv", default="data/train.csv")
    p.add_argument("--val_csv", default="data/val.csv")
    p.add_argument("--test_csv", default="data/test.csv")
    p.add_argument("--out", default="runs")
    p.add_argument("--submission", default="submission.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval_mode(args)
    else:
        infer(args)
