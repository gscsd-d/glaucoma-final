"""
独立推理脚本 —— 生成 submission.csv

支持:
  · 单模型 / 多模型加权集成
  · 多尺度 TTA(同一张图缩放到多个尺寸后取平均)
  · 翻转 TTA(原图 / 水平 / 垂直 / 双向)
  · 可选: 在验证集上打印 AUROC / AUPRC(用于复核)

复现本次线上 98.58 的提交(在提交包根目录执行):
  python code/predict.py \
    --ckpts model/pretrain_model/runs_b3_s42/best.pt \
            model/pretrain_model/runs_b3_s123/best.pt \
            model/pretrain_model/runs_b4/best.pt \
    --weights 1 1 0.6 \
    --sizes 512,576,640 --tta \
    --test_csv data/test.csv --test_dir data/test \
    --val_csv data/valid.csv --val_dir data/valid \
    --submission submission.csv

说明:
  --sizes 不传 = 每个模型用其 checkpoint 中记录的 img_size(单尺度)
  --sizes "512,576,640"        = 所有模型统一用这三档
  --sizes "512,576,640;576"    = 分号分隔, 按 --ckpts 顺序逐模型指定
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GlaucomaModel
from utils import (RetinaDataset, build_transforms, resolve_columns,
                   read_image, amp_ctx, DEVICE)


def load_models(ckpts, weights=None):
    """加载并返回 (models, weights), 统一供 cost.py 复用"""
    models = [load_model(c) for c in ckpts]
    w = weights if weights else [1.0] * len(models)
    assert len(w) == len(models), "weights 数量必须等于 ckpts 数量"
    return models, w


@torch.no_grad()
def infer_one(models, weights, sizes, tta, img_path, age_value=None):
    """单张图片端到端推理(供 cost.py 调用), 返回 sigmoid 概率。

    与批量推理共用 read_image / build_transforms / predict_tta, 保证口径一致。
    """
    img = read_image(img_path)                     # 读图(含 JPEG 解码)
    probs = []
    for m, szs in zip(models, sizes):
        a = m["args"]
        ag = torch.tensor([[0.0]], dtype=torch.float32)
        if age_value is not None and m["use_age"]:
            v = float(age_value)
            if 1.0 <= v <= 110.0:
                ag = torch.tensor([[(v - m["age_mean"]) / m["age_std"]]],
                                  dtype=torch.float32)
        ag = ag.to(DEVICE)
        per = []
        for sz in (szs or [a["img_size"]]):
            tf = build_transforms(sz, False)
            x = tf(image=img)["image"].unsqueeze(0).to(DEVICE)
            with amp_ctx():
                p = predict_tta(m["model"], x, ag, tta=tta)
            per.append(float(p.item()))
        probs.append(float(np.mean(per)))
    w = np.asarray(weights, dtype=float)
    return float((np.asarray(probs) * w).sum() / w.sum())


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    a = ckpt["args"]
    model = GlaucomaModel(a["backbone"], False, a["use_age"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return {"model": model, "args": a,
            "age_mean": ckpt["age_mean"], "age_std": ckpt["age_std"],
            "tag": Path(ckpt_path).parent.name or Path(ckpt_path).stem}


@torch.no_grad()
def predict_tta(model, img, age, tta=True):
    """4 视图翻转 TTA: 原图 / 水平 / 垂直 / 双向"""
    outs = [torch.sigmoid(model(img, age))]
    if tta:
        outs.append(torch.sigmoid(model(torch.flip(img, dims=[3]), age)))
        outs.append(torch.sigmoid(model(torch.flip(img, dims=[2]), age)))
        outs.append(torch.sigmoid(model(torch.flip(img, dims=[2, 3]), age)))
    return torch.stack(outs, 0).mean(0)


@torch.no_grad()
def predict(m, df, img_dir, is_test, tta, batch, workers, sizes=None):
    """返回与 df 行序对齐的概率数组; sizes 给多个则做多尺度 TTA 取平均"""
    a = m["args"]
    sizes = sizes or [a["img_size"]]
    all_probs = []
    for sz in sizes:
        tf = build_transforms(sz, False)
        ds = RetinaDataset(df, img_dir, tf, a["use_age"],
                           m["age_mean"], m["age_std"], is_test=is_test)
        dl = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=True)
        probs = []
        for item in tqdm(dl, desc=f"  {m['tag']}/{sz}", leave=False):
            img, age = item[0].to(DEVICE), item[1].to(DEVICE)
            with amp_ctx():
                p = predict_tta(m["model"], img, age, tta=tta)
            probs.append(p.float().cpu().numpy())
        all_probs.append(np.concatenate(probs))
    return np.mean(all_probs, axis=0)


def parse_sizes(spec, n_models):
    """None -> [None]*n; 'a,b,c' -> 所有模型同尺度; 'x;y' -> 逐模型"""
    if not spec:
        return [None] * n_models
    groups = [g for g in spec.split(";") if g.strip()]
    per = [[int(x) for x in g.replace(" ", "").split(",") if x] for g in groups]
    if len(per) == 1:
        return per * n_models
    if len(per) != n_models:
        raise SystemExit(f"--sizes 分组数 {len(per)} != 模型数 {n_models}")
    return per


def pick_dirs(single, multi, n_models, name):
    if not multi:
        return [single] * n_models
    if len(multi) != n_models:
        raise SystemExit(f"--{name} 数量 {len(multi)} != 模型数 {n_models}")
    return multi


def best_threshold(ys, probs, precision_floor=0.5):
    """在 precision >= floor 约束下最大化召回; 无解时回退 0.5"""
    ys = np.asarray(ys).astype(int)
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (ys == 1)).sum())
        fp = int(((pred == 1) & (ys == 0)).sum())
        fn = int(((pred == 0) & (ys == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        if prec >= precision_floor and rec > best_v:
            best_t, best_v = t, rec
    return best_t


def main():
    p = argparse.ArgumentParser(description="青光眼眼底图像推理脚本")
    p.add_argument("--ckpts", nargs="+", required=True, help="一个或多个 best.pt")
    p.add_argument("--weights", nargs="*", type=float, default=None,
                   help="集成权重, 数量须与 --ckpts 一致; 不传则等权")
    p.add_argument("--test_csv", default="data/test.csv")
    p.add_argument("--test_dir", default="data/test")
    p.add_argument("--test_dirs", nargs="*", default=None,
                   help="逐模型的测试图目录(混整图/裁剪图时用)")
    p.add_argument("--val_csv", default=None, help="可选: 给定则打印 AUROC/AUPRC")
    p.add_argument("--val_dir", default="data/valid",
                   help="验证图目录(与 --val_csv 配套)")
    p.add_argument("--val_dirs", nargs="*", default=None)
    p.add_argument("--sizes", default=None,
                   help="多尺度 TTA: '512,576,640' 或 '512,576,640;576'")
    p.add_argument("--tta", action="store_true", help="启用翻转 TTA")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threshold", type=float, default=None,
                   help="不传 = 有验证集时按召回优先自动选, 否则 0.5")
    p.add_argument("--submission", default="submission.csv")
    args = p.parse_args()

    models = [load_model(c) for c in args.ckpts]
    n = len(models)
    weights = args.weights if args.weights else [1.0] * n
    assert len(weights) == n, "weights 数量必须等于 ckpts 数量"
    sizes = parse_sizes(args.sizes, n)
    test_dirs = pick_dirs(args.test_dir, args.test_dirs, n, "test_dirs")
    print(f"[*] 集成 {n} 个模型, weights={weights}")

    threshold = args.threshold if args.threshold is not None else 0.5
    if args.val_csv and Path(args.val_csv).exists():
        if not args.val_dirs and not Path(args.val_dir).is_dir():
            raise SystemExit(f"[!] --val_dir 不存在: {args.val_dir} "
                             f"(请显式指定验证图目录)")
        va_df = pd.read_csv(args.val_csv)
        _, lbl_col, _ = resolve_columns(va_df, need_label=True, need_age=True)
        ys = va_df[lbl_col].values.astype(int)
        val_dirs = pick_dirs(args.val_dir, args.val_dirs, n, "val_dirs")
        vp = np.stack([predict(m, va_df, d, False, args.tta, args.batch,
                               args.workers, s)
                       for m, d, s in zip(models, val_dirs, sizes)], axis=0)
        ens_val = np.average(vp, axis=0, weights=weights)
        auc = roc_auc_score(ys, ens_val)
        ap = average_precision_score(ys, ens_val)
        print(f"[ensemble] val AUROC={auc:.4f} AUPRC={ap:.4f}")
        if args.threshold is None:
            threshold = best_threshold(ys, ens_val)
        print(f"[*] threshold = {threshold:.2f}")

    te_df = pd.read_csv(args.test_csv)
    t_img, _, _ = resolve_columns(te_df, need_label=False, need_age=True)
    tp = np.stack([predict(m, te_df, d, True, args.tta, args.batch,
                           args.workers, s)
                   for m, d, s in zip(models, test_dirs, sizes)], axis=0)
    ens = np.average(tp, axis=0, weights=weights)
    names = te_df[t_img].astype(str).tolist()

    sub = pd.DataFrame({
        "Image_name": names,
        "Probability": np.round(ens, 6),
        "Pred_Label": (ens >= threshold).astype(int),
    })
    sub.to_csv(args.submission, index=False, encoding="utf-8")
    print(f"[OK] saved {args.submission}  rows={len(sub)}  "
          f"threshold={threshold:.2f}  pos_rate={sub['Pred_Label'].mean():.4f}")


if __name__ == "__main__":
    main()
