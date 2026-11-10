"""
推理耗时脚本（严格按《比赛参赛规则》第7条）

用法（官方规定接口）:
    python cost.py <submission.csv> <pictures_100> <pictures_5> \
        [--ckpts ckpt1 ckpt2 ckpt3] [--weights 1 1 0.6] \
        [--sizes "512,576,640"] [--no_tta] [--test_csv test.csv]

口径:
  · batch=1 串行, 单张端到端(读图→预处理→前向→输出预测), 含预处理
  · 5 张预热图不计入; 模型加载/初始化不计入
  · 连续 3 轮: 先对每张取 3 轮均值, 再对 100 张取平均
  · 与 submission.csv 逐张比对: 二值判定须完全一致, |Δp|<=1e-3 视为正常波动
"""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 保证无论从哪个目录执行，都能找到同包内的 code/ 模块
import sys
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import predict as P
from utils import DEVICE

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_sizes(spec, n):
    per = [[int(x) for x in g.replace(" ", "").split(",") if x]
           for g in spec.split(";") if g.strip()]
    if len(per) == 1:
        return per * n
    assert len(per) == n, "--sizes 分组数与模型数不一致"
    return per


def list_imgs(d):
    return sorted([p for p in Path(d).iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXT])


def find_age_csv(cli, d100, sub_csv=None):
    """年龄文件多路径查找: 官方只传 3 个参数, 必须能自动定位"""
    if cli and Path(cli).exists():
        return Path(cli)
    here = Path(__file__).resolve().parent
    cands = [
        Path(d100) / "test.csv",
        Path(d100).parent / "test.csv",
        Path(d100).parent / "data" / "test.csv",
        Path(sub_csv).parent / "test.csv" if sub_csv else None,
        here / "data" / "test.csv",
        here / ".." / "data" / "test.csv",
        here.parent / "autodl-tmp" / "data" / "test.csv",
    ]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def derive_threshold(sub):
    """由 submission 的 Pred_Label 反推阈值（可分离时取中点）"""
    p0 = sub[sub.Pred_Label == 0].Probability
    p1 = sub[sub.Pred_Label == 1].Probability
    if len(p0) and len(p1):
        return float((p0.max() + p1.min()) / 2)
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_csv")
    ap.add_argument("pictures_100")
    ap.add_argument("pictures_5")
    ap.add_argument("--ckpts", nargs="+", default=None)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--sizes", default="",
                    help="默认空=各模型用自身训练尺度（与提交链路一致）")
    ap.add_argument("--tta", action="store_true",
                    help="启用翻转TTA；默认关闭，与提交链路一致")
    ap.add_argument("--test_csv", default=None, help="提供 age 的标注文件")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="调试用: 只跑前 N 张")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    ckpts = args.ckpts or [
        str(root / "model" / "pretrain_model" / n / "best.pt")
        for n in ("runs_b3_s42", "runs_b3_s123", "runs_b4")]
    weights = args.weights or [1.0, 1.0, 0.6]
    tta = bool(args.tta)

    # ---- 模型加载（不计时）----
    t0 = time.perf_counter()
    models, w = P.load_models(ckpts, weights)
    sizes = parse_sizes(args.sizes, len(models))
    load_s = time.perf_counter() - t0

    imgs100 = list_imgs(args.pictures_100)
    imgs5 = list_imgs(args.pictures_5)
    if args.limit:
        imgs100 = imgs100[:args.limit]

    print(f"[INFO] 模型 {len(models)} 个, 权重 {w}")
    print(f"[INFO] 推理配置: sizes={sizes} TTA={tta} batch=1")
    print(f"[INFO] 预热图 {len(imgs5)} 张, 计时图 {len(imgs100)} 张, 轮数 {args.rounds}")
    print(f"[INFO] 模型加载耗时 {load_s*1000:.0f} ms（不计入）")
    print(f"[INFO] device={DEVICE}")

    # ---- age 映射 ----
    age_map = {}
    csv = find_age_csv(args.test_csv, args.pictures_100, args.submission_csv)
    if csv:
        df = pd.read_csv(csv)
        cols = {str(c).lower(): c for c in df.columns}
        ic = cols.get("image_name") or cols.get("image")
        ac = cols.get("age")
        if ic and ac:
            for _, r in df.iterrows():
                age_map[Path(str(r[ic])).name.lower()] = r[ac]
            print(f"[INFO] 年龄取自 {csv}（{len(age_map)} 条）")
    if not age_map:
        print("[WARN] 未找到年龄文件, age 按均值处理, 可能与 submission 不一致")

    # ---- 预热（不计时）----
    for p in imgs5:
        P.infer_one(models, w, sizes, tta, str(p), age_map.get(p.name.lower()))

    # ---- 计时轮 ----
    per_img = {p.name: [] for p in imgs100}
    for r in range(args.rounds):
        t_round = time.perf_counter()
        for p in imgs100:
            t0 = time.perf_counter()
            prob = P.infer_one(models, w, sizes, tta, str(p),
                               age_map.get(p.name.lower()))
            float(prob)                          # 输出预测结果
            per_img[p.name].append((time.perf_counter() - t0) * 1000.0)
        print(f"  第 {r+1} 轮完成: {time.perf_counter()-t_round:.1f} s")

    img_mean = [float(np.mean(v)) for v in per_img.values()]
    avg_ms = float(np.mean(img_mean))

    # ---- 与 submission.csv 比对（重跑一次取稳定概率）----
    sub = pd.read_csv(args.submission_csv)
    sub.columns = [str(c).strip() for c in sub.columns]
    name_col = "Image_name" if "Image_name" in sub.columns else sub.columns[0]
    prob_col = "Probability" if "Probability" in sub.columns else sub.columns[1]
    th = derive_threshold(sub) if "Pred_Label" in sub.columns else 0.5

    sub_map = {}
    for _, r in sub.iterrows():
        sub_map[Path(str(r[name_col])).name.lower()] = float(r[prob_col])

    checked, mism, max_dp = 0, 0, 0.0
    for p in imgs100:
        key = p.name.lower()
        if key not in sub_map:
            continue
        prob = P.infer_one(models, w, sizes, tta, str(p),
                           age_map.get(key))
        checked += 1
        dp = abs(prob - float(sub_map[key]))
        max_dp = max(max_dp, dp)
        if dp > 1e-3 and ((prob >= th) != (float(sub_map[key]) >= th)):
            mism += 1

    status = ("未超过上限" if avg_ms <= 100
              else ("明显超限(100-150ms)" if avg_ms <= 150 else "严重超限(>150ms)"))
    print("\n" + "=" * 70)
    print(f"[CONCLUSION] 结论：{len(imgs100)} 张样本、batch=1 串行、连续{args.rounds}轮；"
          f"单张端到端墙钟时间先按每张{args.rounds}轮取均值，再对{len(imgs100)}张取平均，"
          f"最终平均耗时 {avg_ms:.0f} ms，{status}；"
          f"预测结果与 submission.csv 比对 {checked} 张，不一致 {mism} 张；"
          f"推理精度 {'fp16/bf16' if 'cuda' in str(DEVICE) else 'fp32'}。")
    print(f"[STATS] 单张均值 {avg_ms:.1f} ms | 中位 {statistics.median(img_mean):.1f} ms "
          f"| 最大 {max(img_mean):.1f} ms | 最大概率偏差 {max_dp:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
