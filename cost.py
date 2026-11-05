"""
耗时测试脚本 —— 统计推理各阶段耗时

输出:
  1) 模型加载耗时
  2) 单张推理耗时(含预处理 / 仅前向)
  3) 批推理吞吐(images/s)
  4) 开启 TTA、多尺度后的耗时倍率
  5) 按测试集规模估算总耗时

用法:
  python cost.py --ckpt ../model/pretrain_model/b3_s42.pt --img_dir data/test --n 50
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# cost.py 位于提交包根目录, 代码在 ./code 下
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from model import GlaucomaModel
from utils import RetinaDataset, build_transforms, read_image, amp_ctx, DEVICE


def load(ckpt_path):
    t0 = time.perf_counter()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    a = ckpt["args"]
    m = GlaucomaModel(a["backbone"], False, a["use_age"]).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval()
    if str(DEVICE).startswith("cuda"):
        torch.cuda.synchronize()
    return m, a, ckpt, (time.perf_counter() - t0)


def sync():
    if str(DEVICE).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def official_bench(model, files, tf, repeat=3, warmup=3):
    """官方口径: batch=1, 连续 N 张, 每张图重复 repeat 次取中位数, 再对全部图求平均。

    返回 (每张图的中位数 ms 数组, 每次重复的平均 ms 数组)
    """
    age = torch.zeros(1, 1, device=DEVICE)
    for p in files[:warmup]:                      # 预热
        img = read_image(p)
        x = tf(image=img)["image"].unsqueeze(0).to(DEVICE)
        with amp_ctx():
            model(x, age)
    sync()

    per_run = []                                  # 每次重复: 每张图的 ms
    for _ in range(repeat):
        times = []
        for p in files:
            t0 = time.perf_counter()
            img = read_image(p)                       # 读图(含 JPEG 解码)
            x = tf(image=img)["image"].unsqueeze(0)   # 预处理
            x = x.to(DEVICE)
            with amp_ctx():
                model(x, age)                         # 前向
            sync()
            times.append((time.perf_counter() - t0) * 1000.0)
        per_run.append(times)
    arr = np.asarray(per_run)                     # (repeat, n_images)
    return np.median(arr, axis=0), arr.sum(axis=1) / arr.shape[1]


def bench_forward(model, size, batch, n_iter, tta):
    x = torch.randn(batch, 3, size, size, device=DEVICE)
    age = torch.zeros(batch, 1, device=DEVICE)
    # 预热
    for _ in range(3):
        with amp_ctx():
            _ = model(x, age)
    sync()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        with amp_ctx():
            if tta:
                model(x, age)
                model(torch.flip(x, dims=[3]), age)
                model(torch.flip(x, dims=[2]), age)
                model(torch.flip(x, dims=[2, 3]), age)
            else:
                model(x, age)
    sync()
    dt = time.perf_counter() - t0
    per_img = dt / (n_iter * batch)
    return per_img, (batch / per_img if per_img > 0 else float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_dir", default="data/test")
    ap.add_argument("--sizes", default=None, help="如 512,576,640; 不传用 ckpt 的 img_size")
    ap.add_argument("--n", type=int, default=50, help="预处理测速用的图片数")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--total", type=int, default=5855, help="用于估算总耗时")
    ap.add_argument("--official", action="store_true",
                    help="官方口径: batch=1, 连续100张, 重复3次取中位数, 含预处理")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--n_images", type=int, default=100, help="官方口径使用的图片数")
    args = ap.parse_args()

    model, a, ckpt, t_load = load(args.ckpt)
    sizes = ([int(x) for x in args.sizes.split(",")] if args.sizes
             else [a["img_size"]])
    print(f"[*] device={DEVICE}  backbone={a['backbone']}  "
          f"train_size={a['img_size']}  ckpt加载={t_load:.2f}s")

    if args.official:
        files = [p for p in Path(args.img_dir).iterdir()
                 if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png",
                                                         ".JPG", ".JPEG", ".PNG")]
        files = sorted(files)[:args.n_images]
        if len(files) < args.n_images:
            print(f"[!] 只找到 {len(files)} 张图, 少于要求的 {args.n_images} 张")
        if not files:
            print("[!] img_dir 下没有图片, 无法按官方口径测试")
            return
        tf = build_transforms(sizes[0], False)
        print(f"[*] 官方口径: batch=1, {len(files)} 张, 重复 {args.repeat} 次, "
              f"尺寸 {sizes[0]}")
        med_per_img, run_means = official_bench(model, files, tf,
                                               repeat=args.repeat)
        mean = float(np.mean(med_per_img))
        p50 = float(np.percentile(med_per_img, 50))
        p90 = float(np.percentile(med_per_img, 90))
        print("\n  每次重复的平均(ms/张): "
              + "  ".join(f"{x:.2f}" for x in run_means))
        print(f"  每张图取中位数后的统计: 平均={mean:.2f}  中位={p50:.2f}  "
              f"P90={p90:.2f} ms/张")
        print(f"  （端到端：读图+预处理+前向，batch=1，{len(files)} 张）")
        print(f"  按 {args.total} 张估算: {mean * args.total / 1000:.1f} s "
              f"({mean * args.total / 60000:.1f} min)")
        print("\n[填写到清单 F1] " + f"{mean:.2f} ms/张")
        return

    # --- 预处理耗时 ---
    files = [p for p in Path(args.img_dir).iterdir()
             if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")][:args.n]
    if files:
        tf = build_transforms(sizes[0], False)
        t0 = time.perf_counter()
        for p in files:
            im = read_image(p)
            tf(image=im)
        pre = (time.perf_counter() - t0) / len(files)
        print(f"[*] 预处理(读图+变换) 单张 = {pre * 1000:.1f} ms  (n={len(files)})")
    else:
        pre = 0.0
        print("[!] 未找到图片, 跳过预处理测速")

    # --- 前向耗时 ---
    print(f"\n{'batch':>6} {'size':>6} {'TTA':>5} {'单张前向':>12} {'吞吐(张/秒)':>14}")
    print("-" * 50)
    results = []
    for sz in sizes:
        for b in (1, args.batch):
            for tta in (False, True):
                n_iter = max(2, 20 // b)
                per_img, ips = bench_forward(model, sz, b, n_iter, tta)
                print(f"{b:>6} {sz:>6} {str(tta):>5} {per_img * 1000:>10.1f}ms {ips:>14.1f}")
                results.append((sz, b, tta, per_img, ips))

    # --- 总耗时估算 ---
    n_scale = len(sizes)
    best = max(results, key=lambda r: r[4])
    per_img_total = 1.0 / best[4] + pre
    print(f"\n[*] 最快配置: size={best[0]} batch={best[1]} tta={best[2]}")
    print(f"[*] 单图总耗时(前向+预处理) ≈ {per_img_total * 1000:.1f} ms")
    for tta in (False, True):
        mult = 4 if tta else 1
        est = per_img_total * n_scale * mult * args.total
        print(f"[*] 估算 {args.total} 张, {n_scale} 尺度, TTA={tta}: {est:.1f}s "
              f"({est / 60:.1f} min)")
    print("\n[提示] 上述为单模型耗时; 3 模型集成需乘以 3。")


if __name__ == "__main__":
    main()
