# 青光眼眼底图像二分类 —— 最终提交材料

线上评分 **98.58** ｜ 验证集 AUROC **0.9907** / AUPRC **0.8782**

本仓库包含代码、训练说明与许可证。模型权重见 Release 附件（或提交包内 `model/pretrain_model/`）。

---

## 1. 环境

| 项 | 说明 |
|---|---|
| Python | 3.9+ |
| PyTorch | 2.x（CUDA 12.1 验证通过） |
| timm | 0.9+ |
| albumentations | 1.3+ |
| 其他 | 见 `code/requirements.txt` |

```bash
pip install -r code/requirements.txt
```

---

## 2. 数据

| 集合 | 图像数 |
|---|---|
| train | 51,396 |
| valid | 5,849 |
| test | 5,855 |
| 合计 | 63,100 |

标注文件字段：`Image_name`、`Age`、`True_Label`（0/1）。验证集阳性率 3.27%（191 张）。

目录结构：

```
data/
├── train.csv  valid.csv  test.csv
├── train/  valid/  test/
```

---

## 3. 方法

**单模型**：ImageNet 预训练骨干（timm，去掉分类层）→ 池化特征 → 拼接标准化年龄 → 两层 MLP 头 → sigmoid 概率。

**训练**：Focal Loss（α=0.5, γ=2.0）+ pos_weight≈30；AdamW（lr 3e-4, wd 1e-4）+ 余弦退火；batch 64；bf16 混合精度；EMA（decay 0.9998）；mixup α=0.2、标签平滑 0.05；按验证集 AUROC 保存最优权重。

**集成**：3 个 EfficientNet 概率加权平均（1 : 1 : 0.6），配合多尺度 TTA（512/576/640 × 4 翻转视图 = 每模型 12 次前向）。

---

## 4. 训练

```bash
python code/train.py --mode train \
  --backbone efficientnet_b3 --img_size 576 \
  --batch 64 --epochs 30 --lr 3e-4 \
  --use_age --use_pos_weight \
  --mixup 0.2 --label_smooth 0.05 --seed 42 \
  --train_dir data/train --val_dir data/valid \
  --train_csv data/train.csv --val_csv data/valid.csv \
  --workers 8 --out runs_b3_s42
```

最终提交使用的三个模型：

| 输出目录 | 骨干 | 分辨率 | 轮数 | seed | 单模 val AUROC |
|---|---|---|---|---|---|
| `runs_b3_s42` | efficientnet_b3 | 576 | 30 | 42 | 0.9860 |
| `runs_b3_s123` | efficientnet_b3 | 576 | 30 | 123 | 0.9886 |
| `runs_b4` | efficientnet_b4 | 576 | 25 | 42 | 0.9885 |

---

## 5. 推理 / 复现提交

```bash
python code/predict.py \
  --ckpts model/pretrain_model/runs_b3_s42/best.pt \
          model/pretrain_model/runs_b3_s123/best.pt \
          model/pretrain_model/runs_b4/best.pt \
  --weights 1 1 0.6 \
  --sizes 512,576,640 --tta \
  --test_csv data/test.csv --test_dir data/test \
  --val_csv data/valid.csv --val_dir data/valid \
  --submission submission.csv
```

**一致性校验**：生成的 `submission.csv` 应为 5855 行、概率均值 0.092409，
指纹 `sha256[:32] = 62c4fd6488ff281616ccb477e7264a37`。

```bash
python -c "import hashlib,pandas as pd; d=pd.read_csv('submission.csv'); n=d.Image_name.astype(str); q=d.Probability.astype(float); print(len(d), round(q.mean(),6), hashlib.sha256(chr(10).join('%s,%.6f'%(a,b) for a,b in zip(n,q)).encode()).hexdigest()[:32])"
```

耗时测试：

```bash
python cost.py --ckpt model/pretrain_model/runs_b3_s42/best.pt --img_dir data/test
```

---

## 6. 目录结构

```
.
├── code/
│   ├── train.py          训练 / 推理 / 阈值分析（自包含）
│   ├── predict.py        独立推理脚本（支持多模型 + 多尺度 + TTA）
│   ├── model.py          网络结构 / FocalLoss / EMA
│   ├── utils.py          数据读取 / 增强 / 数据集
│   └── requirements.txt
├── model/pretrain_model/ 模型权重（runs_b3_s42 / runs_b3_s123 / runs_b4）
├── licenses/             许可证原文
├── cost.py               耗时测试
└── submission.csv        线上最高分提交（5855 行）
```

---

## 7. 许可证

| 对象 | 许可证 | 文件 |
|---|---|---|
| `timm` | Apache-2.0 | `licenses/timm-LICENSE.txt` |
| EfficientNet-B3/B4 预训练权重 | Apache-2.0（模型卡声明） | `licenses/efficientnet_b{3,4}-modelcard.md` |
| PyTorch | BSD 风格 | `licenses/pytorch-LICENSE.txt` |
| albumentations | MIT | `licenses/albumentations-LICENSE.txt` |
| Apache-2.0 正文 | — | `licenses/apache-2.0.txt` |

---

## 8. 数据质量核查结论

- 图片级：三集合间精确重复 **0 组**；训练↔验证近重复 **0 张**（48×48 归一化描述子余弦 ≥0.998）。
- 患者级：相邻编号同年龄比例 0.470，随机打乱基线 0.028（16.8 倍），判定原始数据按患者成对组织且**切分按患者进行**。

---

## 9. 联系

张孝昊 3462215861@qq.com
