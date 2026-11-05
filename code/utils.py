"""
通用工具: 运行设备 / 混合精度 / 列名识别 / 数据读取 / 数据增强 / 数据集

与 train.py 中的定义一一对应。train.py 为产生权重的原始训练脚本(自包含),
本模块是从中抽取的同源定义, 供 predict.py 独立推理使用。
"""
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.amp import autocast
from torch.utils.data import Dataset

# ----------------------------- 全局配置 -----------------------------
COL_IMG, COL_LABEL, COL_AGE = "image", "label", "age"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available() and os.environ.get("DISABLE_AMP", "0") != "1"
AMP_DEVICE = "cuda" if USE_AMP else "cpu"


def amp_ctx():
    """有 CUDA 时用 bf16 混合精度(指数范围同 fp32, 不会溢出); 老卡回退 fp16; CPU 上空操作"""
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
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


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
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05),
                     rotate=(-15, 15), p=0.5),
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
