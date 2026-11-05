# 最终提交材料 —— 打包清单

> 截止：**2026-11-06 23:59**
> 最终文件名：`队长名_最终提交材料.zip`

---

## 当前包内已有（已完成）

```
提交包/
├── code/
│   ├── train.py            ✅ 产生权重的原始训练脚本（自包含，可直接运行）
│   ├── predict.py          ✅ 独立推理脚本（已用真实权重端到端验证）
│   ├── model.py            ✅ 网络结构 / FocalLoss / EMA
│   ├── utils.py            ✅ 数据读取 / 增强 / 数据集
│   └── requirements.txt    ✅ 依赖清单
├── model/
│   └── pretrain_model/
│       ├── runs_b3_s42/best.pt   ✅ 42.8 MB，B3，11.1M 参数
│       ├── runs_b3_s123/best.pt  ✅ 42.8 MB，B3，11.1M 参数
│       ├── runs_b4/best.pt       ✅ 69.4 MB，B4，18.0M 参数
│       └── 放入说明.txt           ✅ 权重清单与核验记录
├── licenses/                    ✅ 6 个文件，全部核验为真实文本
│   ├── timm-LICENSE.txt              Apache-2.0（11,343 字符）
│   ├── apache-2.0.txt                Apache-2.0 正文（11,358 字符）
│   ├── pytorch-LICENSE.txt           BSD 风格（3,464 字符）
│   ├── albumentations-LICENSE.txt    MIT（1,114 字符）
│   ├── efficientnet_b3-modelcard.md  license=apache-2.0
│   └── efficientnet_b4-modelcard.md  license=apache-2.0
├── cost.py                 ✅ 耗时测试脚本（已验证可运行）
├── submission.csv          ✅ 线上最高分 98.58 的提交
├── 模型核验材料与数据申明清单-填写草稿.md  ✅ 已含实测参数量
└── 提交清单-CHECKLIST.md    ← 本文件
```

### 已验证项（真实执行过，非声明）

| 验证内容 | 结果 |
|---|---|
| `model.py` / `utils.py` / `predict.py` / `cost.py` / `train.py` 语法 | 全部编译通过 |
| `train.py` 可训练 | 用 demo 数据跑通 1 轮，产出 best.pt |
| `predict.py` 端到端（真实权重） | 包内 `runs_b3_s42/best.pt` 跑通，输出 6 行合法 submission |
| `cost.py` 可运行 | 输出各 batch/尺度/TTA 的耗时与吞吐 |
| 三个权重可加载 | backbone / img_size=576 / use_age=True 全部正确，合计 40.2M 参数 |
| 许可证真实性 | 6 个文件均为纯文本，无 HTML/404 污染；两个模型卡均声明 `apache-2.0` |
| `submission.csv` | 5855 行，指纹 `0.092409 / 62c4fd64…` |
| `code/train.py` 完整性 | 与原始训练脚本 MD5 一致（`38d3b339…`） |

---

## ❌ 仍需完成

| # | 事项 | 说明 |
|---|---|---|
| ~~1~~ | ~~下载 3 个权重~~ | ✅ **已完成**，放在 `model/pretrain_model/runs_*/best.pt` |
| ~~2~~ | ~~下载许可证~~ | ✅ **已完成**，6 个文件均已核验 |
| 3 | 下载官方模板并填写 | 《模型核验材料与数据申明清单_模板(更新).xlsx》，内容照抄 `填写草稿.md` |
| 4 | 提供公网可访问链接 | 权重/代码/许可证，需**无痕窗口能直接打开**，不要求登录 |
| 5 | 补全草稿里的 `【待填】` | 队伍信息、数据来源、GPU 型号、训练耗时、推理耗时 |
| 6 | 重命名并压缩 | `队长名_最终提交材料.zip` |

---

## 打包命令

```bash
cd "C:/Users/ASUS/Desktop/提交包"
# 补齐后执行（把 队长名 换成真实姓名）
powershell -NoProfile -Command "Compress-Archive -Path * -DestinationPath '../队长名_最终提交材料.zip' -Force"
```

> 注意：压缩时**不要把 `提交清单-CHECKLIST.md` 和 `模型核验材料与数据申明清单-填写草稿.md`
> 之外的临时文件打进去**；上面两条是本工作文件，正式包内可保留也可删。
> `code/__pycache__` 若存在请先删除。

---

## 截止前 30 分钟自检

```bash
# 1) 权重可加载
python -c "
import torch
for f in ['b3_s42.pt','b3_s123.pt','b4.pt']:
    ck=torch.load('model/pretrain_model/'+f,map_location='cpu',weights_only=False)
    print(f, ck['args']['backbone'], ck['args']['img_size'], ck['args']['use_age'])
"

# 2) 提交文件指纹
python -c "
import hashlib,pandas as pd
d=pd.read_csv('submission.csv'); n=d.Image_name.astype(str); q=d.Probability.astype(float)
print(len(d), round(q.mean(),6), hashlib.sha256(chr(10).join('%s,%.6f'%(a,b) for a,b in zip(n,q)).encode()).hexdigest()[:32])
"

# 3) 链接无痕可访问（手动）
# 4) zip 解压后目录结构完整
```

---

## 风险提示

1. **规则第 11 条**：伪造链接、权重哈希、许可证 = 严重违规。本包内所有数字均可复算，
   但 `【待填】` 项必须用真实信息，**宁缺毋滥**。
2. **权重与 submission.csv 必须对应**。若最终提交的 csv 不是 `sub_ms.csv`（98.58 那份），
   请替换 `submission.csv` 并同步更新草稿第六节的指纹。
3. **体积**：三个 best.pt 约 150–200 MB，确认未超过赛事限制。
4. **时间**：距截止不足 24 小时，第 1、2 项（下载权重与许可证）建议**优先做**，其余可并行。
