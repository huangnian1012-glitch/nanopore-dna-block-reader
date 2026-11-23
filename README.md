# nanopore-dna-block-reader

从分子动力学模拟的纳米孔穿孔电流（`electric-t.csv`）读取 DNA 链碱基顺序的识别器。
输入一条陌生链的三通道电流时间序列，输出碱基顺序假设、分段时间边界与置信度。
底座为预训练时序基础模型（IBM PatchTST、Amazon Chronos-Bolt-mini）的部分微调，
配合保边去噪、多模型集成与测试时平均。

## 署名

- 作者：王涛、沈杨莹镧
- 指导教师：杨恺、张鑫柯

## 能力与实测指标（诚实版）

所有数字来自留出验证 / 留一交叉验证，括号内为随机猜测基线。

| 输入链类型 | 输出 | 准确率 | 随机基线 |
|---|---|---|---|
| 纯链 20 聚体（同碱基×20） | 碱基种类 A/C/G/T | ~0.88 | 0.25 |
| 双段混合链（如 A10C10） | 完整过孔顺序（12 类有序碱基对） | ~0.60 | 0.083 |
| 双段混合链 | 成分（哪两种碱基） | ~0.64 | 0.167 |
| 四段链 40 聚体（`--k 4`） | 逐段碱基（4 字母过孔顺序） | ~0.58（留一法） | 0.25 |
| 四段链 40 聚体 | 真值落在每段前二候选内 | ~0.92（留一法） | 0.50 |

明确的能力边界：

- **纯链 / 混合链类型无法仅凭电流判定**（T 与 A 的阻塞深度几乎相同，二类门控 AUC≈0.50）。
  因此输出同时给出"纯链假设"和"双段链假设"，由使用者按已知链类型取用。
- **每段碱基数（长度）不能从电流可靠读出**：段边界以纳秒给出；纳秒→碱基数的换算
  依赖链内变化较大的逐碱基停留时间。全长串输出（`--lengths`）中的长度来自使用者提供，
  不是模型读取结果，JSON 中对此有显式标注。
- 双段顺序模型仅在 25 条带标签双段链上训练；四段逐段模型在 5 条四段链
  （2 条 janus + 3 条新增）上训练。指标存在小样本与配置选择带来的乐观偏差，
  陌生链实际表现可能略低。
- 超过 4 段的链超出训练范围，分段结果仅供参考。

## 创新点

1. **MD×ML 闭环**：训练与评测数据全部来自仿生纳米通道全原子分子动力学模拟
   （CHARMM36 + TIP3P，多层官能团化碳膜，KCl 0.1–3.0 M，外加电场驱动穿孔），
   解码器的每一项指标都能在受控模拟条件下复训与复核，学习结论与物理条件
   在同一套数据上互相印证，而非借用真实测序仪器数据做黑箱拟合。
2. **物理一致的输入表示**：分数阻塞深度 `f = (I - I0) / |I0|` 逐通道保留深度幅值
   （z-score 归一会抹掉碱基的幅值主签名），按总电流符号统一穿孔方向，
   使不同开孔基线 I0、不同信噪水平的轨迹进入同一量纲。
3. **保边去噪的选择性**：一维全变差（TV，PDHG 求解）与滑动平均、Savitzky-Golay、
   小波软阈值等共 7 种去噪方式系统对比，只有 TV 在去噪的同时保留段间台阶，
   双段顺序准确率不降反升（0.52 → 0.57）。
4. **预训练时序基础模型的部分微调**：Chronos-Bolt-mini 冻结主干，仅解冻输入
   patch 嵌入与末两层编码器，并以固定尺度归一化替换逐迹 InstanceNorm 来保留
   深度幅值；配合 3 种子 × 2 编码器集成与每模型 6 视角测试时平均，
   在小样本（25 条双段链）下仍稳定给出留出验证结果。
5. **诚实评测口径**：全部指标为留出验证或留一交叉验证，随机基线并列给出；
   明确声明"纯链/混合链类型无法仅凭电流判定""段长不可从电流可靠读出"两条
   能力边界，全长串输出中的长度字段显式标注来源为使用者提供。

## 方法概述

1. **物理归一化**：分数阻塞深度 `f = (I - I0) / |I0|`，逐通道计算，I0 为开孔基线
   （有 `cutoff.json` 时用穿孔窗口外尾段，否则用末 10% 近似）；按总电流符号定方向。
   保留深度幅值（z-score 会抹掉碱基主签名）。重采样到 800 点。
2. **保边去噪**：一维全变差（TV，PDHG 求解）。实验对比 7 种去噪方式：滑动平均、
   Savitzky-Golay、小波软阈值均破坏段间台阶与细微形状（双段顺序 0.52→0.33），
   仅 TV 在去噪同时保留台阶（0.52→0.57）。
3. **编码器微调**：
   - PatchTST（`ibm-granite/granite-timeseries-patchtst`，3 通道输入，context 512）；
   - Chronos-Bolt-mini（`amazon/chronos-bolt-mini`）：将逐迹 InstanceNorm 替换为
     固定尺度归一化以保留深度幅值；冻结主干，仅解冻输入 patch 嵌入与末两层编码器。
   - 纯链识别头（4 类）在 167 条纯链副本的随机 engaged 裁剪上训练；
   - 双段顺序头（12 类）在 25 条双段链整迹上训练，时间反转增强（标签同步反转）；
   - 四段逐段头在纯链裁剪 + janus 注册真值段裁剪 + 新增四段链的
     方向无关软标签（{正向, 反向} 各 0.5）段裁剪上训练。
4. **分段**：多通道分段常数动态规划 + BIC 选段数（可用 `--k` 强制已知段数）。
5. **集成与测试时平均**：3 种子 × 2 编码器 = 6 模型；每模型对原序 / 反序 /
   相位平移共 6 视角取概率平均（反序视角按反转碱基对重映射）。

## 快速开始

环境：Python ≥3.10，CPU 即可（无需 GPU）。

```bash
pip install -r requirements.txt
```

模型权重（约 415 MB）不在 git 历史中，从
[Releases → Model weights v1](https://github.com/huangnian1012-glitch/nanopore-dna-block-reader/releases/tag/weights-v1)
下载 `weights_v1.zip`，在仓库根目录解压，恢复 `runs_alt/ runs_mix2/ runs_four/` 三个目录。

中国大陆网络需设置 HuggingFace 镜像（模型底座首次运行时自动下载）：

```bash
# Linux / macOS / Git Bash
export HF_ENDPOINT=https://hf-mirror.com
# Windows cmd
set HF_ENDPOINT=https://hf-mirror.com
```

读一条陌生链：

```bash
python recognize3.py --current path/to/electric-t.csv --cutoff path/to/cutoff.json --out runs_mix2/rec3_demo
```

已知链为四段、且已知各段碱基数（按过孔顺序）时：

```bash
python recognize3.py --current e.csv --cutoff c.json --k 4 --lengths 10,10,10,10 --out runs_mix2/rec3_demo
```

输出：屏幕摘要 + `<out>.json` + `<out>.png`（三通道阻塞曲线与分段彩条）。
JSON 关键字段：`hypothesis_homopolymer`（纯链假设）、`hypothesis_two_block`
（双段假设，含 top3 概率与正反方向）、`segments`（逐段时间边界与碱基猜测）、
`full_string_guess`（仅当提供 `--lengths`；长度来源为用户）、`reliability`。

## 输入格式

`electric-t.csv`：四列 `time_ns,I_cat_nA,I_an_nA,I_tot_nA`，10ps 采样。
`cutoff.json`（可选但强烈建议）：tail 分析产出，含 `cutoff_ns`、`t_first_ns`、
`fully_translocated` 等字段，用于确定穿孔窗口与开孔基线。

## 复现训练

数据准备：将 `nanopore_dataset / mixed_dataset / other_membranes_dataset /
janus_dna_dataset` 四个文件夹置于同一根目录（默认桌面），或用环境变量覆盖：
`NANOPORE_DATA_ROOT`（prep_supervised）、`MIXED_DATA_ROOT`（prep_mix2）、
`TEST_DATA_ROOT`（fm_four 的三条新增四段链）。数据本身不随仓库分发。

```bash
python prep_supervised.py          # -> runs_sup/dataset.npz（纯链+janus 注册真值）
python prep_mix2.py                # -> runs_mix2/dataset.npz（25 条双段链）
python fm_block4.py                # 纯链逐段识别器（chronos/patchtst）
python fm_mix5.py --tv --seeds 3   # 双段顺序 5 折 CV（TV+集成协议）
python fm_final2.py                # 双段顺序最终 6 模型权重
python fm_four.py                  # 四段逐段收编训练 + 留一验证
```

交叉验证与消融记录见各脚本输出 JSON（`runs_mix2/order_v*.json`、
`runs_dn/sweep.json`、`runs_four/four.json` 的生成逻辑）。

## 目录结构

```
recognize3.py        最终识别器接口（唯一推理入口）
recognize.py         电流加载与早期监督读序器（load_current 被复用）
denoise_sweep.py     7 种去噪对比 + TV 实现
fm_block*.py         分段与纯链逐段识别器演进
fm_chronos.py        Chronos 固定尺度归一化与部分微调封装
fm_mix2/3/5.py       双段顺序分类器、TTA、集成 CV
fm_final2.py         双段最终权重训练
fm_four.py           四段链收编训练与留一验证
prep_*.py            数据集构建（物理归一化、盐浓度、注册真值）
bench_new.py         统一预处理与基线 benchmark（CNN/Conformer, CTC 及更新头）
sup_read.py          早期滑窗监督读序器
eval_mix2.py         纯链模型盲测双段链的失效分析
```

## 声明

本项目为科研训练成果展示。指标为小样本留出验证结果，不构成测序级可靠性承诺；
模拟数据与未发表结果不随仓库分发。模型底座版权归 IBM / Amazon 原项目所有，
本仓库仅包含微调代码与自训权重。
