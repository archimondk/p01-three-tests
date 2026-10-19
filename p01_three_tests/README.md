# P01 三重检验：冻结分析流程

用一条流程，把旧版的 `audit.py`、`p1`、`p5`、`p7`–`p10` 合并起来。所有参数集中在 `config.py`，规则写在 `PROTOCOL.md`。

## 环境

- Python ≥ 3.10
- 依赖：pandas、numpy、scipy、matplotlib、openpyxl、statsmodels、pytest

## 用法

```bash
# 1. 测试（需要 statsmodels，否则示范模型的测试会被跳过）
python -m pytest tests

# 2. 试跑（结果不进稿件）
python run.py --raw 数据目录 --B 200

# 3. 正式运行
python run.py --raw 数据目录 [--code-register 四川中医医院代码表对照表.xlsx] [--custodian-flags 质控名单.csv]

# 4. 新旧提取比对
python revision_audit.py --old 旧数据目录 --new 新数据目录
```

数据目录里放平台导出的季度申报表，文件名形如 `t_25_S_1.csv`，表示 2025 年第 1 季度。

每次运行会在 `outputs/run_时间戳/` 下生成以下内容：

| 输出 | 内容 |
|---|---|
| `tables/*.csv` 与 `results.xlsx` | Table 1–5、S1–S10，以及过程表 |
| `figures/` | Figure 1–4、S1–S3，PNG 与 TIFF 两种格式，300 dpi |
| `numbers.json` | 正文引用的全部数字，必须以它为唯一来源 |
| `manifest.json` | 输入文件哈希、代码哈希、git tag、软件版本、自助次数与种子 |

## 文件

| 文件 | 作用 |
|---|---|
| `config.py` | 冻结参数。冻结后只允许修改 PATHS、CUSTODIAN、COLUMN_ALIASES |
| `data.py` | 读入、校验、清洗、上报状态编码 |
| `core.py` | REML ICC、整群自助、Spearman–Brown、判定规则、过度离散、DL 估计（不依赖 statsmodels） |
| `analysis.py` | 三重检验、Table 1–3、S1–S9、敏感性分析 |
| `models.py` | 示范分析与 MixedLM 交叉核对（需要 statsmodels） |
| `consequence.py` | 后果分析：名次不确定区间、粗率排名与校正方法的一致性、朴素排名、质控名单比对 |
| `figures.py` | 图件 |
| `run.py` | 入口 |
| `revision_audit.py` | 新旧提取比对 |
| `tests/` | 模拟数据生成器与 45 项测试 |
| `docs/v16_code_review.md` | 对 v1.6 代码的审读记录 |
