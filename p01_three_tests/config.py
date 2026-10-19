# -*- coding: utf-8 -*-
"""冻结分析参数。

本文件与 PROTOCOL.md 一起打 tag 冻结。冻结后，新提取数据只改 PATHS 和
CUSTODIAN 两节（数据位置和质控中心确认的口径），其余参数不得改动；
确需改动的，作为方案偏离写入 PROTOCOL.md 第 9 节并另打 tag。
"""

FREEZE_VERSION = "2.0"

# ---------------------------------------------------------------- 路径（冻结后允许修改）
PATHS = {
    # 平台导出的各季度申报表所在目录，文件名按 RAW_FILE_PATTERN 识别季度
    "raw_dir": "data/raw",
    # 机构编码对照表（第 2 列为 ID）；用于统计已开通编码机构数，可为 None
    "code_register": None,
    # 输出目录；每次运行在其下新建带时间戳的子目录
    "out_dir": "outputs",
}
RAW_FILE_PATTERN = r"t_(\d{2})_S_([1-4])\.csv$"   # t_25_S_1.csv -> 2025Q1
RAW_ENCODINGS = ("gbk", "utf-8-sig", "utf-8")
ID_PATTERN = r"^H\d+$"
# 新提取若字段改名，在这里映射到下面使用的旧字段名：{"新名": "旧名"}
COLUMN_ALIASES = {}

# ---------------------------------------------------------------- 质控中心口径（冻结后允许修改）
CUSTODIAN = {
    "framework_n": 273,          # 纳入全省质控管理框架的二、三级中医机构数
    "framework_source": "Provincial quality-control centre (to be confirmed)",
    "coded_n": 205,              # 无编码对照表时使用；有对照表时以对照表计数为准
    "coded_source": "Platform code register (to be confirmed)",
    # 质控中心确认尚未截止或导出不完整的季度，直接排除出主分析
    "force_exclude_quarters": [],
    # 质控中心确认已完整的季度，即使触发不完整规则也纳入主分析
    "force_include_quarters": [],
}

# ---------------------------------------------------------------- 以下为冻结参数
SEED = 20261019
B_BOOT = 5000                 # Test 2 聚类自助重抽次数
MIN_HOSP = 20                 # 估计 ICC / 容量校正信度所需最少机构数
MIN_OBS_PER_HOSP = 2          # 进入 ICC 的机构至少需要的可分析季度数

# 不完整季度规则：上报机构数 < 0.6 × 其他季度上报机构数中位数 -> 不完整，排除出主分析
INCOMPLETE_QUARTER_RATIO = 0.60

# ---- 监测模块：分母字段与务实阈值（与 v1.6 相同）
MODULES = {
    "overall":  {"en": "Overall HAI",              "den": "同期住院天数",          "thr": 1000},
    "mdro":     {"en": "MDRO",                     "den": "同期住院天数",          "thr": 1000},
    "hh":       {"en": "Hand hygiene",             "den": "应执行手卫生时机数",    "thr": 200},
    "micro":    {"en": "Microbiological testing",  "den": "联合使用抗菌药物总人数", "thr": 20},
    "class1":   {"en": "Class I incision surgery", "den": "1类切口手术总例次数",    "thr": 30},
    "clabsi":   {"en": "CLABSI",                   "den": "中央血管导管累计使用天数", "thr": 100},
    "vap":      {"en": "VAP",                      "den": "有创呼吸机累计使用天数",  "thr": 100},
    "cauti":    {"en": "CAUTI",                    "den": "导尿管累计使用天数",      "thr": 100},
    "hd":       {"en": "Haemodialysis",            "den": "血液透析总人数",          "thr": 30},
    "neonatal": {"en": "Neonatal",                 "den": "新生儿住院天数",          "thr": 200},
}

# 分析可得性指数使用的模块。overall 与 mdro 共用同一分母和阈值，状态恒相同，
# 两者都计入会把住院日模块重复计权，因此主指数只计 9 个不同分母的模块（v1.6 为 10 个）。
INDEX_MODULES = ["overall", "hh", "micro", "class1", "clabsi", "vap", "cauti", "hd", "neonatal"]
# 敏感性：只用不依赖专科服务配置的核心模块
CORE_INDEX_MODULES = ["overall", "hh", "micro", "class1"]
# 敏感性：复现 v1.6 的 10 模块定义
V16_INDEX_MODULES = list(MODULES)

# ---- 11 个指标
# kind: event = 感染结局（发病密度或感染比例），process = 过程比例
# transform: Test 2 计算 ICC 前的变换。v1.6 由数据偏度(skew>1)临时决定，这里固定写死；
# 冻结前须用旧数据核对与 v1.6 的实际选择一致（见 PROTOCOL.md 第 10 节）。
INDICATORS = [
    dict(key="overall_hai", en="Overall HAI incidence density", module="overall",
         num="感染例次数", mult=1000, kind="event", unit="per 1000 patient-days", transform="log1p"),
    dict(key="mdro", en="MDRO incidence density", module="mdro",
         num="多重耐药菌感染新发例次数", mult=1000, kind="event", unit="per 1000 patient-days", transform="log1p"),
    dict(key="cauti", en="CAUTI rate", module="cauti",
         num="尿路感染新发病例例次数", mult=1000, kind="event", unit="per 1000 catheter-days", transform="log1p"),
    dict(key="clabsi", en="CLABSI rate", module="clabsi",
         num="血流感染新发病例例次数", mult=1000, kind="event", unit="per 1000 central-line days", transform="log1p"),
    dict(key="vap", en="VAP rate", module="vap",
         num="呼吸机相关肺炎新发病例例次数", mult=1000, kind="event", unit="per 1000 ventilator-days", transform="log1p"),
    dict(key="ssi", en="SSI, class I incisions", module="class1",
         num="1类切口手术部位感染例次数", mult=100, kind="event", unit="%", transform="log1p"),
    dict(key="hd", en="Haemodialysis-associated infection", module="hd",
         num="血液透析相关感染例次数", mult=100, kind="event", unit="%", transform="log1p"),
    dict(key="neonatal", en="Neonatal HAI incidence density", module="neonatal",
         num="新生儿感染例次数", mult=1000, kind="event", unit="per 1000 patient-days", transform="log1p"),
    dict(key="hh", en="Hand hygiene compliance", module="hh",
         num="手卫生实际执行时机数", mult=100, kind="process", unit="%", transform="none"),
    dict(key="sap", en="Surgical antibiotic prophylaxis, class I", module="class1",
         num="预防使用抗菌药物1类切口例次数", mult=100, kind="process", unit="%", transform="none"),
    dict(key="micro", en="Microbiological testing rate", module="micro",
         num="送检住院患者人数", mult=100, kind="process", unit="%", transform="none"),
]
INDEX_TRANSFORM = "none"

# ---- 结构变量
STRUCT_COLS = {"staff": "同期感控专职人员数", "beds": "实际开放床位数", "level": "级", "grade": "等"}
REDUNDANT_PAIRS = [("住院患者累计住院天数", "同期住院天数"), ("1类切口手术总例次数1", "1类切口手术总例次数")]

# ---- 清洗：合理性上限（与 v1.6 相同），超过则分子置缺失
PLAUSIBILITY_CEILING = {"overall_hai": 20.0, "mdro": 10.0, "clabsi": 20.0, "vap": 30.0, "cauti": 20.0}
MIN_PATIENT_DAYS = 500        # 同期住院天数 (0, 500) 视为分母异常，置缺失
STAFF_PER100_CEILING = 10.0   # 每百床感控专职人员数上限（D7）；超过则人员数置缺失

# ---------------------------------------------------------------- Test 1
T1_PASS_AVAIL = 80.0          # 可分析记录比例 >= 80% 且
T1_MIN_HOSP_PCT = 50.0        # 有可分析数据的机构 >= 50% -> Pass
T1_COND_AVAIL = 50.0          # 50–80% -> Conditional；<50% 或机构 <50% -> Fail

# 精度阈值（次级压力检验）
PRECISION_EXPECTED_EVENTS = 1.96 ** 2   # 3.84：Wald 相对 95% CI 半宽 <= 100%（v1.6 正文误写为 4）
PRECISION_HALF_WIDTH = 0.05             # 过程比例：95% CI 半宽 <= 5 个百分点

# ---------------------------------------------------------------- Test 2
RELIABILITY_TARGET = 0.70

# ---------------------------------------------------------------- Test 3
T3_CEILING_PCT = 25.0         # 过程指标取值 = 100% 的比例 > 25% -> Fail
T3_ZERO_PCT = 75.0            # 零分子比例 > 75% -> excess zeros
T3_MIN_EVENTS = 150           # 总事件数 < 150 -> sparse events（与 excess zeros 同时出现 -> Fail）
T3_PHI = 10.0                 # 过度离散因子 > 10 -> extreme overdispersion
PHI_WINSOR = (0.10, 0.90)     # Spiegelhalter 缩尾分位

# ---------------------------------------------------------------- 综合判定
POOLING_HORIZON_Q = 12        # 达到 R=0.70 所需季度数的置信上限 <= 12（三年）视为可通过合并达到

# ---------------------------------------------------------------- 示范分析
DEMO_OUTCOMES = [
    dict(key="overall_hai", device=None),
    dict(key="mdro", device=None),
    dict(key="cauti", device="util_catheter"),        # 器械使用率 = 器械日 / 住院日
    dict(key="clabsi", device="util_central_line"),
]
DEMO_MAX_ABS_COEF = 10.0      # 任一系数 |β| > 10 视为不稳定，不报告
FUNNEL_INDICATORS = ["overall_hai", "mdro", "cauti"]

# ---------------------------------------------------------------- 后果分析（PROTOCOL.md 第 10A 节）
# 哪个方向更好：感染结局越低越好；手卫生、病原学送检越高越好；I 类切口预防用药越低越好
BETTER = {"overall_hai": "lower", "mdro": "lower", "cauti": "lower", "clabsi": "lower", "vap": "lower",
          "ssi": "lower", "hd": "lower", "neonatal": "lower", "hh": "higher", "sap": "lower", "micro": "higher"}
CONSEQ_PERIODS = ["latest", "pooled"]   # latest = 最后一个主分析季度（模拟季度排名，主）；pooled = 全部主分析季度
CONSEQ_POSTERIOR_DRAWS = 2000
CONSEQ_DECILE = 0.10                    # 粗率排名的“最好/最差 10%”
CONSEQ_QUARTILE = 0.25                  # “可信地位于前/后 25%” = 排名 95% 区间整体落在该范围内
CONSEQ_WIDE_INTERVAL = 0.50             # 排名区间宽度 >= 50% 名次视为“排名基本不可分辨”
CUSTODIAN_FLAG_COLUMNS = ["ID", "period", "indicator", "flag"]   # flag: good / bad；indicator: 指标 key 或 composite

# ---------------------------------------------------------------- 敏感性
THRESHOLD_SCALES = [0.25, 0.32, 0.42, 0.55, 0.76, 1.0, 1.32, 1.74, 2.30, 3.03, 4.0]
VERDICT_SENSITIVITY = [
    # (名称, 改动的参数)
    ("ICC cut-off 0.60", {"RELIABILITY_TARGET": 0.60}),
    ("ICC cut-off 0.80", {"RELIABILITY_TARGET": 0.80}),
    ("Test 2 on point estimate", {"T2_POINT_ESTIMATE_RULE": True}),
    ("Availability pass at 70%", {"T1_PASS_AVAIL": 70.0}),
    ("Availability pass at 90%", {"T1_PASS_AVAIL": 90.0}),
    ("Pooling horizon 8 quarters", {"POOLING_HORIZON_Q": 8}),
    ("Pooling horizon 16 quarters", {"POOLING_HORIZON_Q": 16}),
    ("Excess-zero cut-off 65%", {"T3_ZERO_PCT": 65.0}),
    ("Overdispersion cut-off 5", {"T3_PHI": 5.0}),
]
