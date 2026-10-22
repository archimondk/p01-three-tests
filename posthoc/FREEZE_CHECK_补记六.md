## 补记六：外部审阅后的呈现修改与事后分析（2026-10-22，冻结后）

- **结论标签改名（仅稿件呈现）**：稿件把冻结判定的四类标签改为筛查用语——Definitely usable → Passed；Provisionally usable → Provisional pass；Usable only with pooling and/or adjustment → Conditional: pooling and/or adjustment；Not usable as currently specified → Did not pass as specified。判定规则、阈值和结果不变，代码中的标签不改，稿件方法部分写明了两者的对应关系。
- **事后分析一：并列处理。** 冻结代码 `consequence.part_a` 用 `nsmallest(k, "crude_rank")` 取粗率“最好 10%”，并列时按机构编号截断。稿件改为报告与第 k 名并列的整组医院，以及随机打破并列时的期望值。数据直接取自冻结运行输出的 `TS12_rank_intervals.csv`，没有重新抽样。2026Q1 全院 HAI：并列组 15 家，全部零事件，可信地位于前 25% 的 1 家；冻结输出为 13 家中 1 家。结论方向不变。
- **事后分析二：朴素排名的进入途径。** `posthoc/posthoc_ties.py` 把朴素排名中无可分析数据的医院分成三类：分母全部缺失或为零、只有低于阈值的分母、分母足够但分子缺失；同时报告并列感知的最好层级构成。该脚本只导入冻结函数，不修改冻结文件，输出只含汇总数。2026-10-22 本地运行（主分析记录数核对为 674），并列组数字与由 `TS12_rank_intervals.csv` 重算的结果逐项一致，冻结十分位中“无可分析数据”的家数也与 `TS11_consequence_summary.csv` 一致；输出文件 `PH1_ties_and_mechanisms.csv` 不含机构编号，不随代码提交。
- **呈现调整**：IPC 人员配备的示范模型从正文移到附加文件 1（Methods and results S1，表 S8 不变）。Spearman–Brown 季度数统一写作投影；CLABSI 精度门槛统一写作压力测试。
- 以上都不改变任何检验、判定或冻结运行的输出。
