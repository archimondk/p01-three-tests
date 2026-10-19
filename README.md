# P01 三重检验：冻结分析流程（v2.0）

| 文件 | 内容 |
|---|---|
| `p01_three_tests/` | 冻结的代码包，`PROTOCOL.md` 为方案正本 |
| `FREEZE_CHECK_v2.0.md` | 冻结前核对的完整记录（测试、旧数据对照、D1–D7） |
| `v16_code_review.md` | 对 v1.6 代码的审读记录 |
| `p01_three_tests_freeze_v2.0.zip` | 冻结前的原始包（未修改） |
| `PROTOCOL.md` | 与 `p01_three_tests/PROTOCOL.md` 同步的副本 |

冻结 tag：`p01-three-tests-freeze-v2.0`。

```bash
cd p01_three_tests
python -m pytest tests                                  # 45 项
python run.py --raw <数据目录> --code-register <代码表.xlsx>   # 正式运行 B=5000
```
