# QQQM 公开行情机器学习与每日视频流程

## 每日建议操作

1. 启动 `dashboard.py`，进入「多平台视频脚本与配图」。打开页面本身不会联网，也不会训练。
2. 点击「更新行情 → 复盘历史预测 → 训练盘中候选 → 生成曲线和视频包」。这个按钮不依赖 Alpaca；程序会尝试刷新公开日线、十五分钟分时和新闻。
3. 程序选择一个数据源（EODHD、Yahoo 或已有 IEX），训练和复盘阶段不会把不同来源的价格拼接起来。只有至少 25 个完整交易日、最新完整分时与最近已完成交易日对齐，并且生成了带有 Ridge 主路径的候选，流程才会把第二张图标记为「机器学习候选主路径」。
4. 页面会重新读取最新日期并保留上次运行摘要。口播稿默认 30–60 秒，图二里的开盘/收盘参考价若属于机器学习候选，取自橙色模型主路径；历史 P10/P50/P90 区间仍只是经验范围。
5. 下一个交易日收盘并且同一来源的真实十五分钟数据已可用后，再运行一次流程。系统会把上次事前保存的预测与真实路径对账，记录 MAE 等误差，再使用全部已完成交易日重新拟合候选。

## 查看状态和成绩

- 最新公开行情学习流程：`reports/public_intraday_learning_cycle/latest_status.json`
- 当前盘中候选快照：`reports/intraday/QQQM_intraday_forecast_latest.json`
- 不可变目标日预测档案：`reports/intraday/archive/QQQM_intraday_forecast_YYYY-MM-DD*.json`
- 盘中路径滚动回测：`reports/intraday/QQQM_intraday_path_backtest_daily_latest.csv` 和同目录下的 JSON 报告
- 历史预测反馈数据集：`reports/model_lab/`
- 最新视频素材：`reports/video_scripts/YYYYMMDD/`

检查 `latest_status.json` 时，重点看：

- `latest_complete_intraday_date` 是否等于最近已完成交易日。
- `forecast_as_of_date` 是否与该完整交易日一致。
- `forecast_target_date` 是否是下一次尚未开始的完整交易日。
- `completed_training_sessions` 是否至少为 25。
- `model_family` 是否为 Ridge 分时主路径，而不是历史模板回退。
- `walk_forward_backtest.ridge_ml_path_mae_bps` 与 `ridge_ml_improvement_pct_vs_historical_profile` 是否持续改善。MAE 越低越好；改善率为负说明候选不如基准。
- `model_promotion_performed` 保持 `false`。候选暂不自动晋级，需要积累足够样本外证据。

## 这是不是会自动越用越准？

不能保证。流程会持续累积完整行情、结算历史预测、重新拟合 Ridge 候选并做滚动验证；但它不会通过“预测失败一次就自动惩罚某个参数”的简单规则来学习。只有在后续真实记录中持续胜过历史统计基准，才有理由考虑晋级。新闻目前主要用于市场背景与受限的实验性叠加，不应描述成模型已经学会了全部新闻影响。

## 没有新曲线时

若公开行情不可用、交易日不完整或历史样本不足，程序会保存失败原因，并可能输出明确标记为「历史统计情景」的内容包。此时视频不应把图二介绍成机器学习预测。如果出现 Alpaca 缺少凭据或 HTTP 401，那只影响原有 Alpaca 高级流程，不会自动影响公开行情流程；公开行情自身失败则需要查看日志中的数据源状态。

## 独立命令

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe .\scripts\run_public_intraday_learning_cycle.py
```

这条命令会写入 `reports/public_intraday_learning_cycle/latest_status.json`。日常使用建议仍从 GUI 主按钮启动，以便运行结束后同时生成图片和中文口播稿。
