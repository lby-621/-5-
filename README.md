项目名称：多变量时间序列预测 —— 5种算法对比与最优模型筛选

项目背景：基于2011-2025年月度业务数据（含5个自变量），构建时间序列预测模型，
         使用上月自变量预测当月目标变量Y，禁止同期拟合。

技术方案：
• 数据预处理：平稳性检验、缺失值处理、时间顺序8:2划分
• 特征工程：构造12期滞后项、滚动窗口统计（3/6期）、周期性正弦/余弦编码
• 模型对比：ARIMA（基线）、Random Forest、XGBoost、LightGBM、MLP神经网络
• 评估指标：RMSE、MAPE、R²

核心成果：
• LightGBM表现最优：RMSE=0.691, MAPE=8.18%, R²=0.91
• 关键有效变量：X5_lag1、X1_lag1、X5_roll6_std
• 相比ARIMA基线，RMSE降低44.5%

技术栈：Python, Pandas, Scikit-learn, XGBoost, LightGBM, Statsmodels, Matplotlib
