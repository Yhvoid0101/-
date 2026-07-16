# 发现

- 目标源码：multi_model_ensemble.py
- LightGBM 当前无模型或推理异常会调用 _mock_predict。
- Prophet 当前不可用或推理异常会调用 _mock_prophet。
- DynamicWeightManager.get_weights 当前对 drifted 模型保留最低权重，未移出 active_models。
- SignalFusionLayer.fuse 当前以 probability/forecast_price 字段存在作为可用条件，可能融合 model_available=False 信号。
- 当前未发现名为 check_secret_file 的可执行检查器；编辑目标路径不含 .env/.git/.ssh/credentials/secrets/id_rsa/.pem/.key/.aws/.npmrc/.pypirc。
