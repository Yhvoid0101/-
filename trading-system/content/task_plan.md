# 零降级零模拟根治计划

目标：移除多模型集成中的 LightGBM/Prophet 模拟回退，漂移模型从活动集合硬阻断，融合只接受 available=true，所有模型不可用时明确不可交易。

阶段：
- [x] 定位 DynamicWeightManager、LightGBM、Prophet、融合器与验证入口
- [ ] 创建并运行失败测试，确认红灯
- [ ] 最小实现根治逻辑
- [ ] 运行目标测试、py_compile、项目检查与真实短沙盘
- [ ] 汇总验证证据，不提交 Git

约束：保留真实训练/推理路径；禁止 mock、fallback、degrading 结果参与交易；编辑前执行敏感路径模式检查。
