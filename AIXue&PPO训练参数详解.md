# AIXue/PPO训练参数详解

### 1. `--total_episodes 1024`
**作用**：总训练轮数/回合数
- **含义**：整个训练过程中总共进行1024个episode
- **影响**：控制训练的总时长，值越大训练时间越长
- **建议**：根据数据集大小调整，通常设置为数据集大小的1-10倍

### 2. `--num_ppo_epochs 4`
**作用**：每个episode内的PPO更新轮数
- **含义**：每个episode收集到数据后，进行4轮PPO策略更新
- **影响**：控制策略更新的频率，值越大策略更新越充分
- **建议**：通常设置为2-10，过大会导致过拟合

### 3. `--local_rollout_forward_batch_size 4`
**作用**：本地rollout时的前向传播批次大小
- **含义**：在生成回复时，每次处理4个样本
- **影响**：控制内存使用和生成速度
- **建议**：根据GPU内存调整，内存越大可以设置越大

### 4. `--missing_eos_penalty 1.0`
**作用**：缺失EOS token的惩罚系数
- **含义**：当模型生成没有正确结束符(EOS)的回复时，给予1.0的惩罚
- **影响**：鼓励模型生成完整、有明确结束的回复
- **建议**：通常设置为0.5-2.0，值越大惩罚越重

### 5. `--kl_coef 0.2`
**作用**：KL散度系数
- **含义**：控制策略模型与参考模型之间的KL散度惩罚强度
- **影响**：防止策略偏离参考模型太远，保持生成质量
- **建议**：通常设置为0.1-0.5，值越大约束越强

## 参数关系图

```
total_episodes (1024)
    ↓
每个episode收集数据
    ↓
num_ppo_epochs (4) - 对收集的数据进行4轮PPO更新
    ↓
local_rollout_forward_batch_size (4) - 每次生成4个样本
    ↓
missing_eos_penalty (1.0) - 对无EOS的回复进行惩罚
    ↓
kl_coef (0.2) - 限制策略与参考模型的差异
```

## 调优建议

### 内存优化
- 如果GPU内存不足，减小 `local_rollout_forward_batch_size`
- 如果训练不稳定，增加 `kl_coef`

### 训练效果优化
- 如果收敛太慢，增加 `num_ppo_epochs`
- 如果过拟合，减少 `total_episodes`
- 如果生成质量差，调整 `missing_eos_penalty`

### 当前配置分析
```bash
--total_episodes 1024        # 中等训练量
--num_ppo_epochs 4           # 适中的更新频率
--local_rollout_forward_batch_size 4  # 适合32B模型的内存
--missing_eos_penalty 1.0    # 标准惩罚强度
--kl_coef 0.2               # 适中的KL约束
```

这个配置对于32B模型来说是相对保守和稳定的设置，适合初次训练。