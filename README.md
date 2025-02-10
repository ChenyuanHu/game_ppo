# PPO算法实现

这是一个简单的PPO（Proximal Policy Optimization）算法实现，用于强化学习入门学习。本项目使用CartPole-v1环境作为示例。

## 环境设置

推荐使用 conda 创建独立的环境：

```bash
# 创建新的conda环境
conda create -n ppo python=3.10
# 激活环境
conda activate ppo
```

## 安装依赖

在激活的环境中安装依赖：
```bash
pip install -r requirements.txt
```

## PPO算法核心概念

PPO算法是一种重要的策略梯度算法，其核心特点包括：

1. **近端策略优化**：通过限制新旧策略的差异来确保稳定的学习过程
2. **优势函数**：使用优势函数来评估动作的好坏，减少方差
3. **价值函数**：同时学习状态价值函数，用于计算优势
4. **策略裁剪**：使用裁剪机制来限制策略更新的幅度

## 项目结构

- `ppo.py`: PPO算法的核心实现
- `train.py`: 训练脚本
- `play.py`: 运行训练好的模型
- `requirements.txt`: 项目依赖

## 使用方法

1. 训练模型：
```bash
python train.py
```

2. 运行训练好的模型：
```bash
python play.py
```

## 代码结构说明

### PPO类的主要组件：

1. **策略网络**：输出动作概率分布
2. **价值网络**：评估状态价值
3. **get_action方法**：根据当前策略采样动作
4. **update方法**：使用收集的数据更新策略

### 训练过程：

1. 收集轨迹数据
2. 计算优势函数
3. 多次更新策略和价值网络
4. 记录和可视化训练过程

## 训练结果

训练过程会生成一个`training_curve.png`文件，显示奖励随时间的变化。模型参数将保存在`ppo_policy.pth`和`ppo_value.pth`文件中。

## IDE 设置

如果您使用 VS Code 或 Cursor 等 IDE：
1. 确保在 IDE 中选择了正确的 Python 解释器（与安装依赖的环境相同）
2. 对于 VS Code/Cursor：使用 Command Palette (Cmd+Shift+P) -> "Python: Select Interpreter" 选择正确的环境
3. 建议选择 conda 环境下的 Python 解释器 