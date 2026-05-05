
## 整体结构：三大部分 + 最终聚合

PPO 的总 loss 由公式（**ppo.py 第 282 行**）驱动：

```python
loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
```

在计算图中，这对应 **根节点** `SubBackward0`（ID: `1748843311408`），它执行最终的减法：

```
SubBackward0 = AddBackward0 - MulBackward0
             = (surrogate_loss + value_coef * value_loss) - (entropy_coef * entropy.mean())
```

其中：

- **左输入** `AddBackward0`（ID: `1748843311552`）= 策略损失 + 价值损失
- **右输入** `MulBackward0`（ID: `1748843311504`）= 熵系数 × 熵

---

## 分支 1：策略损失（Surrogate Loss）

**对应代码：ppo.py 第 265-271 行**

```python
ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
surrogate = -torch.squeeze(batch.advantages) * ratio
surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

**对应计算图节点链（从上到下 = 从 loss 到参数）：**

| 图节点 ID         | 节点类型                   | 对应代码行                                            | 数学含义                                            |
| ----------------- | -------------------------- | ----------------------------------------------------- | --------------------------------------------------- |
| `1748843311696` | **MeanBackward0**    | 第 271 行 `.mean()`                                 | `surrogate_loss = max(surrogate, clipped).mean()` |
| `1748843311744` | **MaximumBackward0** | 第 271 行 `torch.max(surrogate, surrogate_clipped)` | 取两者中的较大值（clip 机制）                       |

### 路径 A：未裁剪的 surrogate（左输入）

| 图节点 ID                   | 节点类型                 | 对应代码行                                             | 数学含义                                          |
| --------------------------- | ------------------------ | ------------------------------------------------------ | ------------------------------------------------- |
| `1748843311840`           | **MulBackward0**   | 第 267 行                                              | `surrogate = -advantages * ratio`               |
| `1748843311984`           | **ExpBackward0**   | 第 266 行 `torch.exp(...)`                           | `ratio = exp(log_prob_new - log_prob_old)`      |
| `1748843312080`           | **SubBackward0**   | 第 266 行                                              | `actions_log_prob - old_actions_log_prob`       |
| `1748843312176`           | **SumBackward1**   | 第 231 行 + distribution.py 第 240 行 `.sum(dim=-1)` | 对所有动作维度求和得到标量 log_prob               |
| `1748843312272`           | **SubBackward0**   | distribution.py `log_prob` 内部                      | `-0.5*z^2 - log(std)`（高斯对数概率的两个分量） |
| `1748843312368`           | **SubBackward0**   | distribution.py `log_prob` 内部                      | 二次项与 log(std) 的组合                          |
| `1748843312464`           | **DivBackward0**   | distribution.py `log_prob` 内部                      | `z = (action - mean) / std`（标准化）           |
| `1748842598496`           | **NegBackward0**   | distribution.py `log_prob` 内部                      | `-std`（用于除法运算）                          |
| `1748843312704`           | **PowBackward0**   | distribution.py `log_prob` 内部                      | `std^(-2)` 或类似幂运算                         |
| `1748843312800`           | **SubBackward0**   | distribution.py `log_prob` 内部                      | `action - mean`                                 |
| **`1748843312896`** | **AddmmBackward0** | **ppo.py 第 225-230 行 `self.actor(obs)`**     | **Actor 网络前向传播 → 输出 mean**         |

### 路径 B：裁剪后的 surrogate_clipped（右输入）

| 图节点 ID         | 节点类型                 | 对应代码行                                       | 数学含义                                                |
| ----------------- | ------------------------ | ------------------------------------------------ | ------------------------------------------------------- |
| `1748843311888` | **MulBackward0**   | 第 268-270 行                                    | `surrogate_clipped = -advantages * clamp(ratio, ...)` |
| `1748843312128` | **ClampBackward1** | 第 268-269 行 `torch.clamp(ratio, 1-ε, 1+ε)` | 裁剪 ratio                                              |
| `1748843311984` | **ExpBackward0**   | 第 266 行（与路径 A**共享**）              | 同一个 ratio                                            |

**关键：路径 A 和 B 在 `1748843311984`（ExpBackward0 / ratio）处汇合**，说明 ratio 被两条路径复用。

### 路径 A 中与 std 参数相关的子图

| 图节点 ID                   | 节点类型                  | 对应代码             | 数学含义                                                                     |
| --------------------------- | ------------------------- | -------------------- | ---------------------------------------------------------------------------- |
| `1748843312608`           | **MulBackward0**    | distribution.py 内部 | `std * ...` （标准化缩放）                                                 |
| `1748843312848`           | **PowBackward0**    | distribution.py 内部 | `std^2`                                                                    |
| **`1748843313088`** | **ExpandBackward0** | distribution.py 内部 | **Actor 的 std 参数**（展开到 batch 维度）                             |
| `1748843313136`           | **ClampBackward1**  | distribution.py 内部 | `clamp(std, min_val)` 数值稳定性                                           |
| `1748843313664`           | **AccumulateGrad**  | —                   | 收集 `clip_param` 常量 **`(4,)`**（即 num_actions 维度的 min_std） |
| `1748823952992`           | **参数 (4)**        | lightblue 叶子节点   | `clip_param` 常量                                                          |

### 路径 A 中 log(std) 子图

| 图节点 ID         | 节点类型                  | 对应代码                          | 数学含义     |
| ----------------- | ------------------------- | --------------------------------- | ------------ |
| `1748843312512` | **LogBackward0**    | distribution.py `log_prob` 内部 | `log(std)` |
| `1748843313088` | **ExpandBackward0** | 与上面**共享** std 参数     | 同一个 std   |

---

## 分支 2：价值损失（Value Loss）

**对应代码：ppo.py 第 274-278 行**（`use_clipped_value_loss=True` 时）

```python
value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
value_losses = (values - batch.returns).pow(2)
value_losses_clipped = (value_clipped - batch.returns).pow(2)
value_loss = torch.max(value_losses, value_losses_clipped).mean()
```

**对应计算图节点链：**

| 图节点 ID         | 节点类型                   | 对应代码行                   | 数学含义                                                        |
| ----------------- | -------------------------- | ---------------------------- | --------------------------------------------------------------- |
| `1748843311456` | **MulBackward0**     | 第 282 行                    | `value_loss_coef * value_loss`（与 surrogate 相加前先乘系数） |
| `1748843311792` | **MeanBackward0**    | 第 278 行 `.mean()`        | `value_loss = max(losses, clipped).mean()`                    |
| `1748843312320` | **MaximumBackward0** | 第 278 行 `torch.max(...)` | 取较大值（clip 机制）                                           |

### 路径 C：未裁剪的 value_losses（左输入）

| 图节点 ID                   | 节点类型                 | 对应代码行                                      | 数学含义                                     |
| --------------------------- | ------------------------ | ----------------------------------------------- | -------------------------------------------- |
| `1748843312032`           | **PowBackward0**   | 第 276 行 `.pow(2)`                           | `(values - returns)^2`                     |
| `1748843313520`           | **SubBackward0**   | 第 276 行                                       | `values - returns`                         |
| **`1748843313760`** | **AddmmBackward0** | **ppo.py 第 232 行 `self.critic(obs)`** | **Critic 网络前向传播 → 输出 values** |

### 路径 D：裁剪的 value_losses_clipped（右输入）

| 图节点 ID         | 节点类型                 | 对应代码行                    | 数学含义                                                              |
| ----------------- | ------------------------ | ----------------------------- | --------------------------------------------------------------------- |
| `1748843311936` | **PowBackward0**   | 第 277 行 `.pow(2)`         | `(value_clipped - returns)^2`                                       |
| `1748843313424` | **SubBackward0**   | 第 277 行                     | `value_clipped - returns`                                           |
| `1748843314432` | **AddBackward0**   | 第 275 行                     | `value_clipped = batch.values + (values - batch.values).clamp(...)` |
| `1748843314624` | **ClampBackward1** | 第 275 行 `.clamp(-ε, ε)` | 裁剪 value 变化量                                                     |
| `1748843314192` | **SubBackward0**   | 第 275 行                     | `values - batch.values`                                             |
| `1748843313760` | **AddmmBackward0** | 与路径 C**共享**        | 同一个 Critic 输出 values                                             |

**关键：路径 C 和 D 在 `1748843313760`（Critic 前向传播）处汇合**。

---

## 分支 3：熵正则化（Entropy Bonus）

**对应代码：ppo.py 第 235 行 + 第 282 行**

```python
entropy = self.actor.output_entropy[:original_batch_size]  # 第 235 行
loss = ... - self.entropy_coef * entropy.mean()              # 第 282 行
```

高斯分布的熵公式（distribution.py 第 228-230 行）：

```python
entropy = self._distribution.entropy().sum(dim=-1)
# H = 0.5 * log(2πeσ²) = 0.5 + 0.5*log(2π) + log(σ)，对所有动作维度求和
```

**对应计算图节点链：**

| 图节点 ID                   | 节点类型                  | 对应代码行                                 | 数学含义                                       |
| --------------------------- | ------------------------- | ------------------------------------------ | ---------------------------------------------- |
| `1748843311504`           | **MulBackward0**    | 第 282 行                                  | `entropy_coef * entropy.mean()`（作为减数）  |
| `1748843312224`           | **MeanBackward0**   | 第 282 行 `.mean()`                      | 对 batch 维度取平均                            |
| `1748843311648`           | **SliceBackward0**  | 第 235 行 `[:original_batch_size]`       | 切片（仅取原始样本，排除 symmetry 扩展）       |
| `1748843313904`           | **SumBackward1**    | distribution.py 第 230 行 `.sum(dim=-1)` | 对所有动作维度求和                             |
| `1748843314576`           | **AddBackward0**    | PyTorch Normal.entropy() 内部              | `0.5 + 0.5*log(2π) + log(std)` 的加法组合   |
| `1748843314720`           | **LogBackward0**    | PyTorch Normal.entropy() 内部              | `log(std)` 分量                              |
| **`1748843313088`** | **ExpandBackward0** | —                                         | **与 Surrogate 分支共享同一个 std 参数** |

---

## Actor 网络前向传播（梯度回传的关键路径）

**对应代码：ppo.py 第 225-230 行**

```python
self.actor(batch.observations, masks=batch.masks, 
           hidden_state=batch.hidden_states[0], stochastic_output=True)
```

计算图节点 `AddmmBackward0`（ID: `1748843312896`）代表 Actor MLP 的输出层，其完整的 3 层结构：

| 图节点 ID                   | 节点类型                 | 网络层                    | 参数形状               | 说明                                 |
| --------------------------- | ------------------------ | ------------------------- | ---------------------- | ------------------------------------ |
| **`1748843312896`** | **AddmmBackward0** | **输出层 Linear**   | —                     | `mean = Elu(hidden2) @ W3 + b3`    |
| `1748843312992`           | AccumulateGrad           | —                        | **`(4,)`**     | 输出层偏置 b3（num_actions=4）       |
| `1748843313376`           | AccumulateGrad           | —                        | **`(4, 32)`**  | 输出层权重 W3                        |
| `1748843312656`           | TBackward0               | —                        | `(4, 32)`            | W3 转置（反向传播用）                |
| `1748843313040`           | **EluBackward0**   | **ELU 激活**        | —                     | 第 2 个隐藏层后的激活函数            |
| **`1748843313184`** | **AddmmBackward0** | **隐藏层 2 Linear** | —                     | `hidden2 = Elu(hidden1) @ W2 + b2` |
| `1748843313280`           | AccumulateGrad           | —                        | **`(32,)`**    | 隐藏层 2 偏置 b2                     |
| `1748844584496`           | AccumulateGrad           | —                        | **`(32, 32)`** | 隐藏层 2 权重 W2                     |
| `1748843313232`           | TBackward0               | —                        | `(32, 32)`           | W2 转置                              |
| `1748843313328`           | **EluBackward0**   | **ELU 激活**        | —                     | 第 1 个隐藏层后的激活函数            |
| **`1748843313472`** | **AddmmBackward0** | **隐藏层 1 Linear** | —                     | `hidden1 = obs @ W1 + b1`          |
| `1748843313568`           | AccumulateGrad           | —                        | **`(32,)`**    | 隐藏层 1 偏置 b1                     |
| `1748843313712`           | AccumulateGrad           | —                        | **`(32, 8)`**  | 隐藏层 1 权重 W1（obs_dim=8）        |
| `1748843313616`           | TBackward0               | —                        | `(32, 8)`            | W1 转置                              |

---

## Critic 网络前向传播

**对应代码：ppo.py 第 232 行**

```python
values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
```

计算图节点 `AddmmBackward0`（ID: `1748843313760`）代表 Critic MLP 的输出层：

| 图节点 ID                   | 节点类型                 | 网络层                    | 参数形状               | 说明                                 |
| --------------------------- | ------------------------ | ------------------------- | ---------------------- | ------------------------------------ |
| **`1748843313760`** | **AddmmBackward0** | **输出层 Linear**   | —                     | `value = Elu(hidden2) @ W3 + b3`   |
| `1748843313808`           | AccumulateGrad           | —                        | **`(1,)`**     | 输出层偏置 b3（输出维度=1）          |
| `1748843314144`           | AccumulateGrad           | —                        | **`(1, 32)`**  | 输出层权重 W3                        |
| `1748843312944`           | TBackward0               | —                        | `(1, 32)`            | W3 转置                              |
| `1748843313856`           | **EluBackward0**   | **ELU 激活**        | —                     | 隐藏层 2 后的激活                    |
| **`1748843313952`** | **AddmmBackward0** | **隐藏层 2 Linear** | —                     | `hidden2 = Elu(hidden1) @ W2 + b2` |
| `1748843314048`           | AccumulateGrad           | —                        | **`(32,)`**    | 隐藏层 2 偏置 b2                     |
| `1748843314288`           | AccumulateGrad           | —                        | **`(32, 32)`** | 隐藏层 2 权重 W2                     |
| `1748843314000`           | TBackward0               | —                        | `(32, 32)`           | W2 转置                              |
| `1748843314096`           | **EluBackward0**   | **ELU 激活**        | —                     | 隐藏层 1 后的激活                    |
| **`1748843314240`** | **AddmmBackward0** | **隐藏层 1 Linear** | —                     | `hidden1 = obs @ W1 + b1`          |
| `1748843314336`           | AccumulateGrad           | —                        | **`(32,)`**    | 隐藏层 1 偏置 b1                     |
| `1748843314480`           | AccumulateGrad           | —                        | **`(32, 8)`**  | 隐藏层 1 权重 W1（obs_dim=8）        |
| `1748843314384`           | TBackward0               | —                        | `(32, 8)`            | W1 转置                              |

---

## 梯度回传路径总结图

```
loss (SubBackward0)                          ← ppo.py:282
├── AddBackward0: surrogate + value_coef*value_loss
│   ├── MeanBackward0 → MaximumBackward0     ← ppo.py:271 (surrogate_loss)
│   │   ├── MulBackward0 (-A*ratio)          ← ppo.py:267 (surrogate)
│   │   │   └── ExpBackward0 (ratio)         ← ppo.py:266
│   │   │       ├── SumBackward1 (log_prob)  ← ppo.py:231 + dist.py:240
│   │   │       │   └── [Gaussian log_prob 计算]
│   │   │       │       ├── SubBackward0 (action-mean) 
│   │   │       │       │   └── ★ Actor MLP (1748843312896)  ← ppo.py:225-230
│   │   │       │       │       → 梯度回传到 Actor 的 W1(32,8), W2(32,32), W3(4,32)
│   │   │       │       └── LogBackward0 (log_std)
│   │   │       │           └── ★ ExpandBackward0 (std)     ← 与 Entropy 共享!
│   │   │       └── old_actions_log_prob (无梯度，从 storage 读取)
│   │   └── MulBackward0 (-A*clamp(ratio))   ← ppo.py:268-270
│   │       └── ClampBackward0               ← ppo.py:268-269
│   │           └── [同上 ExpBackward0, 共享 ratio]
│   │
│   └── MulBackward0 (value_coef * value)    ← ppo.py:282 (系数乘法)
│       └── MeanBackward0 → MaximumBackward0 ← ppo.py:278 (value_loss)
│           ├── PowBackward0 ((V-R)²)        ← ppo.py:276
│           │   └── SubBackward0 (V-R)       ← ppo.py:276
│           │       └── ★ Critic MLP (1748843313760)  ← ppo.py:232
│           │           → 梯度回传到 Critic 的 W1(32,8), W2(32,32), W3(1,32)
│           └── PowBackward0 ((V_clip-R)²)   ← ppo.py:277
│               └── SubBackward0 (V_clip-R)
│                   └── AddBackward0 (V_clip)← ppo.py:275
│                       └── ClampBackward0
│                           └── [同上 Critic MLP, 共享 values]
│
└── MulBackward0 (entropy_coef * entropy)    ← ppo.py:282 (作为减数)
    └── MeanBackward0                        ← ppo.py:282 (.mean())
        └── SliceBackward0                   ← ppo.py:235 ([:original_batch_size])
            └── SumBackward1                 ← dist.py:230 (.sum(dim=-1))
                └── AddBackward0 (entropy公式)
                    └── LogBackward0 (log_std)
                        └── ★ ExpandBackward0 (std)  ← 与 Surrogate 共享!
```

---

## 参数梯度归属总结

| 参数      | 形状         | 来源                                      | 梯度来自哪些 loss 组件                                         |
| --------- | ------------ | ----------------------------------------- | -------------------------------------------------------------- |
| Actor W1  | `(32, 8)`  | `1748823953152`                         | **Surrogate loss**（通过 log_prob → mean）              |
| Actor W2  | `(32, 32)` | `1748823953072`                         | **Surrogate loss**                                       |
| Actor W3  | `(4, 32)`  | `1748823953312`                         | **Surrogate loss**                                       |
| Actor b1  | `(32,)`    | `1748823952912`                         | **Surrogate loss**                                       |
| Actor b2  | `(32,)`    | `1748823953232`                         | **Surrogate loss**                                       |
| Actor b3  | `(4,)`     | `1748823953392`                         | **Surrogate loss**                                       |
| Actor std | `(4,)`     | `1748843313088` (via `1748823952992`) | **Surrogate loss** + **Entropy**（被两条路径共享） |
| Critic W1 | `(32, 8)`  | `1748823952752`                         | **Value loss**                                           |
| Critic W2 | `(32, 32)` | `1748823953552`                         | **Value loss**                                           |
| Critic W3 | `(1, 32)`  | `1748823953712`                         | **Value loss**                                           |
| Critic b1 | `(32,)`    | `1748823953472`                         | **Value loss**                                           |
| Critic b2 | `(32,)`    | `1748823953632`                         | **Value loss**                                           |
| Critic b3 | `(1,)`     | `1748823953792`                         | **Value loss**                                           |

**关键发现：**

1. **Actor 的 mean 参数**（W1~W3, b1~b3）仅通过 **Surrogate loss** 接收梯度
2. **Actor 的 std 参数**同时通过 **Surrogate loss**（log_prob 中的 `log(std)` 项）和 **Entropy**（`entropy = 0.5*log(2πe*std²)`）接收梯度
3. **Critic 的全部参数**（W1~W3, b1~b3）仅通过 **Value loss** 接收梯度
4. 这解释了为什么 `ppo.py` 第 94-96 行中 Actor 和 Critic 共享同一个 optimizer——因为 `loss.backward()` 会自动将梯度分派到各自参数上
