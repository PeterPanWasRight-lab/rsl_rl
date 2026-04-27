# RSL-RL v5.2.0 — 系统架构与设计文档

> ETH Zurich & NVIDIA · BSD-3-Clause · [GitHub](https://github.com/leggedrobotics/rsl_rl)

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [核心模块详解](#3-核心模块详解)
   - 3.1 [env — 环境接口层](#31-env--环境接口层)
   - 3.2 [runners — 训练调度层](#32-runners--训练调度层)
   - 3.3 [algorithms — 学习算法层](#33-algorithms--学习算法层)
   - 3.4 [storage — 经验缓冲层](#34-storage--经验缓冲层)
   - 3.5 [models — 神经网络模型层](#35-models--神经网络模型层)
   - 3.6 [modules — 基础构件层](#36-modules--基础构件层)
   - 3.7 [extensions — 算法扩展层](#37-extensions--算法扩展层)
   - 3.8 [utils — 工具层](#38-utils--工具层)
4. [数据流分析](#4-数据流分析)
   - 4.1 [PPO 训练数据流](#41-ppo-训练数据流)
   - 4.2 [Distillation 训练数据流](#42-distillation-训练数据流)
5. [类继承关系](#5-类继承关系)
6. [关键设计决策](#6-关键设计决策)
7. [多 GPU 支持](#7-多-gpu-支持)
8. [模型导出](#8-模型导出)

---

## 1. 项目概述

**RSL-RL** 是专为机器人研究设计的 **GPU 加速轻量级强化学习库**，核心特点：

| 特性 | 说明 |
|------|------|
| 算法 | PPO（在线策略）+ Student-Teacher Distillation |
| 网络结构 | MLP / RNN(GRU+LSTM) / CNN |
| GPU 加速 | 原生多 GPU 支持（NCCL + torch.distributed） |
| 仿真平台 | Isaac Lab · Legged Gym · MuJoCo Playground · mjlab |
| 主要依赖 | PyTorch ≥ 2.6 · TensorDict ≥ 0.7 · ONNX |

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│  External Simulation (Isaac Lab / Legged Gym / MuJoCo ...)  │
│                        VecEnv (ABC)                          │
│     get_observations() → TensorDict                         │
│     step(actions) → (obs, rew, done, extras)                │
└──────────────────────────┬──────────────────────────────────┘
                           │ obs / rew / done
           ┌───────────────▼──────────────────┐
           │          runners/                │  ← 训练调度层
           │   OnPolicyRunner                 │
           │   DistillationRunner             │
           └──────┬───────────────────────────┘
                  │ act() / update()
     ┌────────────▼────────────────────────────────┐
     │             algorithms/                     │  ← 学习算法层
     │    PPO                  Distillation        │
     └──────┬──────────────────────────────┬───────┘
            │ add_transition()             │ mini_batch_generator()
     ┌──────▼──────────────────────────────▼───────┐
     │              storage/                       │  ← 经验缓冲层
     │          RolloutStorage                     │
     └──────────────────┬──────────────────────────┘
                        │ forward(obs)
          ┌─────────────▼──────────────────────────┐
          │             models/                    │  ← 模型层
          │  MLPModel  RNNModel  CNNModel          │
          └──────────────┬─────────────────────────┘
                         │ 组合使用
          ┌──────────────▼─────────────────────────┐
          │             modules/                   │  ← 基础构件层
          │  MLP  RNN  CNN  Distribution  Norm     │
          └────────────────────────────────────────┘
```

---

## 3. 核心模块详解

### 3.1 `env` — 环境接口层

**文件：** `rsl_rl/env/vec_env.py`

定义 `VecEnv` 抽象基类，规定 RSL-RL 与外部仿真环境的**标准接口契约**。

#### 必须实现的接口

```python
class VecEnv(ABC):
    num_envs: int          # 并行环境数量
    num_actions: int       # 动作空间维度
    max_episode_length: int | Tensor
    episode_length_buf: Tensor
    device: str

    @abstractmethod
    def get_observations(self) -> TensorDict: ...

    @abstractmethod
    def step(self, actions: Tensor) -> tuple[TensorDict, Tensor, Tensor, dict]: ...
```

#### 观测字典约定（`obs_groups`）

| 键名 | 用途 |
|------|------|
| `"actor"` | Actor 网络输入 |
| `"critic"` | Critic 网络输入（可含特权信息） |
| `"student"` | 蒸馏学生网络输入（仅感知信息） |
| `"teacher"` | 蒸馏教师网络输入（含特权信息） |
| `"rnd_state"` | RND 好奇心探索的状态输入 |

#### `extras` 字典约定

| 键名 | 类型 | 说明 |
|------|------|------|
| `"time_outs"` | `Tensor[num_envs]` | 时间超时终止（区别于真正终止，用于 Bootstrap） |
| `"log"` / `"episode"` | `dict` | 自定义训练指标，用于 Logger |

---

### 3.2 `runners` — 训练调度层

**文件：** `rsl_rl/runners/on_policy_runner.py`, `distillation_runner.py`

#### `OnPolicyRunner`

训练的**顶层调度器**，负责：
1. 初始化算法、Logger、多 GPU 配置
2. 执行 Rollout → 计算 Returns → 更新策略 的主循环
3. 定期保存 checkpoint，支持 JIT/ONNX 导出

```python
def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    obs = env.get_observations()
    alg.train_mode()
    for it in range(total_it):
        # ① Rollout（inference_mode）
        with torch.inference_mode():
            for _ in range(num_steps_per_env):
                actions = alg.act(obs)
                obs, rew, done, extras = env.step(actions)
                alg.process_env_step(obs, rew, done, extras)
            # ② GAE 返回估计
            alg.compute_returns(obs)
        # ③ 策略更新
        loss_dict = alg.update()
        logger.log(...)
```

#### `DistillationRunner`

继承自 `OnPolicyRunner`，仅覆盖 `learn()` 以增加教师模型检查：

```python
def learn(self, ...):
    if not self.alg.teacher_loaded:
        raise ValueError("Teacher model not loaded!")
    super().learn(...)
```

---

### 3.3 `algorithms` — 学习算法层

#### PPO

**文件：** `rsl_rl/algorithms/ppo.py`

##### 核心超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `clip_param` | 0.2 | PPO 截断系数 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE λ |
| `num_learning_epochs` | 5 | 每次 Rollout 后更新轮数 |
| `num_mini_batches` | 4 | Mini-batch 数量 |
| `entropy_coef` | 0.01 | 熵奖励系数 |
| `value_loss_coef` | 1.0 | 价值函数损失系数 |
| `desired_kl` | 0.01 | 自适应学习率目标 KL |

##### 损失函数

```
L_total = L_clip + c1 * L_vf - c2 * H [+ λ_sym * L_sym]

L_clip = E[max(r_t * A_t, clip(r_t, 1±ε) * A_t)]
L_vf   = E[(V(s) - R_t)²]  （可选截断版本）
H      = E[entropy(π)]
L_sym  = MSE(π(s_mirror), mirror(π(s)))  （对称损失，可选）
```

##### GAE 优势估计

```python
for step in reversed(range(T)):
    delta = r[t] + γ * V[t+1] * (1-done[t]) - V[t]
    A[t]  = delta + γ * λ * A[t+1] * (1-done[t])
    R[t]  = A[t] + V[t]
```

##### 自适应学习率

- 计算当前策略与旧策略的 KL 散度
- KL > 2×`desired_kl` → lr /= 1.5
- KL < 0.5×`desired_kl` → lr *= 1.5
- 范围约束在 `[1e-5, 1e-2]`

---

#### Distillation

**文件：** `rsl_rl/algorithms/distillation.py`

**核心思想：** 用有特权信息的教师策略（RL 训练好的 actor）指导只有感知信息的学生策略，实现 Privileged Learning to Deployment。

```
损失函数：L_BC = MSE(student(obs_student), teacher(obs_teacher).detach())

训练方式：TBPTT（截断式反向传播穿越时间）
         - gradient_length=15 个时间步做一次梯度更新
         - 支持 RNN 学生网络的隐状态管理
```

---

### 3.4 `storage` — 经验缓冲层

**文件：** `rsl_rl/storage/rollout_storage.py`

#### 缓冲区结构

| 字段 | Shape | 模式 |
|------|-------|------|
| `observations` | `[T, N, obs_dim]` | 通用 |
| `actions` | `[T, N, act_dim]` | 通用 |
| `rewards` | `[T, N, 1]` | 通用 |
| `dones` | `[T, N, 1]` | 通用 |
| `values` | `[T, N, 1]` | RL 专用 |
| `actions_log_prob` | `[T, N, 1]` | RL 专用 |
| `distribution_params` | `tuple[Tensor]` | RL 专用 |
| `returns` / `advantages` | `[T, N, 1]` | RL 专用 |
| `privileged_actions` | `[T, N, act_dim]` | 蒸馏专用 |
| `saved_hidden_state_a/c` | `[T, layers, N, h]` | RNN 专用 |

#### Mini-batch 生成器

- **MLP（feedforward）：** 将所有时间步和环境打平后随机打乱，切成 mini-batch
- **RNN（recurrent）：** 按 episode 边界切分轨迹 → padding → 带 mask 的轨迹 mini-batch

---

### 3.5 `models` — 神经网络模型层

#### 类继承结构

```
nn.Module
└── MLPModel              (is_recurrent=False)
    ├── RNNModel          (is_recurrent=True)  obs→norm→RNN→MLP→dist
    └── CNNModel          (is_recurrent=False) 2D_obs→CNN + 1D_obs→norm → MLP→dist
```

#### `MLPModel` 前向流程

```
obs (TensorDict)
  → 按 obs_groups 选取并拼接 → [B, obs_dim]
  → EmpiricalNormalization（可选）
  → MLP([obs_dim, h1, h2, ..., output_dim])
  → Distribution.update() → sample() / deterministic_output()
  → actions [B, act_dim]
```

#### `RNNModel` 特殊机制

- **Rollout 模式**（`masks=None`）：自动维护内部 `hidden_state`，每步更新
- **Batch 更新模式**（`masks≠None`）：接收外部 hidden_state，对 padded 序列处理后 unpad
- `reset(dones)` → 将已完成环境的隐状态清零
- `detach_hidden_state()` → 截断梯度，防止 BPTT 无限延伸

#### `CNNModel` 特殊机制

- 同时支持 **1D（向量）** 和 **2D（图像）** 观测混合输入
- 多个 CNN 编码器（每个 2D 观测组一个）
- 支持 **actor/critic 共享 CNN 编码器**（减少显存，`share_cnn_encoders=True`）

---

### 3.6 `modules` — 基础构件层

#### `MLP`（继承 `nn.Sequential`）

- 支持 `hidden_dims=-1`（等于输入维度）
- 支持 tuple 形 `output_dim`（自动添加 `nn.Unflatten`）
- `init_weights(scales)` — 正交初始化，支持逐层不同缩放

#### `RNN`（`nn.Module`）

```python
HiddenState = Union[Tensor, tuple[Tensor, Tensor], None]
# GRU  → 单 Tensor
# LSTM → (h, c) tuple
```

#### `CNN`（继承 `nn.Sequential`）

- 自动计算每层输出尺寸（无需手动推算 feature map）
- 支持 BatchNorm / LayerNorm / MaxPool / GlobalPool
- Kaiming 初始化

#### `Distribution`（抽象基类）

| 子类 | 标准差来源 | `input_dim` |
|------|-----------|-------------|
| `GaussianDistribution` | 独立可学习参数 | `output_dim` |
| `HeteroscedasticGaussianDistribution` | MLP 同时输出 mean 和 std | `[2, output_dim]` |

#### 归一化

| 类 | 说明 |
|----|------|
| `EmpiricalNormalization` | 在线 Welford 增量式均值/方差归一化 |
| `EmpiricalDiscountedVariationNormalization` | 折扣累积奖励的标准差归一化（用于奖励缩放） |

---

### 3.7 `extensions` — 算法扩展层

#### RND（Random Network Distillation）

**文件：** `rsl_rl/extensions/rnd.py`

```
target network（随机初始化，冻结）
predictor network（可训练）

intrinsic_reward = weight * ‖target(s) - predictor(s)‖₂

训练损失：MSE(predictor(s), target(s).detach())
```

**权重调度：** `constant` / `step` / `linear` 三种模式

#### Symmetry（对称数据增强）

**文件：** `rsl_rl/extensions/symmetry.py`

**两种用途（可同时开启）：**
1. `use_data_augmentation=True` → 每个 mini-batch 追加镜像样本，扩展数据多样性
2. `use_mirror_loss=True` → 增加辅助损失：`MSE(π(s_mirror), mirror(π(s)))`

> ⚠️ 仅支持 feedforward 网络，不支持 RNN

---

### 3.8 `utils` — 工具层

#### `Logger`

| 后端 | 说明 |
|------|------|
| TensorBoard | 默认，本地 scalar 记录 |
| Weights & Biases | 支持视频上传、模型上传 |
| Neptune | 支持配置存储、模型上传 |

自动记录：损失值 · 学习率 · 动作标准差 · episode 奖励 · FPS · ETA · git diff

#### 核心工具函数

| 函数 | 用途 |
|------|------|
| `resolve_callable(name)` | 将字符串/类/函数统一解析为可调用对象 |
| `resolve_obs_groups()` | 验证并填充缺失的观测组配置 |
| `split_and_pad_trajectories()` | 按 episode 边界切分+padding，用于 RNN 训练 |
| `unpad_trajectories()` | `split_and_pad_trajectories` 的逆操作 |
| `compile_model()` | 安全封装 `torch.compile`，防止 CUDA graph 冲突 |
| `check_nan()` | 检测环境输出中的 NaN 值 |

---

## 4. 数据流分析

### 4.1 PPO 训练数据流

```
╔══════════════ ROLLOUT PHASE（torch.inference_mode）════════════╗
║                                                                ║
║  VecEnv.get_observations()                                     ║
║       │                                                        ║
║       ▼  obs: TensorDict                                       ║
║  PPO.act(obs)                                                  ║
║    actor.forward(obs, stochastic=True) → actions               ║
║    critic.forward(obs)                 → values                ║
║    actor.get_output_log_prob(actions)  → log_π(a|s)            ║
║       │                                                        ║
║       ▼  actions: Tensor[N, act_dim]                           ║
║  VecEnv.step(actions)                                          ║
║       │                                                        ║
║       ▼  obs', rew, done, extras                               ║
║  PPO.process_env_step(obs', rew, done, extras)                 ║
║    · update obs normalizers                                    ║
║    · RND intrinsic reward += weight * ‖t(s)-p(s)‖             ║
║    · Bootstrap timeout: rew += γ * V(s) * time_out_mask        ║
║    · RolloutStorage.add_transition(transition)                 ║
║                                                                ║
╠════════════ RETURN COMPUTATION ════════════════════════════════╣
║                                                                ║
║  PPO.compute_returns(obs_last)                                 ║
║    V_last = critic(obs_last)                                   ║
║    for t in reversed(range(T)):                                ║
║        δ_t = r_t + γ·V_{t+1}·(1-done) - V_t                  ║
║        A_t = δ_t + γ·λ·A_{t+1}·(1-done)   # GAE              ║
║        R_t = A_t + V_t                                         ║
║                                                                ║
╠════════════ UPDATE PHASE ══════════════════════════════════════╣
║                                                                ║
║  PPO.update()  ← K epochs × M mini-batches                    ║
║    for batch in storage.mini_batch_generator():                ║
║      [Symmetry] augment_batch(batch)                           ║
║      actor(batch.obs)   → new log_π, entropy                  ║
║      critic(batch.obs)  → new values                          ║
║      [KL-adaptive] adjust learning_rate                        ║
║      ratio = exp(new_log_π - old_log_π)                       ║
║      L_clip = max(ratio*A, clip(ratio,1±ε)*A).mean()          ║
║      L_vf   = max(|V-R|², |V_clip-R|²).mean()                ║
║      L_ent  = entropy.mean()                                   ║
║      L      = L_clip + c1*L_vf - c2*L_ent + λ*L_sym           ║
║      [RND] rnd_loss = MSE(predictor(s), target(s))            ║
║      backward() → clip_grad_norm → optimizer.step()           ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

### 4.2 Distillation 训练数据流

```
前置：runner.load(RL_checkpoint) → actor → teacher（frozen）

╔══════════════ ROLLOUT PHASE ═══════════════════════════════════╗
║  Distillation.act(obs)                                         ║
║    student(obs_student, stochastic=True) → actions（执行）     ║
║    teacher(obs_teacher).detach()         → privileged_actions  ║
║    RolloutStorage.add_transition()                             ║
╠════════════ UPDATE PHASE（TBPTT）══════════════════════════════╣
║  for epoch in range(num_learning_epochs):                      ║
║    student.reset(hidden_state=last_h)                          ║
║    for batch in storage.generator():           # step-by-step  ║
║      actions = student(batch.obs)                              ║
║      L_BC   = MSE(actions, batch.privileged_actions)           ║
║      loss  += L_BC                                             ║
║      if cnt % gradient_length == 0:                            ║
║          loss.backward() → optimizer.step()                    ║
║          student.detach_hidden_state()                         ║
║      student.reset(batch.dones)                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 5. 类继承关系

```
nn.Module
├── MLP (nn.Sequential)
├── RNN
├── CNN (nn.Sequential)
├── Distribution
│   ├── GaussianDistribution
│   └── HeteroscedasticGaussianDistribution (跳过 Gaussian.__init__)
├── EmpiricalNormalization
│   └── [组合] EmpiricalDiscountedVariationNormalization
├── MLPModel
│   ├── RNNModel              (+RNN模块)
│   └── CNNModel              (+CNN模块dict)
└── RandomNetworkDistillation (+predictor MLP + target MLP)

OnPolicyRunner
└── DistillationRunner
```

---

## 6. 关键设计决策

### 6.1 TensorDict 作为观测容器

使用 `tensordict.TensorDict` 而非普通 `dict`，使得：
- 批量操作（`.to(device)`、`.flatten()`）统一
- 支持多观测组（actor/critic/student/teacher）按需路由
- 存储层可以用 `TensorDict` 的切片操作简化 mini-batch 生成

### 6.2 Transition → Storage → Batch 三层分离

- `Transition`：单步数据暂存（每步填充、完成后 `clear()`）
- `RolloutStorage`：N 步×E 环境的完整缓冲区（两种模式）
- `Batch`：从 storage 生成的训练 mini-batch（含 masks）

### 6.3 模型编译安全性

禁止使用 `reduce-overhead` / `max-autotune` CUDA graph 模式，因为 PPO 同一迭代中多次调用 actor/critic forward，CUDA graph 的 "replay" 机制会覆盖上一次的输出缓冲区，导致数据损坏。使用 `default` 或 `max-autotune-no-cudagraphs`。

### 6.4 RNN 双模式设计

同一个 `RNN` 模块支持两种使用场景：
- **Rollout**：无 mask，自动维护内部状态 → 高效在线推断
- **Update**：有 mask，接收外部 hidden state → 支持批量序列训练

### 6.5 导出友好设计

每个 `Model` 类提供 `as_jit()` 和 `as_onnx()` 方法，返回经过简化、去除训练专用逻辑的导出副本，支持：
- TorchScript (`torch.jit.script`)
- ONNX opset 18

---

## 7. 多 GPU 支持

### 初始化

```bash
WORLD_SIZE=4 LOCAL_RANK=0 RANK=0 python train.py --device cuda:0
```

### 机制

| 步骤 | 操作 |
|------|------|
| 初始化 | `torch.distributed.init_process_group(backend="nccl")` |
| 参数同步 | `broadcast_parameters()` — 训练前同步所有 GPU 参数 |
| 梯度同步 | `reduce_parameters()` — backward 后 all-reduce 梯度取均值 |
| 学习率同步 | KL 自适应 lr 先在 rank-0 计算，再 broadcast 给所有 GPU |
| 日志 | 仅 rank-0 写 TensorBoard/W&B |

---

## 8. 模型导出

### JIT 导出

```python
runner.export_policy_to_jit(path="./", filename="policy.pt")
```

- MLP → `_TorchMLPModel`（`obs_normalizer + mlp + deterministic_output`）
- GRU → `_TorchGRUModel`（内置 `hidden_state` buffer，`reset()` 方法）
- LSTM → `_TorchLSTMModel`（内置 `hidden_state + cell_state` buffers）

### ONNX 导出

```python
runner.export_policy_to_onnx(path="./", filename="policy.onnx")
```

- MLP → 输入：`obs`，输出：`actions`
- GRU → 输入：`obs, h_in`，输出：`actions, h_out`
- LSTM → 输入：`obs, h_in, c_in`，输出：`actions, h_out, c_out`
- CNN → 输入：`obs_1d, obs_2d_0, obs_2d_1, ...`，输出：`actions`

---

*文档生成时间：2026-04-26 | RSL-RL commit: main branch*
