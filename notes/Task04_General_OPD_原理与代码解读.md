# Task4 · General OPD 原理与代码解读

> **课程**：Datawhale Agentic RL Lab · LAB 05「蒸馏与自学习」
> **教程**：[General OPD — 让学生在自己的轨迹上学习](https://datawhalechina.github.io/agentic-rl-lab/experiments/general-opd/)
> **开源仓库**：[datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab)
> **笔记作者**：李梓鹏　|　**日期**：2026-11-26　|　**形态**：原理推导 + 代码逐段精读 + 工程陷阱 + 批判性评估
> **引用规范**：本文所有外部结论均在文末「参考资料」列出来源链接，正文中以 `[编号]` 标注。

---

## 0. 一页速览

| 问题 | 答案 |
|---|---|
| OPD 一句话是什么 | 轨迹由**当前 student** 生成，**teacher 只负责给这条轨迹逐 token 打分**，再用**反向 KL** 把 student 拉向 teacher |
| 它解决什么痛点 | SFT/离线蒸馏是 off-policy + 密集信号（训练/推理分布偏移）；RL 是 on-policy + 稀疏信号（每回合只有 1 比特）。OPD 是 **on-policy + 密集信号** |
| 核心公式 | `r_t^KL = log π_θ(y_t\|x,y_<t) − log π_T(y_t\|x,y_<t)`，`A_t = −β · r_t^KL` |
| 代码只有几行 | `reverse_kl = student_lps − teacher_lps; advantages = −β * reverse_kl` |
| 最硬的约束 | **student 与 teacher 必须共享兼容 tokenizer**（信号是 token ID 级 logprob 差） |
| 最小实验成本 | DeepMath-103K 单分片 + Qwen3.5-4B(LoRA r32) + Qwen3.8-27B teacher，先跑 `--steps 1 --group-size 1` smoke test |
| 我最大的收获 | OPD 不是"新算法"，而是**把 RL 训练循环里的 reward 换成了一个 frozen teacher 的 logprob**——它把"奖励建模"这件最难的事，偷换成了"调用一次前向" |

**一句话记忆**：SFT 是**背老师写好的答案**；RL 是**自己下棋只知道输赢**；OPD 是**老师给学生的每一步棋打分**。

---

## 1. 定位：OPD 在三种后训练范式中的坐标

LLM 后训练对"学生模型"的处理方式，可以按两个正交维度切开：**轨迹从哪来**（on-policy / off-policy）和**信号密度**（密集 / 稀疏）。OPD 恰好填上了那个空位。

| 方法 | 采样方式 | 奖励信号 | 每回合信息量 | 典型代表 |
|---|---|---|---|---|
| 监督微调（SFT）/ 离线蒸馏 | off-policy | 密集 | O(N) bit | Alpaca、OpenThoughts-3 |
| 强化学习（RL/GRPO） | on-policy | 稀疏 | **O(1) bit** | PPO、GRPO、DAPO |
| **同策略蒸馏（OPD）** | **on-policy** | **密集** | **O(N) bit** | GKD、MiniLLM、Tinker OPD |

> 表格框架来自 Thinking Machines Lab 2025-10-27 的博文 [1]，其中"RL 每回合只传授 O(1) 比特、蒸馏每回合 O(N) 比特"的论断出自其同系列文章《LoRA Without Regret》。

### 1.1 两种失败模式，正好互补

**off-policy 失败在"状态分布偏移"**。学生学的是 teacher 常遇到的上下文 `x ~ π_T`，而部署时它要走的是自己的上下文 `x ~ π_θ`。一旦学生在早期犯了一个 teacher 从不犯的错，它就漂进了训练时从没见过的状态空间，且越走越远——这就是 **exposure bias**（暴露偏差），在长序列上尤其致命。这就是 DAgger [2] 早在 2010 年就指出的问题：off-policy 模仿学习的误差上界是 `O(εT²)`，而不断在 learner 实际访问的状态上查询专家，可以降到 `O(εT)`。

**on-policy RL 失败在"信号太稀"**。学生答错了一道 512 token 的数学题，只得到一个 `reward=0`。它知道了"这条路错了"，但不知道错在**哪一步**：是运算顺序错了？还是算错了？还是读题错了？功劳分配（credit assignment）无法完成。

OPD 的定位就一句话：**保留 RL 的 on-policy 采样，把稀疏的 outcome reward 换成 teacher 的密集 token 级 logprob**。

### 1.2 一个不会忘的类比

国际象棋 [1]：

- **on-policy RL** = 自己下棋，只知道最后赢/输，不知道哪一步是败着 → 反馈直接但极稀。
- **off-policy 蒸馏** = 看特级大师下棋，招法极妙，但那些局面新手自己根本遇不到 → 反馈密集但错位。
- **OPD** = 老师坐在旁边，**看你自己下的每一步棋**，从"严重失误"到"妙手"逐手打分。

这个类比精确对应了公式里的下标：期望是 `E_{x ∼ π_θ}`（学生自己的状态），监督来自 `log π_T`（老师的偏好）。

---

## 2. 理论溯源：OPD 不是 2025 年才有的

很多人以为 OPD 是 Thinking Machines 2025 年 10 月那篇博文 [1] 提出的。其实它有三条独立的技术脉络在 2023 年就已经成型。

| 时间 | 工作 | 贡献 | 与本 lab 代码的关系 |
|---|---|---|---|
| 2010 | **DAgger**（Ross et al.）[2] | 交互式模仿学习；把误差从 `O(εT²)` 降到 `O(εT)` | OPD 的**理论祖先**，解释了"为什么要 on-policy" |
| 2023-06 | **GKD**（Agarwal et al.）[3] | 统一下一 token 蒸馏框架：前向 KL / 反向 KL / JSD 可切换；学生**在自己的序列上**接受老师反馈 | 提供**divergence 选择**的理论依据 |
| 2023-06 | **MiniLLM**（Gu et al.）[4] | 系统论证生成式模型该用 **reverse KL**；序列级目标 + REINFORCE + 若干方差缩减技巧 | 提供**为什么是反向 KL** 的核心论证 |
| 2025-05 | **Qwen3 技术报告** [5] | 报告 OPD 以约 1/10 GPU 小时取得比 RL 更高的 AIME'24（74.4 vs 67.6） | 让 OPD 变成"工业可选项"的关键数据点 |
| 2025-10 | **Thinking Machines**（Kevin Lu）[1] | 工程化落地：在 RL 脚本上"改一行"即可；系统性对比 OPD / SFT / RL 的算力账 | **本 lab 代码的直接原型**（PyTRIO ≈ Tinker） |
| 2026-04 | **OPD 综述**（Song & Zheng, 腾讯）[6] | 首个系统性综述，200+ 篇文献，统一 `f-divergence` 框架，三轴分类 | 提供**全局地图**与失效模式清单 |
| 2026-04 | **Rethinking OPD** [7] | 现象学 + 机理 + 配方：OPD 成功依赖师生推理模式兼容、老师有真正新增能力 | 提供**边界条件**：什么情况下 OPD 会白跑 |

> 关键认知：本 lab 的 `01-demo-sync.py` 在算法上等价于「GKD 的反向 KL 分支 + MiniLLM 的反向 KL 目标 + Tinker 的工程实现」。它没有发明新算法，而是**把散落在三篇论文里的思想拼成了一个 400 行可跑的最小闭环**。

---

## 3. 数学推导：从 reverse KL 到 advantage

这一节是我认为本 lab 教程讲得最"跳步"的地方，下面把每一步补齐。

### 3.1 符号

| 符号 | 含义 | 代码中的对应 |
|---|---|---|
| `x` | prompt（DeepMath 的 `question` 字段） | `prompt_ids` |
| `y = (y_1..y_T)` | student 采样出的 completion | `seq.tokens` |
| `π_θ` | 正在训练的 student（LoRA r32） | `training_client` |
| `π_T` | 冻结的 teacher（Qwen3.8-27B） | `teacher_client` |
| `β` | KL 强度系数 | `args.kl_penalty_coef`，默认 `1.0` |

### 3.2 目标：最小化 reverse KL

$$D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_T) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\left[\log \pi_\theta(y|x) - \log \pi_T(y|x)\right]$$

注意期望的下标是 `y ~ π_θ` —— 这就是 **on-policy** 的全部含义：**采样分布必须是当前 student**。反过来 `E_{y ~ π_T}` 就是前向 KL（mode-covering），那是 SFT 的目标。

### 3.3 从序列级 KL 拆到 token 级（关键一步）

把序列概率按自回归链式法则展开：

$$\log \pi(y|x) = \sum_{t=1}^{T} \log \pi(y_t \mid x, y_{<t})$$

代回 KL：

$$D_{\mathrm{KL}} = \mathbb{E}_{y \sim \pi_\theta}\left[\sum_{t=1}^{T}\left(\log \pi_\theta(y_t|x,y_{<t}) - \log \pi_T(y_t|x,y_{<t})\right)\right]$$

把求和与期望交换顺序：

$$D_{\mathrm{KL}} = \sum_{t=1}^{T} \mathbb{E}_{y \sim \pi_\theta}\left[\underbrace{\log \pi_\theta(y_t|x,y_{<t}) - \log \pi_T(y_t|x,y_{<t})}_{:=\, r_t^{KL}}\right]$$

**这一步是整篇教程的技术枢纽**：序列级 KL 被拆成了 T 个 token 级 KL 项之和，而每个 `r_t^KL` 只需要**同一个前序上下文下的两个 logprob**。这就解释了为什么 OPD 只需要 teacher 做一次前向（`compute_logprobs`），而不需要 teacher 重新生成——**两个模型在同一段 token 前的条件分布，本来就是逐点可比的**。

> **⚠️ 易错点**：`r_t^KL` **单个 token 可以为负**。因为 `log π_θ(y_t|·) − log π_T(y_t|·)` 只是两个数相减，只有当对 student 完整分布取期望后，KL 才保证非负（Gibbs 不等式）。所以在日志里看到 `reverse_kl_mean = −0.03` 不等于公式失效，但要结合 step 数、序列长度、方差一起看。这一点教程原文明确提示过，也是最容易让人误判"训练崩了"的地方。

### 3.4 为什么必须是反向 KL，而不是前向 KL？

这是我认为最值得写清的一点。两个 KL 的**期望分布不同**，导致完全不同的行为：

| | 前向 KL `D_KL(π_T ‖ π_θ)` | 反向 KL `D_KL(π_θ ‖ π_T)` |
|---|---|---|
| 期望采样自 | teacher | **student** |
| 惩罚什么 | `π_T > 0` 而 `π_θ ≈ 0` 的地方 | `π_θ > 0` 而 `π_T ≈ 0` 的地方 |
| 行为倾向 | **mode-covering**：把所有 teacher 可能的模式都摊平覆盖，包括垃圾模式 | **mode-seeking**：锁定 teacher 的一个主峰，宁可少覆盖也不碰低概率区 |
| 生成任务适用性 | 差：会学到"平均化的、啰嗦的"分布 | 好：学到 teacher 最像的那一路具体行为 |

生成式模型要的是"一个具体的好答案"，不是"所有可能答案的加权平均"，所以 **mode-seeking 的反向 KL 天然更合适**（MiniLLM [4] 的核心论证；教程 [7] 也指出反向 KL 是 "mode seeking" 且能减轻 exposure bias）。

反向 KL 还有两个工程上极其重要的性质 [1]：

1. **不可钻空子（unhackable）**：KL 低 ⟺ 从 teacher 视角看"好行为"的概率高。它不像学出来的 reward model 那样可以被 policy 找到对抗样本。
2. **等价于"模型自身即奖励模型"**：任何经过指令微调的开放权重模型，只要你能调 `compute_logprobs`，就能当 teacher / reward model 用 [1]。这与 DPO 的"你的语言模型其实就是奖励模型"[8] 是同一个洞察。

### 3.5 从 KL 到 advantage：接进 RL 框架

反向 KL 项取负，就变成了最大化目标里的"奖励"：

$$A_t = -\beta \cdot r_t^{KL} = \beta\left(\log \pi_T(y_t|x,y_{<t}) - \log \pi_\theta(y_t|x,y_{<t})\right)$$

对应的代码只有两行（`01-demo-sync.py:344-345`）：

```python
reverse_kl = np.asarray(student_lps) - np.asarray(teacher_lps)
advantages = -args.kl_penalty_coef * reverse_kl
```

**符号直觉**（务必理解，这是调试时唯一的抓手）：

| 情形 | 含义 | advantage | 训练效果 |
|---|---|---|---|
| `student_lp > teacher_lp` | student **比老师更偏爱**这个 token（可能是学生的坏习惯） | `< 0` | 压低该 token 概率 ✅ |
| `student_lp < teacher_lp` | teacher **更偏爱**这个 token（学生该学但没学到） | `> 0` | 提高该 token 概率 ✅ |
| 两者相等 | 学生已经在该状态上与老师一致 | `0` | 不动 |

这一步把 OPD 变成了一个**标准的 policy gradient 问题**：唯一被替换的是 `reward` 的来源——从"环境/奖励模型"换成了"teacher 的 logprob 差"。

### 3.6 OPD ≡ KL 正则化 RL

把上面串起来可以发现一个漂亮的等价关系：

$$\text{OPD} \;=\; \text{RL}\Big(\text{reward}_t := -\beta \cdot r_t^{KL}\Big)$$

也就是说，**OPD 就是"用 teacher 当逐 token 奖励模型"的 RL**。Thinking Machines 原文说得更直接：在已有的、带 KL 正则的 RL 实现上，OPD 只是"把正则化所用的模型换掉"这一行的改动 [1]。

> 反向理解也成立：RL 的 KL 正则项本身就是一种隐式的 OPD，只不过 teacher 是**上一个版本的自己**（reference policy）。这也解释了为什么 2026 年综述把 OPD 形式化地链接到 **KL-constrained RL** [6]。

---

## 4. 训练闭环：五个动作转圈

```
   ┌──────────────────────────────────────────────────┐
   │  ① 刷新 sampler                                   │
   │     training_client.save_weights_and_get_sampling_client()
   │                     ↓                            │
   │  ② student 采样 completion（on-policy！）           │
   │     student_sampler.sample(num_samples=group_size) │
   │     同时保存每个 token 的 old logprob               │
   │                     ↓                            │
   │  ③ teacher 给【同一条轨迹】算 logprob               │
   │     teacher_client.compute_logprobs(prompt+completion)
   │                     ↓                            │
   │  ④ 构造 reverse KL advantage                       │
   │     A_t = −β · (student_lp − teacher_lp)          │
   │                     ↓                            │
   │  ⑤ importance_sampling 更新 + optim_step          │
   │     forward_backward() → optim_step()             │
   └──────────────────────┬───────────────────────────┘
                          └──────→ 回到 ①（下一 step）
```

对应教程配图：[opd-training-loop.png](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/general-opd/images/opd-training-loop.png)

**教程总结的五步最小流程**（原文第 10 节）：

1. 当前 student 采样 completion；
2. 保存 student 采样时的 token logprob；
3. teacher 给同一条 completion 计算 token logprob；
4. 用两者之差构造 reverse-KL advantage；
5. 通过 importance sampling 更新 student，并刷新 sampler。

> **和第 3 步对照着看**：③ 是 OPD 与"假 OPD"的分水岭。如果 teacher 自己重新生成一条答案再和 student 比，token 对不上、状态对不上，那就退回成了 off-policy 蒸馏，**不是** OPD。

---

## 5. 代码逐段精读

### 5.1 两个文件的分工

```text
experiments/general-opd/
├── 01-demo-sync.py     18KB  同步版：最适合阅读和理解完整数据流
├── 02-demo-async.py    21KB  异步版：并发 batch 内 rollout 与 teacher 打分
├── readme.md           20KB  教程正文
└── images/                   训练闭环图 + DeepMath 训练曲线
```

两个版本的**算法完全一致**，差别只在并发组织方式。下面以同步版为主，最后给出差异。

### 5.2 函数地图（同步版，13 个函数）

| 函数 | 行号 | 职责 | 一句话评价 |
|---|---|---|---|
| `parse_args` | 39-92 | 参数解析 + 合法性校验 | 把"不能为 0"的参数课以显式报错，工程习惯好 |
| `shard_name` | 95-97 | 生成 HF parquet 分片命名 | `data/train-00000-of-00010.parquet` |
| `modelscope_file_url` | 100-113 | 拼 ModelScope 文件下载 URL | 绕过 `modelscope` SDK，用裸 urllib |
| `download_if_needed` | 116-130 | 幂等下载，`.tmp` 后 rename | 原子写入，避免半截 parquet |
| `load_deepmath` | 133-166 | 下载 + 读本地 parquet + shuffle + 截断 | **只读 `question`**，不用标答 |
| `build_prompt` | 169-179 | question → chat template → token ids | 加 `\boxed{}` 指令 |
| `completion_teacher_logprobs` | 182-200 | **teacher 对 student 轨迹打分** | OPD 的心脏 |
| `build_opd_datum` | 203-236 | 右移对齐 → PyTRIO `Datum` | 最容易写错的一段 |
| `start_swanlab` | 239-253 | 实验记录初始化 | 与训练逻辑解耦 |
| `main` | 256-394 | 训练主循环 | 5 步闭环 |

### 5.3 第 1 步：准备 prompt 数据（`load_deepmath`）

```python
dataset = load_dataset(
    "parquet",
    data_files=shard_paths,
    split="train",
    cache_dir=str(args.dataset_dir / ".datasets_cache"),
)
```

工程细节值得学：**数据实际从 ModelScope 下载，但用 HF `datasets` 读本地文件**。这样既不依赖 `modelscope` SDK 的 API 变动，又保留了 `datasets` 的缓存与 `select/shuffle` 能力。

```python
if "question" not in dataset.column_names:
    raise ValueError(f"DeepMath must contain 'question', got {dataset.column_names}")
```

**为什么只要 `question`？** DeepMath 在这个实验里是 **prompt-only 数据集**。代码不读标准答案，也不算数学正确率。因为 OPD 的监督信号**完全来自 teacher 的 token 级偏好**，与数据集有没有标答无关。

> 这是一个容易被忽略的设计洞察：**OPD 让"没有标注的数据"变得可用**。这直接决定了它能用在代码、对话、Agent 轨迹这些"标答难拿"的场景上。

```python
dataset = dataset.shuffle(seed=args.seed)
if args.sample_size > 0:
    dataset = dataset.select(range(min(args.sample_size, len(dataset))))
```

固定 seed 打乱再截断，保证试跑可复现——注意顺序不能反（先截断再打乱就等于每次抽同样一批）。

### 5.4 第 2 步：创建 student 与 teacher（`main` 开头）

```python
service_client = trio.ServiceClient()

training_client = service_client.create_lora_training_client(
    base_model=args.base_model,      # Qwen/Qwen3.5-4B
    rank=args.lora_rank,             # 32
    seed=args.seed,
)

teacher_client = service_client.create_sampling_client(
    base_model=args.teacher_base_model or args.base_model,   # Qwen/Qwen3.8-27B
    model_path=args.teacher_model_path,                       # None → 用 base
)
```

两个 client 的**类型不同、权限不同**：

| | `training_client` | `teacher_client` |
|---|---|---|
| 类型 | `create_lora_training_client`（可训练） | `create_sampling_client`（只推理） |
| 能做什么 | forward / backward / optim / 存权重 / 导出 sampler | `compute_logprobs` / `sample` |
| 是否更新参数 | ✅ | ❌ 永远冻结 |

> **设计亮点**：teacher 用的是 **sampling client** 而非 training client，因为**不需要通过 teacher 反传梯度**——只需要一次前向拿 logprob。这个选择直接决定了显存与算力开销（对比 MiniLLM 需要 teacher 的完整分布做序列级优化，Tinker/PyTRIO 这版是纯 logprob）。
>
> **进阶用法**：`teacher_model_path` 如果指向一个已训好的 LoRA sampler，就可以蒸馏**专门领域的 teacher**（比如某个领域的 SFT 专家 LoRA）；不传则直接用 `teacher_base_model`。这就是"同尺寸蒸馏"的入口。

### 5.5 第 3 步：每个 step 刷新 sampler

```python
if student_sampler is None or step % args.sampler_refresh_steps == 0:
    # 严格 on-policy：用当前 student 权重采样
    student_sampler = training_client.save_weights_and_get_sampling_client()
```

训练 client **不能直接当 sampler 用**，必须显式导出权重。默认 `sampler_refresh_steps=1`，即每步都刷新，这是**最严格的 on-policy 定义**。

**权衡**：把刷新间隔调大，能减少权重同步开销，但轨迹会变成由 **stale policy**（几步之前的 student）产生的数据，等价于引入了 off-policy 偏差。这是一个"严格性 ↔ 吞吐"的旋钮，不是纯性能优化。

### 5.6 第 4 步：student 采样自己的轨迹

```python
result = student_sampler.sample(
    prompt=trio.ModelInput.from_ints(prompt_ids),
    num_samples=args.group_size,        # 默认 2
    sampling_params=sampling_params,     # max_tokens=512, T=1.0, top_p=1.0, top_k=-1
    return_text=False,                   # 要 token ids，不要文本
).result()
```

每条 `sequence` 里最关键的两个字段：

- `seq.tokens` —— student 实际生成的 completion token（**这就是 on-policy 轨迹**）
- `seq.logprobs` —— 生成这些 token 时**旧策略**给出的 logprob（后面 importance sampling 要用）

> **⚠️ 关键区别：这里的 `group_size` 不是 GRPO 的 group**
>
> 在 GRPO 里，同一个 prompt 的 `group_size` 条回答会**互相比较算组内相对优势**（`A_i = (r_i − mean(r)) / std(r)`）。
> 在 OPD 里，`group_size` 只是"同一个 prompt 多采几条 on-policy 轨迹"，**每条 completion 独立接受 teacher 的逐 token 监督，彼此之间没有任何比较**。
> 这一点如果搞混，会写出完全错误的代码——OPD 里根本没有"组内 baseline"这个概念。

### 5.7 第 5 步：teacher 给同一条轨迹打分（心脏）

```python
def completion_teacher_logprobs(teacher_client, prompt_ids, completion_ids):
    all_ids = prompt_ids + completion_ids              # ① 拼成完整上下文
    all_logprobs = teacher_client.compute_logprobs(
        trio.ModelInput.from_ints(all_ids)             # ② 一次前向
    ).result()
    completion_logprobs = all_logprobs[len(prompt_ids):]   # ③ 切掉 prompt 区间
    if len(completion_logprobs) != len(completion_ids) or any(v is None for v in completion_logprobs):
        raise ValueError("Invalid teacher logprobs for completion tokens")   # ④ 严格校验
    return [float(v) for v in completion_logprobs]
```

四个细节，每个都有坑：

1. **`all_ids = prompt_ids + completion_ids`** —— teacher 看到的上下文必须与 student rollout 时**逐 token 完全一致**。这是 on-policy 的物理含义。
2. **`compute_logprobs` 不生成，只打分** —— 返回 `all_ids` 中每个 token 在**其前文条件下**的 logprob。这是 OPD 比 RL 便宜的根本原因：teacher 只需一次前向，不需要自回归解码 N 个 token。
3. **`[len(prompt_ids):]` 切片** —— completion 的第 1 个 token 是在"完整 prompt 之后"被预测出来的，所以它对应的 logprob 位置正好是 `all_logprobs[len(prompt_ids)]`。**这是一个 off-by-one 高发区**：写成 `len(prompt_ids)-1` 会把 prompt 的最后一个 token 的 logprob 混进来，整条轨迹的 KL 全错但不报错（长度还对得上，因为你在另一边也错）。
4. **`any(v is None ...)` 严格校验** —— 第一个 token 通常没有前文，返回值可能含 `None`。**必须显式失败**，不能静默转成 0：`None` 一旦被 `float()` 或 `np.asarray` 吞掉，reverse KL 的 token 对齐就不可靠了，而误差会被稀释到均值里，**你只会看到"KL 不降"**。

### 5.8 第 6 步：构造右移对齐的训练 Datum（最容易写错）

```python
prompt_loss_len = len(prompt_ids) - 1

input_ids        = prompt_ids + completion_ids[:-1]
target_ids       = [0] * prompt_loss_len + completion_ids
padded_logprobs  = [0.0] * prompt_loss_len + student_lps
padded_advantages= [0.0] * prompt_loss_len + list(advantages)

assert len(input_ids) == len(target_ids) == len(padded_logprobs) == len(padded_advantages)
```

**为什么要右移？** 自回归训练是"用位置 `t` 的输入 token 预测位置 `t+1` 的目标 token"。

设 prompt 长度为 `P`，completion 长度为 `T`：

| 序列 | 内容 | 长度 |
|---|---|---|
| `input_ids` | `[p_1..p_P] + [c_1..c_{T-1}]` | `P + T − 1` |
| `target_ids` | `[0]*(P−1) + [c_1..c_T]` | `P + T − 1` |
| `logprobs` | `[0.0]*(P−1) + [lp_1..lp_T]` | `P + T − 1` |
| `advantages` | `[0.0]*(P−1) + [A_1..A_T]` | `P + T − 1` |

**关键对应关系**：`input_ids` 的第 `P` 个位置（即 `p_P`）预测 `target_ids` 的第 `P` 个位置（即 `c_1`）。所以 `target_ids` 里 prompt 区间只需要 `P−1` 个占位符 —— 这解释了 `prompt_loss_len = P − 1` 而不是 `P`。

**为什么 prompt 区间用 0 占位而不是直接切掉？** 因为 PyTRIO 的 `importance_sampling` 要求四个序列**长度严格一致**，且 prompt 区间不参与 OPD 更新（advantage=0 自然使其梯度贡献为 0）。用 0 占位是最简单的"屏蔽"实现。

> **⚠️ 陷阱**：这里用了 `target_ids` 的 prompt 区间填 `0`。如果 `0` 恰好是一个合法 token id（通常 `<pad>` 或 `<unk>`），理论上会给 prompt 区间带来一个"预测 0 号 token"的 loss。之所以安全，是因为**对应位置的 advantage 也是 0**，policy gradient 形式下梯度为 0。但这依赖 loss 的具体实现——如果换成带 label smoothing 或纯 CE 的 loss，这个 0 就可能是真的训练信号。**换 loss 函数时必须重新检查这段**。

### 5.9 第 7 步：更新 student

```python
fwd_bwd = training_client.forward_backward(datums, loss_fn="importance_sampling")
optim   = training_client.optim_step(adam)     # lr=4e-5, β1=0.9, β2=0.95
fwd_bwd_result = fwd_bwd.result()
optim.result()
```

**为什么需要 `importance_sampling` 而不是普通 CE？**

因为 `logprobs` 保存的是 **rollout 时的 old policy** 概率。一个 step 内可能收集了多个 prompt 的 datum，而 `forward_backward` 是一批提交的；即便 `sampler_refresh_steps=1`，在一次 `forward_backward` 内部当前参数也可能与采样时不同。`importance_sampling` 用 `π_θ_now / π_θ_old` 的概率比来修正这个偏差：

$$\nabla \mathcal{L} \approx -\mathbb{E}\left[\frac{\pi_\theta(y_t|\cdot)}{\pi_{\theta_{old}}(y_t|\cdot)} \cdot A_t \cdot \nabla \log \pi_\theta(y_t|\cdot)\right]$$

> **闭环完整性**：这是"off-policy 修正"在 OPD 里的最小存在形式。即使采集的是 on-policy 数据，**批内异步 + 多步累积**也必然引入一点 off-policy 性，importance sampling 就是必须的保险。

### 5.10 第 8 步：记录指标与保存权重

```python
metrics = {
    "data/datums": len(datums),
    "data/completion_tokens_mean": float(np.mean(completion_token_counts)),
    "data/completion_tokens_total": completion_token_total,
    "data/completion_tokens_per_second": completion_token_total / step_elapsed_time,
    "opd/reverse_kl_mean": float(np.mean(reverse_kls)),
    "opd/reverse_kl_std": float(np.std(reverse_kls)),
    "train/learning_rate": args.learning_rate,
    "time/step_elapsed_time": step_elapsed_time,
}
metrics.update({f"trainer/{k}": float(v) for k, v in dict(fwd_bwd_result.metrics).items()})

save_result = training_client.save_weights_for_sampler(args.save_weights_name).result()
print(f"Saved weights: {save_result.path}")
```

> **注意 `save_weights_for_sampler` 与 `save_weights_and_get_sampling_client` 的区别**：前者返回一个**路径**（可持久化、可传给后续 `model_path` 做推理或评测），后者返回一个**即时可用的 client 对象**（用于训练循环内的 rollout）。最终交付必须用前者。

### 5.11 同步版 vs 异步版：只改了并发，没改算法

| 维度 | `01-demo-sync.py` | `02-demo-async.py` |
|---|---|---|
| teacher 打分 | 串行 `compute_logprobs(...).result()` | `await compute_logprobs_async(...)`，同 prompt 下 `asyncio.gather` 并发 |
| rollout | 串行遍历 prompt | `run_prompt_rollout` 协程，batch 内 `asyncio.gather` 并发 |
| 主循环 | `def main` | `async def main` |
| 单 prompt 内部 | for 循环 | `asyncio.gather(*teacher_tasks)`（`02-demo-async.py:294`） |
| batch 级 | for 循环 | `asyncio.gather(...)`（`02-demo-async.py:418`） |
| 算法 | 完全相同 | 完全相同 |

**结论**：异步版**没有改变 OPD 算法**，纯粹是提高远程采样/打分阶段的并发利用率。所以教程建议"同步版理解 → 异步版跑量"。

### 5.12 依赖与一个容易踩的坑

```toml
requires-python = ">=3.13"
dependencies = [
    "datasets>=5.0.0", "pytrio==0.2.9", "swanlab==0.10.4",
    "numpy>=2.5.1", "torch>=2.9.1", ...
]
[tool.uv.sources]
torch = { index = "pytorch-cpu" }     # ← 注意
```

两个细节：

1. **`torch` 装的是 CPU 版**。注释写得很清楚："本项目本地只做数据与张量处理；训练和采样由远程 PyTRIO 服务执行。" **本地不需要 GPU** —— 这是这个 lab 设计上最友好的地方，一台笔记本就能跑通全流程。
2. **`pytrio==0.2.9` 是精确 pin**。这意味着 `compute_logprobs` 的返回类型（`list[float | None]`）是版本相关的。升级 SDK 前必须回归那两处 `None` 校验。

---

## 6. 工程陷阱清单（我认为这篇笔记最有价值的部分）

这些是我在读代码时标出来的、**能从"看起来能跑"到"真的学到了东西"之间的七个坑**。

| # | 陷阱 | 症状 | 根因 | 修法 |
|---|---|---|---|---|
| 1 | **tokenizer 不兼容** | 训练看起来正常，KL 不降或乱跳 | student 的 token id 在 teacher 词表里是**另一个 token**，比的是两件不同的事 | 只用同系列模型（Qwen 对 Qwen）；跨系列需做文本级对齐或 DSKD 式双空间投影 [6] |
| 2 | **切片 off-by-one** | KL 全程偏大 1 个 token 的贡献，不明显 | `[len(prompt_ids):]` 写错成 `-1` | 用 `len(completion) == len(teacher_lps)` 断言（代码已经做了） |
| 3 | **右移对齐错位** | loss 抖动、生成质量缓慢变差 | `prompt_loss_len` 用了 `P` 而非 `P−1` | 保持四序列等长断言 |
| 4 | **`None` logprob 被静默吞掉** | KL 异常平稳但不下降 | 第一个 token 无前文 → `None` → 被当成 0 | 显式 `any(v is None)` 抛错（代码已做） |
| 5 | **stale policy** | KL 先降后升、指标噪声大 | `sampler_refresh_steps` 调大后轨迹来自旧策略 | 严格 OPD 就设 1；要吞吐就得接受偏差并监控 |
| 6 | **把 `group_size` 当 GRPO 的组** | 写了组内归一化，训练信号变成"组内相对偏好" | 概念混淆 | OPD 里每条 completion 独立监督，**没有 baseline** |
| 7 | **单 token KL 为负就当崩了** | 反复回滚超参 | 忘了 KL 只在期望意义下非负 | 看多 step 趋势 + 长度 + 方差，不看单点 |

---

## 7. 怎么跑：从 smoke test 到正式实验

### 7.1 环境与登录

```bash
cd agentic-rl-lab
uv sync
trio login          # PyTRIO 认证（远程训练服务）
```

### 7.2 Smoke test（先确认链路，不要直接跑正式版）

```bash
uv run python experiments/general-opd/01-demo-sync.py \
  --steps 1 \
  --batch-size 1 \
  --group-size 1 \
  --max-tokens 128 \
  --sample-size 20 \
  --swanlab-mode disabled
```

**验证清单**（必须全部通过再往下走）：

- [ ] ModelScope 分片下载成功，落到 `experiments/general-opd/datasets/DeepMath-103K/data/train-00000-of-00010.parquet`
- [ ] 打印 `Loaded 20 DeepMath prompts`
- [ ] 打印 `step 000/1 | datums N | ... reverse_kl X.XXXX`
- [ ] 打印 `Saved weights: <path>`（**这是唯一真正证明全链路通的标志**）

### 7.3 正式小规模

```bash
uv run python experiments/general-opd/01-demo-sync.py \
  --steps 10 \
  --batch-size 4 \
  --group-size 4 \
  --max-tokens 512 \
  --sample-size 100 \
  --swanlab-mode online
```

### 7.4 异步版（数据流确认无误后）

```bash
uv run python experiments/general-opd/02-demo-async.py \
  --steps 10 --batch-size 4 --group-size 4 \
  --max-tokens 512 --sample-size 100 \
  --swanlab-mode online
```

### 7.5 关键参数速查

| 参数 | 默认 | 作用 | 调整建议 |
|---|---|---|---|
| `--base-model` | `Qwen/Qwen3.5-4B` | student | Qwen 系列内部换，保 tokenizer 兼容 |
| `--teacher-base-model` | `Qwen/Qwen3.8-27B` | teacher | 不必更大，但要在目标能力上更强 |
| `--lora-rank` | 32 | student LoRA 秩 | 32 起步；128 效果更好但更慢 |
| `--kl-penalty-coef` | 1.0 | β，向 teacher 靠拢力度 | **先保持 1.0**，看趋势再动 |
| `--group-size` | 2 | 每 prompt 采样条数 | 越大 on-policy 轨迹越多样，成本线性涨 |
| `--max-tokens` | 512 | completion 长度上限 | 直接决定采样 + 打分 + 训练成本 |
| `--temperature` / `--top-p` / `--top-k` | 1.0 / 1.0 / -1 | 采样分布 | 太小则轨迹单调（改不到错），太大则方差爆炸 |
| `--sampler-refresh-steps` | 1 | sampler 刷新间隔 | 严格 OPD 就设 1 |
| `--num-shards` | 1 | DeepMath 分片数（共 10） | 联调 1，跑量再增 |
| `--sample-size` | 1000 | 抽样 prompt 数 | `<=0` 表示全用 |
| `--enable-thinking` | False | 是否开 Qwen thinking | 开则 prompt 构造与长度都要重估 |

---

## 8. 指标解读：怎样判断"这次真的学到了"

| 指标 | 含义 | 健康信号 | 危险信号 |
|---|---|---|---|
| `opd/reverse_kl_mean` | student 与 teacher 在采样 token 上的平均 logprob 差 | 多 step 整体**趋稳下降** | 长期不动 / 持续上升 |
| `opd/reverse_kl_std` | token 级 KL 离散度 | 与 mean 同量级或更小 | 远大于 mean（个别 token 在主导） |
| `data/completion_tokens_mean` | 每条 completion 平均长度 | 平稳 | **突然坍缩**（学生学会了"短答"逃避打分） |
| `data/completion_tokens_per_second` | **整步**吞吐 | 稳定 | 大幅下滑说明 teacher 侧排队 |
| `trainer/loss` 等 | PyTRIO 返回 | 平滑下降 | `NaN` / 尖峰 |

### 三条铁律

1. **`completion_tokens_per_second` 不是生成服务的 tokens/s**。它包含 sampler 刷新 + student 采样 + teacher logprob + forward/backward + optim step 的**整步**耗时。拿它去对比 vLLM 的吞吐是错的。
2. **不要只看一个 step 的 `reverse_kl_mean`**。教程明确列了四项联合检查：① KL 是否多 step 趋稳；② completion 长度是否突然坍缩；③ loss 是否 NaN/尖峰；④ **训练前后的独立任务评测是否真的改善**。
3. **KL 下降 ≠ 任务正确率提高**。KL 只说明"更接近 teacher"，如果 teacher 本身在该任务上也不强，KL 降到底也没用（2026 综述把这类问题归为 "calibration 与 capability 的差距"[6]）。

---

## 9. 迁移到自己的任务：六个改造点

教程给了非常实用的改造路径，我按"改动量从小到大"重排：

| # | 改造项 | 改哪里 | 关键约束 |
|---|---|---|---|
| 1 | **换数据集** | `load_deepmath` + `build_prompt` | 只要**能构造出 prompt** 即可。数学题用 `question`，代码用 `problem`，对话用 `messages`，Agent 用 observation/历史轨迹 |
| 2 | **换 teacher** | `--teacher-base-model` / `--teacher-model-path` | teacher **不必更大**，但必须在目标能力上更强：更大通用模型 / 领域 SFT 模型 / 同 base 的专家 LoRA / 更强 system prompt |
| 3 | **多轮消息** | `build_prompt` | **直接传完整 `messages`**，不要先把角色压成一段纯文本（会丢掉 chat template 的角色标记，导致 prompt 分布偏移） |
| 4 | **调采样分布** | `--temperature/--top-p/--top-k/--max-tokens` | 太保守 → 轨迹多样不足、teacher 能纠的错类型有限；太随机 → 低质量 token 多、方差与成本齐涨 |
| 5 | **调 KL 强度** | `--kl-penalty-coef` | 太小靠拢慢，太大 advantage 过强导致不稳定。**建议先 1.0** |
| 6 | **切异步** | 换脚本 | 数据流确认无误再切 |

### 唯一的硬前提（两个）

> **① 数据能构造成 prompt；② student 与 teacher 的 tokenizer 兼容。**

第 ② 条是整个方法的地基：因为信号是 `log π(student_token_id) − log π(same_token_id)`，**token ID 不同的模型之间，这个差值没有意义**。2026 综述把跨词表的解法列为 DSKD 一类的"双空间投影 + 精确 token 对齐"[6]，属于另一条工程路线。

---

## 10. 批判性评估：OPD 不是银弹

读完教程再读 2026 年的综述 [6] 与 Rethinking 论文 [7]，能看到教程里没有强调的四个边界。这部分我认为是区分"抄了一遍教程"和"真的理解了"的地方。

### 10.1 四个已知失效模式 [6]

| 失效模式 | 表现 | 机理 |
|---|---|---|
| **多样性坍缩（diversity collapse）** | 学生输出变得高度同质 | 反向 KL 是 mode-seeking，天然收缩到单一主峰 |
| **错误前缀陷阱（flawed prefix trap）** | 学生一开头就错，后续 teacher 的"纠正"已无意义 | teacher 在被污染的前缀上也可能给出误导性偏好 |
| **自博弈饱和（saturation during self-play）** | 自蒸馏版本收益很快见顶 | teacher-free 变体缺少外部新增信息 |
| **校准-能力差距（calibration vs capability）** | KL 很低但任务正确率不涨 | 学到的是"风格和自信度"，不是"事实准确性"（这正是 Gudibande 2023 [9] 指出的 off-policy 蒸馏老问题在 OPD 里的回声） |

### 10.2 Rethinking OPD 的两个前提条件 [7]

2026 年 4 月的这篇论文给出了 OPD 有效的**必要条件**，比教程的"tokenizer 兼容"更本质：

1. **师生推理模式兼容**：如果 student 的思维链风格与 teacher 差异过大，teacher 在 student 的轨迹上给不出有意义的偏好信号。
2. **teacher 提供**真正的新增**能力**：如果 teacher 只是"同一能力的更自信版本"，OPD 只能学到自信，学不到能力。

> 一个反直觉但有实践价值的发现：**只用 1 条或极少量 prompt 训练，就能恢复大部分收益**。这与教程里"数据只要构造成 prompt 就行"是同一件事的两面——OPD 的信息量来自**轨迹上的每 token 打分**，而不是**prompt 的数量**。Thinking Machines 原文也报告了"仅用一条提示训练 20 步、5120 条轨迹，就能大致达到 teacher 在 AIME'24 上的表现"[1]。

### 10.3 成本账：OPD 并不便宜，只是"更值"

| 方法 | AIME'24 | GPU 小时（Qwen3 报告） |
|---|---|---|
| 异策略蒸馏（SFT） | 55.0% | 未报告 |
| RL | 67.6% | 17,920 |
| **OPD** | **74.4%** | **1,800** |

数据来源：Qwen3 技术报告 Table 21，经 Thinking Machines 博文转载 [1][5]。

但注意两个成本口径：

- **相对 SFT 更贵**：2026 综述给出的量级是 **OPD 约为 off-policy SFT 的 4–5 倍算力** [6]。因为每步都要 student 采样 + teacher 全量前向。
- **相对 RL 便宜得多**：约 1/10 GPU 小时，或在特定设定下 9–30 倍计算效率（视是否把 teacher 的 FLOPs 计入、是否假设 SFT 数据已现成）[1]。

**所以 OPD 的正确定位是**：**当你已经有一个强 teacher、且 RL 的稀疏奖励让你烧不起时，OPD 是最优性价比**。反之，如果你没有合适 teacher，或者任务本身有干净的环境 reward，OPD 未必划算。

### 10.4 一个我认为教程没说清的点

教程第 2.3 节写"通过 importance sampling 更新 student"，但没有解释**为什么 on-policy 数据还需要 importance sampling**。我的理解（基于代码结构推断）是：一个 step 内 `datums` 是批量提交的，`forward_backward` 只做一次前向/反向，而 `optim_step` 在其后。因此批内不同 prompt 的 datum 是用**同一组参数**做前向的，与它们各自的采样时刻存在微小偏差；同时 `old_logprobs` 是采样时刻的真实值，用它做比率修正比直接用当前策略的 logprob 更稳。**这不是算法必需，而是工程上的方差控制**。

> 如果有人能确认 PyTRIO `importance_sampling` 的确切形式（是否是 PPO 式的 clip 比率），欢迎补充——我这里是从 `loss_fn_inputs` 只有 `target_tokens/logprobs/advantages` 三个字段反推的。

---

## 11. 和我自己工作的关联（个人补充，非教程内容）

我日常在做工厂产线的视觉检测 + SOP 状态机（5G 线束 OBB、风管、出风口三工位，YOLO 检测 + 规则引擎 + MES 闭环）。OPD 能怎么用、不能怎么用，我梳理了一下：

**❌ 不能直接用**：我的检测任务是**连续视觉回归/框回归**，没有 token 序列，OPD 的 token 级 logprob 差无从定义。想沾边只能用 RL 的连续动作版本（PPO/SAC），那是另一套。

**✅ 可以用**：SOP 的**判定理由生成**、Agent 化的工位助手、CVAT 标注辅助的自然语言推理部分——这些都产出 token 序列，且"标准答案"难标但"老师傅的判断"可获取。具体设想：

- teacher = 一个有经验工艺员写的判定逻辑蒸馏出来的大模型（或 Qwen 大尺寸版 + 长 prompt 注入 SOP 手册）
- student = 部署在工控机上的小模型
- prompt = 当前工位状态 + 规则引擎命中结果 + 图像描述
- 这正是教程第 9 节说的"Agent 任务的 observation 或历史轨迹"

**⚠️ 最大的现实障碍**：tokenizer 兼容 + 现场算力。工控机（RK3588 那几台）跑 student 采样还行，但 teacher 侧的打分必须放在 DGX 上——这就变成了一个"边缘采样 + 中心打分"的分布式 OPD，本 lab 的同步脚本得改造成跨机异步。**这件事有工程价值，但不是这两天的任务。**

---

## 12. 自测题（我给自己出的，答案在文内）

1. 为什么 OPD 的期望必须取在 `π_θ` 上，而不能取在 `π_T` 上？
2. 序列级 reverse KL 是怎么变成 token 级求和的？哪一步用了链式法则？
3. `r_t^KL` 为负数说明公式错了吗？为什么？
4. `input_ids` 长度是 `P+T−1`，`target_ids` 里 prompt 区间为什么只有 `P−1` 个占位？
5. `group_size=4` 在 GRPO 和 OPD 里的语义差别是什么？
6. 为什么 `teacher_client` 用 `create_sampling_client` 而不是 training client？
7. `sampler_refresh_steps` 从 1 调到 10，算法上发生了什么变化？属于"性能优化"吗？
8. 如果 KL 一直在降但 AIME 分数不涨，最可能是什么原因？怎么验证？

---

## 13. 参考资料

> **规范**：以下所有链接均为本文实际引用来源，对应正文 `[编号]`。教程与官方仓库排在前，原始论文排在后。

### 课程与代码（一手来源）

| 编号 | 来源 | 链接 |
|---|---|---|
| — | 本 Task 教程原文（LAB 05 General OPD） | https://datawhalechina.github.io/agentic-rl-lab/experiments/general-opd/ |
| — | Agentic RL Lab 开源仓库 | https://github.com/datawhalechina/agentic-rl-lab |
| — | 同步版代码 `01-demo-sync.py` | https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/general-opd/01-demo-sync.py |
| — | 异步版代码 `02-demo-async.py` | https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/general-opd/02-demo-async.py |
| — | 教程正文 `readme.md` | https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/general-opd/readme.md |
| — | 训练闭环配图 | https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/general-opd/images/opd-training-loop.png |
| — | DeepMath-103K 数据集（ModelScope） | https://modelscope.cn/datasets/AI-ModelScope/DeepMath-103K |
| — | PyTRIO 远程训练平台 | https://pytrio.cn/ |
| — | PyTRIO Skill | https://github.com/SwanHubX/pytrio-skill |
| — | SwanLab 实验记录 | https://swanlab.cn/login |
| — | SwanLab Skill | https://github.com/SwanHubX/swanlab-skill |

### 论文与原始文献

| 编号 | 文献 | 链接 |
|---|---|---|
| [1] | Lu, Kevin & Thinking Machines Lab. *On-Policy Distillation*. 2025-10-27. DOI: 10.64434/tml.20251026 | https://thinkingmachines.ai/blog/on-policy-distillation/ |
| [2] | Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*（DAgger）, AISTATS 2011 | https://arxiv.org/abs/1011.0686 |
| [3] | Agarwal et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*（GKD）, 2023 | https://arxiv.org/abs/2306.13649 |
| [4] | Gu, Dong, Wei, Huang. *MiniLLM: Knowledge Distillation of Large Language Models*, ICLR 2024 | https://arxiv.org/abs/2306.08543 |
| [5] | Qwen Team. *Qwen3 Technical Report*, 2025（Table 21 为 OPD vs RL 成本对比） | https://arxiv.org/abs/2505.09388 |
| [6] | Song, Mingyang & Zheng, Mao（腾讯）. *A Survey of On-Policy Distillation for Large Language Models*, 2026 | https://arxiv.org/abs/2604.00626 |
| [7] | *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*, 2026 | https://arxiv.org/abs/2604.13016 |
| [8] | Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, 2023 | https://arxiv.org/abs/2305.18290 |
| [9] | Gudibande et al. *The False Promise of Imitating Proprietary LLMs*, 2023 | https://arxiv.org/abs/2305.15717 |
| — | Thinking Machines. *LoRA Without Regret*（O(1) vs O(N) 比特的来源） | https://thinkingmachines.ai/blog/lora/ |
| — | Awesome-LLM-On-Policy-Distillation（综述配套文献列表） | https://github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation |
| — | Tinker Cookbook 蒸馏配方（同思路的官方实现） | https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation |

### 本地强化学习基础资源（本次学习一并整理，详见 `本机强化学习资源盘点_2026-11-26.md`）

| 来源 | 链接 |
|---|---|
| 蘑菇书 EasyRL（Datawhale） | https://github.com/datawhalechina/easy-rl |
| 动手学强化学习（上海交大 张伟楠等） | https://github.com/boyu-ai/Hands-on-RL |
| 强化学习的数学原理（西湖大学 赵世钰） | https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning |

---

## 14. 一句话收尾

> OPD 最漂亮的地方不是那个 KL 公式，而是它**把"设计奖励函数"这件最需要人肉判断的事，降级成了"找一个足够强的模型，调一次前向"**。
> 代价是你把方向盘交给了 teacher —— 所以真正决定成败的，从来不是代码那两行，而是 **teacher 选得对不对**。

---

*笔记完成时间：2026-11-26 04:xx | 代码版本：agentic-rl-lab @ main（2026-11-26 clone）| 教程更新：2026-11-16*
