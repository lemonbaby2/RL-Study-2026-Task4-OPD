# Task4 打卡 · General OPD 原理与代码解读

**姓名**：李梓鹏
**学习日期**：2026-11-26
**教程**：https://datawhalechina.github.io/agentic-rl-lab/experiments/general-opd/
**代码**：https://github.com/datawhalechina/agentic-rl-lab （已本地 clone 并逐行通读 `experiments/general-opd/01-demo-sync.py` / `02-demo-async.py`）

---

## 一、一句话理解 OPD

轨迹由**当前 student** 生成，**teacher 只负责给这条轨迹逐 token 打分**，再用**反向 KL** 把 student 拉向 teacher。

把它放进后训练方法的二维坐标里就非常清楚：

| 方法 | 采样方式 | 奖励信号 | 每回合信息量 |
|---|---|---|---|
| SFT / 离线蒸馏 | off-policy | 密集 | O(N) bit |
| RL（PPO/GRPO） | on-policy | 稀疏 | **O(1) bit** |
| **OPD** | **on-policy** | **密集** | **O(N) bit** |

- off-policy 的痛点是 **exposure bias**：学生学的是 teacher 常遇的上下文，部署时走的却是自己的上下文，早期一个小错就漂出训练分布，`O(εT²)` 累积（DAgger, 2010）。
- on-policy RL 的痛点是**信号太稀**：512 token 的推理只拿一个 `reward=0`，知道错了但不知道错在哪一步。
- OPD 把两者缝合：**保留 on-policy 采样，把稀疏 outcome reward 换成 teacher 的密集 token 级 logprob**。

国际象棋类比最好记：RL 是自己下棋只知道输赢；离线蒸馏是看大师下棋但那些局面你遇不到；OPD 是老师坐在旁边给**你自己下的每一步**打分。

来源：Thinking Machines Lab, *On-Policy Distillation*, 2025-10-27 — https://thinkingmachines.ai/blog/on-policy-distillation/

---

## 二、原理推导（补齐教程跳过的步骤）

**符号**：`x` prompt、`y` student 采样出的 completion、`π_θ` 训练中的 student、`π_T` 冻结的 teacher、`β` KL 强度。

**目标**（反向 KL）：

```
D_KL(π_θ ‖ π_T) = E_{y~π_θ}[ log π_θ(y|x) − log π_T(y|x) ]
```

期望采样自 `π_θ` —— 这就是 on-policy 的全部含义。

**拆到 token 级**（这一步是技术枢纽）。用链式法则 `log π(y|x) = Σ_t log π(y_t|x,y_<t)` 展开，再交换期望与求和：

```
D_KL = Σ_t E_{y~π_θ}[ log π_θ(y_t|x,y_<t) − log π_T(y_t|x,y_<t) ]  ≜ Σ_t E[r_t^KL]
```

**为什么这步重要**：序列级 KL 变成 T 个 token 级项之和，每项只需要**同一前序上下文下的两个 logprob**。所以 teacher 只需**一次前向**（`compute_logprobs`），不必重新自回归生成——这就是 OPD 比 RL 便宜的根本原因。

**转为 advantage**：

```
A_t = −β · r_t^KL
```

符号直觉：`student_lp > teacher_lp` 说明学生比老师更偏爱这个 token（可能是坏习惯），`A_t < 0` 压低它；反之 `A_t > 0` 抬高它。代码就两行：

```python
reverse_kl = np.asarray(student_lps) - np.asarray(teacher_lps)
advantages = -args.kl_penalty_coef * reverse_kl
```

**为什么必须是反向 KL 而不是前向 KL**：前向 KL 是 mode-covering（把所有 teacher 可能的模式摊平，包括垃圾），反向 KL 是 mode-seeking（锁定 teacher 的一个主峰，学到具体行为而不是平均行为）。生成任务要"一个具体好答案"，不要"所有答案的加权平均"。来源：MiniLLM, https://arxiv.org/abs/2306.08543

**一个漂亮的等价关系**：`OPD = RL(reward_t := −β·r_t^KL)`。也就是**用 teacher 当逐 token 奖励模型的 RL**。Thinking Machines 原文说得更直白：在已有的带 KL 正则的 RL 实现上，OPD 只是"把正则化所用的模型换掉"这一行的改动。

**易错点**：`r_t^KL` **单个 token 可以为负**。因为只是两个数相减，只有对 student 完整分布取期望后 KL 才保证非负。日志里看到负值不等于公式失效。

---

## 三、代码精读：五个动作转圈

```
① 刷新 sampler：save_weights_and_get_sampling_client()   ← 严格 on-policy
② student 采样 completion，同时保存每个 token 的 old logprob
③ teacher 对【同一条轨迹】compute_logprobs(prompt + completion)
④ A_t = −β · (student_lp − teacher_lp)
⑤ forward_backward(loss_fn="importance_sampling") → optim_step()
```

### 三个最关键的函数

**1. `completion_teacher_logprobs` —— OPD 的心脏**

```python
all_ids = prompt_ids + completion_ids                       # teacher 看到和 rollout 完全一致的上下文
all_logprobs = teacher_client.compute_logprobs(...).result() # 一次前向，不生成
completion_logprobs = all_logprobs[len(prompt_ids):]         # 切掉 prompt 区间
if len(...) != len(completion_ids) or any(v is None for v in ...):
    raise ValueError("Invalid teacher logprobs")
```

- teacher **不重新生成答案**，只读 student 已生成的 token → 这是 OPD 与"假 OPD"（退化成离线蒸馏）的分水岭。
- 切片从 `len(prompt_ids)` 开始（不是 `-1`）：completion 第 1 个 token 的 logprob 正好落在该位置。**off-by-one 高发区**。
- `any(v is None)` 必须显式抛错：`None` 被静默转成 0 后，误差会稀释进均值，**你只会看到"KL 不降"**。

**2. `build_opd_datum` —— 最容易写错的一段**

```python
prompt_loss_len = len(prompt_ids) - 1        # 注意是 P−1，不是 P
input_ids  = prompt_ids + completion_ids[:-1]                # 长度 P+T−1
target_ids = [0]*prompt_loss_len + completion_ids            # 长度 P+T−1
```

自回归训练是"位置 t 的输入预测 t+1 的目标"。`input_ids` 第 P 位（`p_P`）预测 `target_ids` 第 P 位（`c_1`），所以 target 的 prompt 区间只需 `P−1` 个占位。四个序列必须严格等长（代码有断言）。

prompt 区间用 0 占位而非切掉，是因为 PyTRIO 的 `importance_sampling` 要求四序列等长；而该区间 `advantage=0`，梯度贡献自然为 0。**但这个安全前提依赖 loss 形式**——换成纯 CE 或带 label smoothing 的 loss，0 就可能是真实训练信号，必须重查。

**3. `importance_sampling` 为何必要**

`logprobs` 存的是 rollout 时的 old policy 值。一个 step 内 datum 是批量提交、一次前向/反向，批内各 datum 的采样时刻与当前参数存在微小偏差，因此用 `π_now/π_old` 比率修正。这是**工程上的方差控制**，不是算法必需。

### 同步版 vs 异步版

算法**完全一致**，差别只在并发：异步版把 `compute_logprobs` 换成 `compute_logprobs_async`，同 prompt 下多条 completion 用 `asyncio.gather` 并发（`02-demo-async.py:294`），batch 级也 gather（:418）。**它没有改变 OPD，只是提高远程采样/打分的并发利用率。**

---

## 四、我认为最有价值的：七个工程陷阱

| # | 陷阱 | 症状 | 修法 |
|---|---|---|---|
| 1 | **tokenizer 不兼容** | KL 不降或乱跳 | 同系列模型（Qwen↔Qwen）；跨系列需 DSKD 式双空间投影 |
| 2 | **切片 off-by-one**（`len(P)` 写成 `len(P)-1`） | KL 全程偏大，不明显 | 断言 `len(completion)==len(teacher_lps)` |
| 3 | **右移对齐错位**（`P−1` 写成 `P`） | loss 抖、质量缓降 | 保持四序列等长断言 |
| 4 | **`None` 被静默吞掉** | KL 平稳但不降 | 显式 `any(v is None)` 抛错 |
| 5 | **stale policy**（`sampler_refresh_steps` 调大） | KL 先降后升 | 严格 OPD 设 1；要吞吐就得接受偏差 |
| 6 | **把 `group_size` 当 GRPO 的组** | 写了组内归一化，信号变味 | OPD 里每条 completion **独立**监督，**没有 baseline** |
| 7 | **单 token KL 为负就回滚** | 反复改超参 | 看多 step 趋势 + 长度 + 方差 |

> 第 6 条是最容易概念混淆的：GRPO 的 `group_size` 条回答要**互相比较算组内相对优势**；OPD 的 `group_size` 只是"同 prompt 多采几条 on-policy 轨迹"，彼此完全不比较。

---

## 五、跑通验证与指标

**Smoke test**（先确认链路，别直接跑正式版）：

```bash
uv sync && trio login
uv run python experiments/general-opd/01-demo-sync.py \
  --steps 1 --batch-size 1 --group-size 1 \
  --max-tokens 128 --sample-size 20 --swanlab-mode disabled
```

唯一能证明全链路通的标志是打印 `Saved weights: <path>`。

**一个设计上很友好的点**：`pyproject.toml` 里 `torch` 用的是 `pytorch-cpu` 索引。注释写明"本地只做数据与张量处理，训练与采样由远程 PyTRIO 执行"——**本地不需要 GPU**，笔记本就能跑通全流程。

**三条指标铁律**：

1. `completion_tokens_per_second` 是**整步**吞吐（含 sampler 刷新/采样/teacher 打分/反传/优化），不要拿去对比 vLLM 的生成 tokens/s。
2. 不看单 step 的 `reverse_kl_mean`，看四项联合：KL 多 step 是否趋稳 / completion 长度是否突然坍缩 / loss 是否 NaN 或尖峰 / **训练前后独立任务评测是否真改善**。
3. **KL 下降 ≠ 正确率提高**。只说明"更接近 teacher"。

---

## 六、批判性补充：OPD 的边界（教程没强调的）

读完 2026 年腾讯那篇 OPD 综述（arXiv:2604.00626）和 Rethinking 论文（arXiv:2604.13016），补了四个边界：

**四个已知失效模式**：① 多样性坍缩（mode-seeking 的副作用）；② 错误前缀陷阱（开头就错，后续"纠正"已无意义）；③ 自博弈饱和；④ **校准-能力差距**（KL 很低但正确率不涨，学的是风格自信度不是事实准确性）。

**OPD 有效的两个必要条件**（Rethinking, 2026）：① 师生推理模式兼容；② teacher 提供**真正的新增能力**——如果 teacher 只是"同能力的更自信版本"，只能学到自信。

**成本要算清**：相对 RL 便宜约 10 倍（Qwen3 报告：OPD 74.4% / 1800 GPU·h vs RL 67.6% / 17920 GPU·h），但相对离线 SFT **更贵**——综述给的量级是 **4–5 倍算力**。正确定位是：**有强 teacher、又烧不起 RL 时性价比最高**。

**一个反直觉的发现**：只用 1 条或极少量 prompt 训练就能恢复大部分收益。因为 OPD 的信息量来自**轨迹上的每 token 打分**，不是 prompt 数量。

---

## 七、迁移到自己的场景

我在做工厂产线视觉检测 + SOP 状态机（YOLO 检测 + 规则引擎 + MES 闭环），梳理了一下：

- **❌ 不能直接用**：检测是连续视觉回归（框回归），没有 token 序列，`r_t^KL` 无从定义。想沾边只能用 PPO/SAC 的连续动作版。
- **✅ 可以用**：SOP 判定理由生成、Agent 化工位助手、CVAT 标注辅助的自然语言推理部分 —— 都产出 token 序列，且"标答难拿但老师傅判断可得"。
- **⚠️ 现实障碍**：teacher 打分必须放 DGX，student 采样在工控机（RK3588）→ 变成"边缘采样 + 中心打分"的分布式 OPD，同步脚本要改造。有工程价值，但不是这两天的任务。

**改自己任务的六个点**：换数据集（只要能构造 prompt）/ 换 teacher（不必更大，但要在目标能力上更强）/ 多轮消息直接传完整 `messages`（别压成纯文本）/ 调采样分布 / 调 `β`（**先保持 1.0**）/ 切异步。

**唯一的两个硬前提**：① 数据能构造成 prompt；② **student 与 teacher tokenizer 兼容**。

---

## 八、一句话收尾

> OPD 最漂亮的地方不是那个 KL 公式，而是它**把"设计奖励函数"这件最需要人肉判断的事，降级成了"找一个足够强的模型、调一次前向"**。
> 代价是你把方向盘交给了 teacher —— 所以真正决定成败的从来不是代码那两行，而是 **teacher 选得对不对**。

---

## 参考资料来源（全部标注）

**课程一手来源**
1. Task4 教程 General OPD：https://datawhalechina.github.io/agentic-rl-lab/experiments/general-opd/
2. Agentic RL Lab 仓库：https://github.com/datawhalechina/agentic-rl-lab
3. 同步版代码：https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/general-opd/01-demo-sync.py
4. 异步版代码：https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/general-opd/02-demo-async.py
5. DeepMath-103K 数据集：https://modelscope.cn/datasets/AI-ModelScope/DeepMath-103K
6. PyTRIO：https://pytrio.cn/ ｜ SwanLab：https://swanlab.cn/login

**论文与原始文献**
7. Lu, Kevin & Thinking Machines Lab. *On-Policy Distillation*. 2025-10-27：https://thinkingmachines.ai/blog/on-policy-distillation/
8. Ross et al. *DAgger*. AISTATS 2011：https://arxiv.org/abs/1011.0686
9. Agarwal et al. *GKD: On-Policy Distillation of Language Models*. 2023：https://arxiv.org/abs/2306.13649
10. Gu et al. *MiniLLM*. ICLR 2024：https://arxiv.org/abs/2306.08543
11. Qwen Team. *Qwen3 Technical Report*. 2025（Table 21 成本对比）：https://arxiv.org/abs/2505.09388
12. Song & Zheng（腾讯）. *A Survey of On-Policy Distillation for LLMs*. 2026：https://arxiv.org/abs/2604.00626
13. *Rethinking On-Policy Distillation of LLMs*. 2026：https://arxiv.org/abs/2604.13016
14. Rafailov et al. *DPO*. 2023：https://arxiv.org/abs/2305.18290
15. Gudibande et al. *The False Promise of Imitating Proprietary LLMs*. 2023：https://arxiv.org/abs/2305.15717
16. Thinking Machines. *LoRA Without Regret*（O(1)/O(N) 比特来源）：https://thinkingmachines.ai/blog/lora/
17. Awesome-LLM-On-Policy-Distillation：https://github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation

**本地基础资源（本次一并整理）**
18. 蘑菇书 EasyRL：https://github.com/datawhalechina/easy-rl
19. 动手学强化学习（上交 张伟楠等）：https://github.com/boyu-ai/Hands-on-RL
20. 强化学习的数学原理（西湖大学 赵世钰）：https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning

---

**完整版笔记（含逐函数精读、完整推导、自测题）见仓库**：`notes/Task04_General_OPD_原理与代码解读.md`
