# 强化学习学习包 2026-11 · Task4 General OPD

> Datawhale **Agentic RL Lab** 打卡仓库 · 第 4 期
> 作者：李梓鹏　|　完成时间：2026-11-26
> 一门课的两条线：**LLM 后训练（OPD/GRPO）** + **强化学习理论基础（MDP→PPO→具身）**

---

## 这是什么

一次任务，三层交付：

| 层 | 内容 | 入口 |
|---|---|---|
| **① 打卡笔记** | Task4「General OPD 原理与代码解读」：原理推导 + 代码逐函数精读 + 七个工程陷阱 + 批判性评估 | [`notes/Task04_General_OPD_原理与代码解读.md`](notes/Task04_General_OPD_原理与代码解读.md) |
| **② 资源包** | 全网最系统的中文 RL 课件（教材 / 讲义 slides / 21 章 notebook / 4 篇原始论文） | [`notes/本机强化学习资源盘点_2026-11-26.md`](notes/本机强化学习资源盘点_2026-11-26.md) |
| **③ 可跑代码** | `agentic-rl-lab` 全仓库，18 个实验（grpo / dapo / gspo / general-opd / opsd / ...） | [`repo/experiments/`](repo/experiments/) |

---

## 快速开始

### 1. 看笔记

```bash
# 完整版：推导 + 逐函数精读 + 自测题
open notes/Task04_General_OPD_原理与代码解读.md

# 打卡版：可直接粘贴到飞书表单
open notes/Task04_打卡正文_可直接粘贴版.md

# 网页版（推荐）
python -m http.server 8848
# 浏览器打开 http://localhost:8848
```

### 2. 补下未入库的大文件

EasyRL 蘑菇书完整 PDF 有 **165MB**，超过 GitHub/Gitee 的单文件限制，未纳入 git。运行脚本自动补齐：

```bash
bash scripts/download_assets.sh
```

### 3. 跑通 OPD（需要 PyTRIO 账号）

```bash
cd repo
uv sync
trio login      # PyTRIO 注册：https://pytrio.cn/

# Smoke test：先确认链路，别直接跑正式版
uv run python experiments/general-opd/01-demo-sync.py \
  --steps 1 --batch-size 1 --group-size 1 \
  --max-tokens 128 --sample-size 20 --swanlab-mode disabled

# 看到 Saved weights: <path> 才算全链路通
```

> **本地不需要 GPU**。`pyproject.toml` 里 `torch` 走 `pytorch-cpu` 索引——本地只做数据处理，训练与采样由远程 PyTRIO 执行。

---

## 目录结构

```
RL学习_2026-11/
├── README.md
├── index.html                     网页预览（单文件，无依赖）
├── scripts/
│   └── download_assets.sh         补齐未入库的大文件
├── notes/
│   ├── Task04_General_OPD_原理与代码解读.md
│   ├── Task04_打卡正文_可直接粘贴版.md
│   └── 本机强化学习资源盘点_2026-11-26.md
├── 课件/
│   ├── 强化学习的数学原理_赵世钰_全本.pdf          19M  西湖大学，Springer
│   ├── 赵世钰_课程slides/                        L1-L10 配套 slides
│   ├── 动手学强化学习_notebooks/                 21 章 + rl_utils.py
│   └── 论文/
│       ├── 2306.13649.pdf                      GKD（OPD 框架奠基）
│       ├── 2306.08543.pdf                      MiniLLM（反向 KL 论证）
│       ├── 2604.00626_OPD综述_腾讯.pdf           2026 首个 OPD 系统综述
│       └── 2604.13016_Rethinking_OPD.pdf       2026 OPD 边界条件
└── repo/                          agentic-rl-lab 全仓库
    └── experiments/               18 个实验
```

---

## Task4 笔记要点（速览）

### OPD 在后训练方法中的坐标

| 方法 | 采样方式 | 奖励信号 | 每回合信息量 |
|---|---|---|---|
| SFT / 离线蒸馏 | off-policy | 密集 | O(N) bit |
| RL（PPO/GRPO） | on-policy | 稀疏 | **O(1) bit** |
| **OPD** | **on-policy** | **密集** | **O(N) bit** |

### 核心公式链

```
D_KL(π_θ ‖ π_T) = E_{y~π_θ}[ log π_θ(y|x) − log π_T(y|x) ]        ← 反向 KL，mode-seeking
   ↓ 链式法则拆 token + 交换期望与求和
D_KL = Σ_t E[ r_t^KL ],  r_t^KL = log π_θ(y_t|x,y_<t) − log π_T(y_t|x,y_<t)
   ↓ 取负作为 advantage
A_t = −β · r_t^KL
   ↓ 接进 RL
OPD ≡ RL(reward_t := −β·r_t^KL)                                    ← teacher 即逐 token 奖励模型
```

### 训练闭环五步

```
① 刷新 sampler（严格 on-policy）
② student 采样 completion，保存 old logprob
③ teacher 对【同一条轨迹】compute_logprobs（一次前向，不生成）
④ A_t = −β · (student_lp − teacher_lp)
⑤ importance_sampling 更新 + optim_step
```

### 七个工程陷阱

| # | 陷阱 | 修法 |
|---|---|---|
| 1 | tokenizer 不兼容 | 同系列模型（Qwen↔Qwen） |
| 2 | 切片 off-by-one | 断言 `len(completion)==len(teacher_lps)` |
| 3 | 右移对齐错位（`P−1` 写成 `P`） | 四序列等长断言 |
| 4 | `None` logprob 被静默吞掉 | 显式 `any(v is None)` 抛错 |
| 5 | stale policy | `sampler_refresh_steps=1` |
| 6 | 把 `group_size` 当 GRPO 的组 | OPD 无 baseline，每条独立监督 |
| 7 | 单 token KL 为负就当崩了 | KL 只在期望意义下非负 |

> 完整版笔记里还有：13 个函数逐段精读、8 道自测题、四个已知失效模式、成本账（OPD vs SFT vs RL）、迁移到自己任务的六个改造点。

---

## 资源来源与版权声明

**本仓库不拥有任何第三方课件的版权**，仅做学习目的的本地归档与索引。所有资料版权归原作者所有，请遵守各原始许可：

| 资料 | 作者/机构 | 许可 | 原始地址 |
|---|---|---|---|
| EasyRL 蘑菇书 | Datawhale · Qi Wang 等 | CC BY-NC-SA 4.0 | https://github.com/datawhalechina/easy-rl |
| 动手学强化学习 | 上海交大 · 张伟楠 / 沈键 / 俞勇 | 见仓库 | https://github.com/boyu-ai/Hands-on-RL |
| 强化学习的数学原理 | 西湖大学 · 赵世钰 | 见仓库 | https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning |
| agentic-rl-lab | Datawhale · KMnO4-zx 等 | 见仓库 | https://github.com/datawhalechina/agentic-rl-lab |
| GKD / MiniLLM / OPD 综述 / Rethinking OPD | 各论文作者 | arXiv 许可 | 见 `notes/` 内链接 |
| General OPD 教程正文 | Datawhale · KMnO4-zx | 见仓库 | https://datawhalechina.github.io/agentic-rl-lab/experiments/general-opd/ |

**本仓库原创部分**（`notes/` 下三份笔记、`index.html`、`scripts/`）采用 **CC BY-NC-SA 4.0**，转载请注明出处。

---

## 参考链接

- 任务教程：https://datawhalechina.github.io/agentic-rl-lab/experiments/general-opd/
- 开源仓库：https://github.com/datawhalechina/agentic-rl-lab
- 打卡地址：https://magicyang.feishu.cn/share/base/shrcnrMFdVcimGmPP5gBeyuMJod
- Thinking Machines OPD 原文：https://thinkingmachines.ai/blog/on-policy-distillation/

---

*本仓库构建于 2026-11-26。代码版本：agentic-rl-lab @ main。教程更新：2026-11-16。*
