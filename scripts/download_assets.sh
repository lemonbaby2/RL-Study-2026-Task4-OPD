#!/usr/bin/env bash
# 补齐未入库的大文件（>100MB 的文件被 GitHub/Gitee 单文件限制排除）
# 用法：bash scripts/download_assets.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$ROOT/课件"
mkdir -p "$ASSETS"

echo "==> 目标目录：$ASSETS"

# ---------------------------------------------------------------
# 1. EasyRL 蘑菇书完整 PDF（165MB，GitHub Release 资产）
# ---------------------------------------------------------------
EASYRL="$ASSETS/EasyRL蘑菇书_v1.0.6.pdf"
if [ -f "$EASYRL" ]; then
  echo "[skip] EasyRL 蘑菇书已存在：$(du -h "$EASYRL" | cut -f1)"
else
  echo "[down] EasyRL蘑菇书_v1.0.6.pdf (165MB) ..."
  curl -L --progress-bar -o "$EASYRL.tmp" \
    "https://github.com/datawhalechina/easy-rl/releases/download/v1.0.6/EasyRL_v1.0.6.pdf"
  mv "$EASYRL.tmp" "$EASYRL"
  echo "[done] $(du -h "$EASYRL" | cut -f1)"
fi

# ---------------------------------------------------------------
# 2. 赵世钰《强化学习的数学原理》全本（19MB）
# ---------------------------------------------------------------
MATH="$ASSETS/强化学习的数学原理_赵世钰_全本.pdf"
if [ -f "$MATH" ]; then
  echo "[skip] 数学原理全本已存在：$(du -h "$MATH" | cut -f1)"
else
  echo "[down] 强化学习的数学原理_赵世钰_全本.pdf (19MB) ..."
  curl -sL --progress-bar -o "$MATH.tmp" \
    "https://raw.githubusercontent.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/main/Book-all-in-one.pdf"
  mv "$MATH.tmp" "$MATH"
  echo "[done] $(du -h "$MATH" | cut -f1)"
fi

# ---------------------------------------------------------------
# 3. 论文（若缺失）
# ---------------------------------------------------------------
mkdir -p "$ASSETS/论文"
declare -A PAPERS=(
  ["2306.13649.pdf"]="https://arxiv.org/pdf/2306.13649"
  ["2306.08543.pdf"]="https://arxiv.org/pdf/2306.08543"
  ["2604.00626_OPD综述_腾讯.pdf"]="https://arxiv.org/pdf/2604.00626"
  ["2604.13016_Rethinking_OPD.pdf"]="https://arxiv.org/pdf/2604.13016"
)
for name in "${!PAPERS[@]}"; do
  p="$ASSETS/论文/$name"
  if [ -f "$p" ]; then
    echo "[skip] $name"
  else
    echo "[down] 论文/$name ..."
    curl -sL -o "$p" "${PAPERS[$name]}"
  fi
done

# ---------------------------------------------------------------
# 4. 动手学强化学习 21 章 notebook（若缺失）
# ---------------------------------------------------------------
NB="$ASSETS/动手学强化学习_notebooks"
if [ -d "$NB" ] && [ "$(ls -1 "$NB" | wc -l)" -ge 20 ]; then
  echo "[skip] 动手学强化学习 notebooks 已齐（$(ls -1 "$NB" | wc -l) 个文件）"
else
  mkdir -p "$NB"
  BASE="https://raw.githubusercontent.com/boyu-ai/Hands-on-RL/main"
  echo "[down] 动手学强化学习 notebooks ..."
  curl -sL -o "$NB/rl_utils.py" "$BASE/rl_utils.py"
  for ch in 2:多臂老虎机问题 3:马尔可夫决策过程 4:动态规划算法 5:时序差分算法 6:Dyna-Q算法 \
            7:DQN算法 8:DQN改进算法 9:策略梯度算法 10:Actor-Critic算法 11:TRPO算法 \
            12:PPO算法 13:DDPG算法 14:SAC算法 15:模仿学习 16:模型预测控制 \
            17:基于模型的策略优化 18:离线强化学习 19:目标导向的强化学习 \
            20:多智能体强化学习入门 21:多智能体强化学习进阶; do
    n="${ch%%:*}"; t="${ch#*:}"
    f="第${n}章-${t}.ipynb"
    curl -sL -o "$NB/$f" "$BASE/$(printf '%s' "$f" | sed 's/ /%20/g')"
  done
  echo "[done] $(ls -1 "$NB" | wc -l) 个文件"
fi

echo
echo "==> 全部就绪"
du -sh "$ASSETS"/* 2>/dev/null | sort -h
