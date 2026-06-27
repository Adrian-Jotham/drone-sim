#!/bin/bash
# Sequential architecture sweep with the SAME (current) reward function.
# Trains PPO+GRU, SAC+MLP, SAC+GRU; evaluates each; writes a results table.
PY=/home/adrian/miniconda3/envs/newton/bin/python
cd /home/adrian/sim/dronesim
RES=/tmp/arch_results.txt
: > "$RES"

run() {  # algo policy num_envs frames curr
  local algo=$1 policy=$2 ne=$3 fr=$4 cur=$5
  local tag="${algo}_${policy}"
  echo "===== TRAIN $tag (envs=$ne frames=$fr) =====" | tee -a "$RES"
  $PY train_drone.py --algo $algo --policy $policy --num_envs $ne \
      --total_timesteps $fr --curriculum_steps $cur --rollouts 24 \
      --logdir /tmp/sweep > /tmp/sweep_${tag}.log 2>&1
  echo "----- EVAL $tag -----" | tee -a "$RES"
  $PY eval_drone.py --algo $algo --policy $policy \
      --model /tmp/sweep/${tag}_s0/checkpoints/best_agent.pt \
      --num_episodes 10 --steps_per_wp 250 2>&1 \
    | grep -iE 'Per-waypoint|All-waypoints|Mean waypoint|Mean motor|Mean ep len' \
    | tee -a "$RES"
  echo "" | tee -a "$RES"
}

run ppo gru  2048 40000000 12000000
run sac mlp  1024 20000000 8000000
run sac gru  1024 20000000 8000000
echo "ALL DONE" | tee -a "$RES"
