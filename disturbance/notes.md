Good — all three files parse clean, and the landing scripts only import from drone_landing_env (which already has the updated import). Everything is in order.

Here's a summary of everything that was done:

What changed
New files (replace the old ones):

New	Replaces	Key additions
quadrotor_hover_env.py	drone_gym_env.py	OU wind model, position trail, wind indicator, param docs fixed
train_hover.py	train_drone.py	--wind_scale arg, wind metric logging
eval_hover.py	eval_drone.py	--mode disturbance eval, --wind_scale, trail via render()
Modified: drone_landing_env.py — import updated to quadrotor_hover_env.

You can delete: drone_gym_env.py, train_drone.py, eval_drone.py

How to use disturbance rejection
Train with wind:


python train_hover.py --algo sac --wind_scale 0.8
Evaluate with disturbance mode + viewer (shows trail + wind indicator):


python eval_hover.py --model sac_hover_final --algo sac --mode disturbance --wind_scale 0.8
What you'll see in the viewer:

Cyan trail — position history fading from dark (old) to bright (recent). A tight trail = good rejection; a drifting comet = the policy is being pushed around.
Orange/red sphere — wind force indicator. Placed 0.6 m in the wind direction from the drone, sized and reddened by force magnitude.
GRU — how to add it when ready
For PPO, swap in RecurrentPPO from sb3-contrib:


pip install sb3-contrib

from sb3_contrib import RecurrentPPO
model = RecurrentPPO(
    "MlpLstmPolicy", train_env,   # or "MlpGruPolicy"
    n_steps=2048, batch_size=64,
    policy_kwargs=dict(
        n_lstm_layers=1,
        lstm_hidden_size=128,
        net_arch=[256, 256],
    ),
    ...
)
For SAC/TD3, increase N_ACTION_HIST from 1 to 4 in quadrotor_hover_env.py:90 — obs grows from 22-D to 34-D but training stays simple.