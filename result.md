## Base Reward from paper
 ep   1/10 | rew=  220.50 | len= 574 | rpm≈14281 | wp1(✓,0.15m)  wp2(✓,0.15m)  wp3(✗,0.53m)  wp4(✗,1.73m) | [2/4]
  ep   2/10 | rew=  350.15 | len= 800 | rpm≈14464 | wp1(✗,0.41m)  wp2(✗,0.19m)  wp3(✗,0.21m)  wp4(✗,0.16m) | [0/4]
  ep   3/10 | rew=  182.96 | len= 474 | rpm≈14678 | wp1(✓,0.15m)  wp2(✓,0.15m)  wp3(✗,1.03m)  wp4(✗,infm) | [2/4]
  ep   4/10 | rew=  265.13 | len= 729 | rpm≈14649 | wp1(✓,0.15m)  wp2(✗,0.39m)  wp3(✗,0.16m)  wp4(✗,1.50m) | [1/4]
  ep   5/10 | rew=  312.55 | len= 752 | rpm≈14504 | wp1(✗,0.46m)  wp2(✓,0.14m)  wp3(✗,1.38m)  wp4(✗,0.33m) | [1/4]
  ep   6/10 | rew=  321.95 | len= 800 | rpm≈14657 | wp1(✗,0.53m)  wp2(✗,0.44m)  wp3(✗,1.33m)  wp4(✗,1.39m) | [0/4]
  ep   7/10 | rew=  324.02 | len= 723 | rpm≈14370 | wp1(✓,0.15m)  wp2(✓,0.15m)  wp3(✓,0.15m)  wp4(✗,0.23m) | [3/4]
  ep   8/10 | rew=  271.71 | len= 607 | rpm≈14383 | wp1(✓,0.15m)  wp2(✓,0.15m)  wp3(✗,0.75m)  wp4(✓,0.13m) | [3/4]
  ep   9/10 | rew=  330.02 | len= 774 | rpm≈14498 | wp1(✓,0.15m)  wp2(✗,0.19m)  wp3(✗,0.75m)  wp4(✗,0.28m) | [1/4]
  ep  10/10 | rew=  300.30 | len= 669 | rpm≈14389 | wp1(✓,0.15m)  wp2(✓,0.14m)  wp3(✓,0.15m)  wp4(✓,0.14m) | [ALL✓]

──────────────────────────────────────────────────────────────────
  Evaluation summary  [PPO]  — Level-5.1 RPM control
──────────────────────────────────────────────────────────────────
  Episodes            : 10
  Waypoints / ep      : 4  (random, r=0.5–1.5 m, z=0.3–1.2 m)
  Step budget / wp    : 200 steps = 2.0 s
  Mean reward         : 287.93 ± 50.21
  Mean ep length      : 690.2 steps
  Mean wpts reached   : 1.70 / 4
  Per-waypoint succ   : 42.5%  (dist < 0.15 m)
  All-waypoints succ  : 10.0%  (all wpts hit)
  Mean waypoint dist  : 0.4325 m
  Mean motor RPM      : 14487 RPM  (100.1% of hover RPM 14476)
  Max motor RPM       : 21702 RPM
──────────────────────────────────────────────────────────────────

## BASE REWARD + GATED APPROACH
  ep   1/10 | rew=  179.04 | len= 477 | rpm≈13981 | wp1(✗,0.24m)  wp2(✗,0.27m)  wp3(✗,2.46m)  wp4(✗,infm) | [0/4]
  ep   2/10 | rew=  352.98 | len= 794 | rpm≈14126 | wp1(✗,0.30m)  wp2(✗,0.24m)  wp3(✗,0.29m)  wp4(✓,0.15m) | [1/4]
  ep   3/10 | rew=  257.69 | len= 639 | rpm≈14014 | wp1(✗,0.22m)  wp2(✓,0.14m)  wp3(✗,0.40m)  wp4(✗,2.20m) | [1/4]
  ep   4/10 | rew=  281.10 | len= 684 | rpm≈14260 | wp1(✓,0.15m)  wp2(✗,0.40m)  wp3(✓,0.15m)  wp4(✓,0.15m) | [3/4]
  ep   5/10 | rew=  330.15 | len= 759 | rpm≈14085 | wp1(✗,0.28m)  wp2(✓,0.15m)  wp3(✗,0.30m)  wp4(✗,0.15m) | [1/4]
  ep   6/10 | rew=  367.42 | len= 800 | rpm≈13899 | wp1(✗,0.29m)  wp2(✗,0.23m)  wp3(✗,0.27m)  wp4(✗,0.19m) | [0/4]
  ep   7/10 | rew=  298.12 | len= 684 | rpm≈14233 | wp1(✗,0.21m)  wp2(✓,0.15m)  wp3(✓,0.14m)  wp4(✗,0.31m) | [2/4]
  ep   8/10 | rew=  319.82 | len= 694 | rpm≈13991 | wp1(✓,0.15m)  wp2(✓,0.15m)  wp3(✗,0.26m)  wp4(✗,0.23m) | [2/4]
  ep   9/10 | rew=  316.63 | len= 745 | rpm≈14208 | wp1(✗,0.37m)  wp2(✗,0.60m)  wp3(✓,0.15m)  wp4(✗,0.21m) | [1/4]
  ep  10/10 | rew=  344.49 | len= 756 | rpm≈13933 | wp1(✓,0.15m)  wp2(✗,0.26m)  wp3(✗,0.36m)  wp4(✓,0.15m) | [2/4]

──────────────────────────────────────────────────────────────────
  Evaluation summary  [PPO]  — Level-5.1 RPM control
──────────────────────────────────────────────────────────────────
  Episodes            : 10
  Waypoints / ep      : 4  (random, r=0.5–1.5 m, z=0.3–1.2 m)
  Step budget / wp    : 200 steps = 2.0 s
  Mean reward         : 304.74 ± 52.41
  Mean ep length      : 703.2 steps
  Mean wpts reached   : 1.30 / 4
  Per-waypoint succ   : 32.5%  (dist < 0.15 m)
  All-waypoints succ  : 0.0%  (all wpts hit)
  Mean waypoint dist  : 0.3454 m
  Mean motor RPM      : 14073 RPM  (97.2% of hover RPM 14476)
  Max motor RPM       : 21702 RPM
──────────────────────────────────────────────────────────────────

## BASE REWARD + GATED APPROACH + VELOCITY-gated hover bonus
  ep   1/10 | rew=  219.73 | len= 539 | rpm≈14432 | wp1(✓,0.15m)  wp2(✗,0.17m)  wp3(✗,0.66m)  wp4(✗,infm) | [1/4]
  ep   2/10 | rew=  269.27 | len= 623 | rpm≈14295 | wp1(✗,0.18m)  wp2(✓,0.14m)  wp3(✓,0.15m)  wp4(✓,0.15m) | [3/4]
  ep   3/10 | rew=  296.72 | len= 724 | rpm≈14569 | wp1(✓,0.15m)  wp2(✓,0.15m)  wp3(✗,0.19m)  wp4(✗,0.27m) | [2/4]
  ep   4/10 | rew=  327.61 | len= 754 | rpm≈14414 | wp1(✓,0.15m)  wp2(✗,0.16m)  wp3(✓,0.15m)  wp4(✓,0.15m) | [3/4]
  ep   5/10 | rew=  324.99 | len= 750 | rpm≈14496 | wp1(✗,0.19m)  wp2(✗,0.19m)  wp3(✓,0.15m)  wp4(✗,0.26m) | [1/4]
  ep   6/10 | rew=  279.98 | len= 620 | rpm≈14383 | wp1(✓,0.15m)  wp2(✓,0.15m)  wp3(✓,0.15m)  wp4(✓,0.15m) | [ALL✓]
  ep   7/10 | rew=  316.36 | len= 705 | rpm≈14407 | wp1(✓,0.15m)  wp2(✗,0.27m)  wp3(✓,0.15m)  wp4(✓,0.15m) | [3/4]
  ep   8/10 | rew=  298.35 | len= 641 | rpm≈14359 | wp1(✓,0.15m)  wp2(✓,0.15m)  wp3(✗,0.21m)  wp4(✓,0.15m) | [3/4]
  ep   9/10 | rew=  280.97 | len= 653 | rpm≈14207 | wp1(✗,0.17m)  wp2(✓,0.15m)  wp3(✓,0.15m)  wp4(✗,0.17m) | [2/4]
  ep  10/10 | rew=  298.38 | len= 660 | rpm≈14349 | wp1(✓,0.15m)  wp2(✓,0.15m)  wp3(✓,0.15m)  wp4(✓,0.15m) | [ALL✓]

──────────────────────────────────────────────────────────────────
  Evaluation summary  [PPO]  — Level-5.1 RPM control
──────────────────────────────────────────────────────────────────
  Episodes            : 10
  Waypoints / ep      : 4  (random, r=0.5–1.5 m, z=0.3–1.2 m)
  Step budget / wp    : 200 steps = 2.0 s
  Mean reward         : 291.24 ± 30.09
  Mean ep length      : 666.9 steps
  Mean wpts reached   : 2.60 / 4
  Per-waypoint succ   : 65.0%  (dist < 0.15 m)
  All-waypoints succ  : 20.0%  (all wpts hit)
  Mean waypoint dist  : 0.1779 m
  Mean motor RPM      : 14391 RPM  (99.4% of hover RPM 14476)
  Max motor RPM       : 21702 RPM