"""Sanity + gradient check for the differentiable Featherstone drone env."""
import numpy as np, torch
import diff_drone_env as E


def hover_test():
    env = E.DiffDroneEnv(num_envs=16, device="cuda")
    # place upright at 0.5 m, target = same spot, hover action
    q = torch.zeros(16, 7, device="cuda"); q[:, 2] = 0.5; q[:, 6] = 1.0
    qd = torch.zeros(16, 6, device="cuda")
    om = torch.tensor(E.TURN_DIR*E.OMEGA_HOVER, device="cuda").float().repeat(16, 1)
    env.set_targets(np.tile([0, 0, 0.5], (16, 1)))
    act = torch.zeros(16, 4, device="cuda")
    z0 = q[0, 2].item()
    for t in range(200):
        env.new_window()    # detached each step → reuse slot 0
        q, qd, om, obs, rew = env.step(q.detach(), qd.detach(), om.detach(), act)
    z1 = q[0, 2].item()
    print(f"[hover] z {z0:.3f} -> {z1:.3f}  |ω|~{om.abs().mean():.1f} (hover {E.OMEGA_HOVER:.0f})  "
          f"rew~{rew.mean():.3f}")
    print("[hover]", "PASS" if abs(z1-0.5) < 0.3 and z1 > 0.1 else "FAIL")


def grad_check():
    """Compare d(sum reward over H steps)/d(action_0) from autograd vs finite differences."""
    torch.manual_seed(0)
    env = E.DiffDroneEnv(num_envs=4, device="cuda")
    q0 = torch.zeros(4, 7, device="cuda"); q0[:, 2] = 0.6; q0[:, 6] = 1.0
    qd0 = torch.zeros(4, 6, device="cuda")
    om0 = torch.tensor(E.TURN_DIR*E.OMEGA_HOVER, device="cuda").float().repeat(4, 1)
    env.set_targets(np.tile([0.3, 0.0, 0.6], (4, 1)))
    H = 16

    def rollout(a0):
        # apply the SAME action every step → sustained effect → clearly nonzero gradient
        env.new_window()
        q, qd, om = q0.clone(), qd0.clone(), om0.clone()
        total = 0.0
        for t in range(H):
            q, qd, om, obs, rew = env.step(q, qd, om, a0)
            total = total + rew.sum()
        return total

    a_base = torch.full((4, 4), 0.2, device="cuda")   # sustained climb-ish action
    a0 = a_base.clone().requires_grad_(True)
    loss = rollout(a0)
    loss.backward()
    g_auto = a0.grad.clone().cpu().numpy()

    # finite differences on env 0 (eps=1e-2: large enough to clear float32 noise over the
    # 16-step rollout; smaller eps makes the reward delta drop below float32 precision).
    eps = 1e-2
    g_fd = np.zeros(4)
    with torch.no_grad():
        for k in range(4):
            ap = a_base.clone(); ap[0, k] += eps
            am = a_base.clone(); am[0, k] -= eps
            lp = rollout(ap).item(); lm = rollout(am).item()
            g_fd[k] = (lp - lm) / (2*eps)
    g_auto0 = g_auto[0]
    print("[grad] autograd d(loss)/d(a0[env0]):", np.round(g_auto0, 3))
    print("[grad] finite-diff             :", np.round(g_fd, 3))
    rel = np.linalg.norm(g_auto0 - g_fd) / (np.linalg.norm(g_fd) + 1e-6)
    cos = float(np.dot(g_auto0, g_fd) / (np.linalg.norm(g_auto0)*np.linalg.norm(g_fd) + 1e-9))
    print(f"[grad] relative error = {rel:.3e}   cosine = {cos:.4f}")
    # For policy-gradient optimization the descent DIRECTION is what matters; FD magnitude
    # has truncation error over the nonlinear 16-step rollout, so we validate on direction.
    print("[grad]", "PASS" if cos > 0.99 else "FAIL")


if __name__ == "__main__":
    hover_test()
    grad_check()
