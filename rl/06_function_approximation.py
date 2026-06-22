# Function Approximation — When Tables Stop Working
r"""
WHAT/WHY:
Everything in files 03–05 stored ONE number per state (or per state-action). That works for
16-cell GridWorlds. It collapses the moment states are continuous or astronomically many:

  - CartPole: state = 4 real numbers → infinitely many states. No table can index them.
  - Backgammon: ~10^20 states. Go: ~10^170. Atari from pixels: 256^(210·160·3) states.

The fix: stop storing values and start PREDICTING them with a parameterized function.

    Tabular:        V(s) = table[s]
    Approximation:  V(s) ≈ V_w(s) = f(s; w)     (w = weights of a linear model or neural net)

Now "learning" means adjusting a small weight vector w instead of a giant table. A bonus:
GENERALIZATION — updating w for one state automatically improves nearby, unseen states.
A table can't do that; an approximator interpolates.

---

1. THE PREDICTION OBJECTIVE

We want V_w(s) close to the true V_π(s). Minimize the (state-weighted) squared error:

    J(w) = E_s [ ( V_π(s) - V_w(s) )^2 ]

Gradient descent step:
    w ← w + α ( V_π(s) - V_w(s) ) ∇_w V_w(s)

Problem: we don't KNOW V_π(s). So we plug in a TARGET (just like file 05):
  - Monte Carlo target:  G_t            → w ← w + α (G_t - V_w(s)) ∇_w V_w(s)
  - TD(0) target:        r + γ V_w(s')  → w ← w + α (r + γ V_w(s') - V_w(s)) ∇_w V_w(s)

---

2. "SEMI-GRADIENT" — THE SUBTLE BUT CRUCIAL POINT

For the TD target r + γ V_w(s'), note the target ITSELF depends on w. A true gradient would
differentiate through V_w(s') too. We DON'T: we treat the target as a fixed constant and
only differentiate the prediction V_w(s). That's why it's called a SEMI-gradient method.

    w ← w + α [ r + γ V_w(s') - V_w(s) ] ∇_w V_w(s)
                \__ treated as a constant target __/

Why? It's cheaper and empirically works far better for control. The cost: combining
(1) function approximation + (2) bootstrapping + (3) off-policy training is the famous
"DEADLY TRIAD" that can cause divergence. On-policy + linear is provably safe; off-policy
+ nonlinear (e.g., DQN) needs the stabilizers you'll see in file 07.

---

3. LINEAR FUNCTION APPROXIMATION (the transparent case)

Represent each state by a FEATURE VECTOR x(s) ∈ R^d, then:

    V_w(s) = wᵀ x(s)          ⇒  ∇_w V_w(s) = x(s)   (gradient is just the features!)

So the update is beautifully simple:
    w ← w + α [ target - wᵀx(s) ] x(s)

Feature engineering matters here. Common schemes:
  - Polynomial / Fourier bases.
  - TILE CODING (a.k.a. CMAC): overlapping grids; each state activates a few binary tiles.
    Coarse coding that gives fast, stable, local generalization. We use a simple version.
  - Radial basis functions (Gaussian bumps).

---

4. NONLINEAR (NEURAL) FUNCTION APPROXIMATION

Let V_w (or Q_w) be a neural network. ∇_w is computed by backprop (autodiff). This is
"Deep RL". Same update rule, but:
  - Hugely expressive (learns its own features from raw inputs/pixels).
  - Loses the convergence guarantees → needs tricks (target networks, replay) in file 07.

We show both: a from-scratch LINEAR learner (so you see every number) and the SAME idea in
PyTorch with an MLP (so you see the autodiff version you'll use from here on).

---

5. CONTROL WITH APPROXIMATION

Swap V for Q and act ε-greedily, exactly like file 05 but parameterized:

    Q_w(s,a),   target = r + γ max_{a'} Q_w(s',a'),   w ← w + α[target - Q_w(s,a)]∇_w Q_w(s,a)

This "semi-gradient Q-learning" with a neural net + the file-07 stabilizers IS the DQN
algorithm. Function approximation is the bridge from tabular RL to everything modern.

---

MATH SUMMARY:
  Objective:   J(w) = E[(V_π - V_w)^2]
  Update:      w ← w + α (target - V_w(s)) ∇_w V_w(s)
  Linear:      V_w(s)=wᵀx(s),  ∇_w V_w(s)=x(s)
  Semi-grad:   target is held constant (not differentiated)

GOTCHAS:
- The deadly triad (approx + bootstrap + off-policy) can diverge — stabilizers needed.
- Feature scaling: unnormalized inputs make SGD crawl or explode. Normalize states.
- Too-large α with bootstrapping → oscillation/blow-up. Start small.
- Don't differentiate the bootstrap target (use semi-gradient / .detach()).
- Aliasing: if features map different states to the same vector, they share a value (bad).

INTUITION:
A table memorizes; an approximator generalizes. We trade exactness for the ability to
handle enormous/continuous state spaces — the only way RL scales to the real world.
"""

import importlib.util
import os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_here, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mdp = _load("mdp_module", "01_markov_decision_processes.py")
_dp = _load("dp_module", "03_dynamic_programming.py")
GridWorld = _mdp.GridWorld


# ----------------------------------------------------------------------------------
# A) LINEAR semi-gradient TD(0) on GridWorld, from scratch (no torch).
#    Features: one-hot per cell PLUS (row, col) coordinates — so the learner must
#    generalize across cells, not just memorize. We compare to exact DP values.
# ----------------------------------------------------------------------------------
def grid_features(env, s):
    """A small feature vector x(s): normalized (row, col, bias). Forces generalization
    because nearby cells share similar features (unlike a pure one-hot table)."""
    r, c = env._to_rc(s)
    return np.array([r / (env.rows - 1), c / (env.cols - 1), 1.0])


def linear_td_prediction(env, policy, n_episodes=4000, alpha=0.02, gamma=None):
    gamma = env.gamma if gamma is None else gamma
    d = len(grid_features(env, 0))
    w = np.zeros(d)
    rng = np.random.default_rng(0)
    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(200):
            a = rng.choice(env.n_actions, p=policy[s])
            s_next, r, done = env.step(a)
            x = grid_features(env, s)
            v_s = w @ x
            target = r if done else r + gamma * (w @ grid_features(env, s_next))
            w += alpha * (target - v_s) * x          # semi-gradient: x is ∇_w V_w(s)
            s = s_next
            if done:
                break
    V = np.array([w @ grid_features(env, s) for s in range(env.n_states)])
    return V, w


# ----------------------------------------------------------------------------------
# B) The SAME prediction with a small PyTorch MLP (nonlinear approximation).
#    This is the pattern every later deep-RL file uses: build target, MSE loss, step.
# ----------------------------------------------------------------------------------
def mlp_td_prediction(env, policy, n_episodes=4000, alpha=1e-3, gamma=None):
    import torch
    import torch.nn as nn

    gamma = env.gamma if gamma is None else gamma
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=alpha)
    rng = np.random.default_rng(0)

    def feat(s):
        return torch.tensor(grid_features(env, s), dtype=torch.float32)

    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(200):
            a = rng.choice(env.n_actions, p=policy[s])
            s_next, r, done = env.step(a)
            v_s = net(feat(s))
            with torch.no_grad():                      # target held constant (semi-grad)
                target = torch.tensor([[r]], dtype=torch.float32) if done \
                    else r + gamma * net(feat(s_next))
            loss = (v_s - target.detach()) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            s = s_next
            if done:
                break

    with torch.no_grad():
        V = np.array([net(feat(s)).item() for s in range(env.n_states)])
    return V


if __name__ == "__main__":
    print("=== Function approximation: predicting V without a table ===\n")
    env = GridWorld(slip=0.0)
    rand_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions

    V_dp = _dp.policy_evaluation(env, rand_policy)          # exact ground truth
    V_lin, w = linear_td_prediction(env, rand_policy)
    V_mlp = mlp_td_prediction(env, rand_policy)

    env.render_values(V_dp, title="Exact V (DP) for random policy:")
    env.render_values(V_lin, title="LINEAR approx V (3 features, semi-gradient TD):")
    env.render_values(V_mlp, title="NEURAL approx V (PyTorch MLP):")

    # Compare only on non-terminal cells (terminals have V=0 by definition and are
    # never bootstrapped toward, so including them would unfairly punish the MLP).
    nt = [s for s in range(env.n_states) if not env.is_terminal(s)]
    lin_err = np.mean([abs(V_dp[s] - V_lin[s]) for s in nt])
    mlp_err = np.mean([abs(V_dp[s] - V_mlp[s]) for s in nt])
    print(f"Learned linear weights w (row, col, bias): {np.round(w, 3)}")
    print(f"Linear  mean|V_DP - V_lin| over non-terminal cells = {lin_err:.3f}")
    print(f"Neural  mean|V_DP - V_mlp| over non-terminal cells = {mlp_err:.3f}")
    print("\nThe linear model can only fit a plane (3 features), so it's a coarse fit;")
    print("the MLP bends to match the true value surface more closely (lower mean error).")
    print("Next (file 07): neural Q-learning + stabilizers = Deep Q-Networks on CartPole.")
