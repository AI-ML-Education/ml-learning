"""Shared toy environments for the deep RL lessons (files 06+).

Pure NumPy, no gym dependency, so every lesson stays self-contained and fast.
Each environment mimics the classic control benchmarks closely enough to learn on:

- CartPole    : the canonical discrete-action balancing task (for DQN, REINFORCE, A2C, PPO).
- PointMass1D : a simple continuous-action task (for DDPG / TD3 / SAC).
- Bandit      : a stateless multi-armed bandit (for the exploration lesson).

These are deliberately small so training loops finish in seconds on a laptop CPU.
Import them with importlib from the digit-prefixed lesson files, e.g.:

    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "envs", os.path.join(os.path.dirname(__file__), "_environments.py"))
    envs = importlib.util.module_from_spec(spec); spec.loader.exec_module(envs)
    env = envs.CartPole()
"""

import numpy as np


class CartPole:
    """A NumPy port of the classic CartPole-v1 dynamics (Barto, Sutton & Anderson).

    State  : [cart_position, cart_velocity, pole_angle, pole_angular_velocity]  (4 floats)
    Action : 0 = push left, 1 = push right                                      (discrete)
    Reward : +1 for every timestep the pole stays up.
    Done   : pole angle > 12 degrees, cart leaves the track, or 500 steps reached.

    Solving = average return >= ~475 over 100 episodes. Even reaching ~200 shows learning.
    """

    def __init__(self, seed=0, max_steps=500):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5            # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02              # seconds between state updates
        self.theta_threshold = 12 * 2 * np.pi / 360   # 12 degrees in radians
        self.x_threshold = 2.4
        self.max_steps = max_steps

        self.state_dim = 4
        self.n_actions = 2
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.steps = 0

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta, sintheta = np.cos(theta), np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

        # semi-implicit Euler integration
        x += self.tau * x_dot
        x_dot += self.tau * x_acc
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        done = bool(
            x < -self.x_threshold or x > self.x_threshold
            or theta < -self.theta_threshold or theta > self.theta_threshold
            or self.steps >= self.max_steps)
        reward = 1.0
        return self.state.copy(), reward, done


class PointMass1D:
    """A minimal continuous-control task: push a point mass to the origin and stay there.

    State  : [position, velocity]                          (2 floats)
    Action : a single continuous force in [-1, 1]          (1 float)
    Reward : -(position^2 + 0.1*velocity^2 + 0.001*action^2)   (closer/calmer = better)
    Done   : after max_steps (continuing-style episodic task).

    Good for DDPG/TD3/SAC because the action is continuous and the optimum is a smooth
    controller (push toward the origin, then brake).
    """

    def __init__(self, seed=0, max_steps=80, dt=0.1):
        self.state_dim = 2
        self.action_dim = 1
        self.action_low, self.action_high = -1.0, 1.0
        self.max_steps = max_steps
        self.dt = dt
        self.pos_limit = 2.0                # bounded track so returns stay finite
        self.vel_limit = 3.0
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.steps = 0

    def reset(self):
        pos = self.rng.uniform(-1.0, 1.0)
        vel = self.rng.uniform(-0.3, 0.3)
        self.state = np.array([pos, vel])
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        action = float(np.asarray(action).reshape(-1)[0])   # accept scalar or 1-elem array
        action = float(np.clip(action, self.action_low, self.action_high))
        pos, vel = self.state
        vel = vel + action * self.dt        # force (acceleration) changes velocity
        vel = float(np.clip(vel, -self.vel_limit, self.vel_limit))
        pos = pos + vel * self.dt           # velocity changes position
        pos = float(np.clip(pos, -self.pos_limit, self.pos_limit))  # bounded track
        self.state = np.array([pos, vel])
        self.steps += 1
        reward = -(pos ** 2 + 0.1 * vel ** 2 + 0.001 * action ** 2)
        done = self.steps >= self.max_steps
        return self.state.copy(), reward, done


class Bandit:
    """A stationary k-armed Gaussian bandit (stateless) for the exploration lesson.

    Each arm a has a hidden true mean q*(a); pulling it returns a noisy reward
    ~ N(q*(a), 1). The agent must balance exploring arms vs exploiting the best-known one.
    """

    def __init__(self, k=10, seed=0):
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.q_star = self.rng.normal(0.0, 1.0, size=k)   # hidden true means
        self.optimal_action = int(np.argmax(self.q_star))

    def pull(self, a):
        return self.rng.normal(self.q_star[a], 1.0)


def run_greedy_episode(env, act_fn, max_steps=500):
    """Utility: roll out one episode with a deterministic act_fn(state)->action; return G."""
    s = env.reset()
    total = 0.0
    for _ in range(max_steps):
        a = act_fn(s)
        s, r, done = env.step(a)
        total += r
        if done:
            break
    return total


class TicTacToe:
    """A minimal, self-contained 3x3 Tic-Tac-Toe game for MCTS / AlphaZero (files 13, 14).

    Board is a length-9 numpy array: +1 = player X, -1 = player O, 0 = empty.
    Convention: the game is described from the CURRENT player's perspective via `to_play`
    (+1 or -1). This is a two-player, deterministic, perfect-information, zero-sum game.

    It is intentionally tiny (game tree < 10^4 states) so MCTS can solve it on a laptop and
    a small neural net can learn it by self-play in seconds.
    """

    WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),    # rows
                 (0, 3, 6), (1, 4, 7), (2, 5, 8),    # cols
                 (0, 4, 8), (2, 4, 6)]               # diagonals

    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.to_play = 1            # X moves first

    def clone(self):
        g = TicTacToe()
        g.board = self.board.copy()
        g.to_play = self.to_play
        return g

    def legal_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def play(self, move):
        """Apply a move for the current player and switch sides."""
        self.board[move] = self.to_play
        self.to_play *= -1

    def winner(self):
        """Return +1 (X), -1 (O), 0 (draw), or None (game not over)."""
        for a, b, c in self.WIN_LINES:
            line = self.board[a] + self.board[b] + self.board[c]
            if line == 3:
                return 1
            if line == -3:
                return -1
        if not (self.board == 0).any():
            return 0          # board full → draw
        return None           # game continues

    def is_terminal(self):
        return self.winner() is not None

    def render(self):
        symbols = {1: "X", -1: "O", 0: "."}
        rows = ["".join(symbols[self.board[r * 3 + c]] for c in range(3)) for r in range(3)]
        return "\n".join(rows)


if __name__ == "__main__":
    print("Smoke-testing shared environments...\n")

    cp = CartPole()
    s = cp.reset()
    print(f"CartPole reset state: {np.round(s, 3)}  (dim={cp.state_dim}, actions={cp.n_actions})")
    total = 0.0
    s = cp.reset()
    for _ in range(500):
        s, r, done = cp.step(cp.rng.integers(2))
        total += r
        if done:
            break
    print(f"CartPole random-policy return: {total:.0f} (random balances only briefly)\n")

    pm = PointMass1D()
    s = pm.reset()
    print(f"PointMass reset state: {np.round(s, 3)}  (action in [{pm.action_low}, {pm.action_high}])")
    total = run_greedy_episode(pm, lambda s: -np.sign(s[0]))   # naive "push toward origin"
    print(f"PointMass naive-controller return: {total:.2f}\n")

    b = Bandit(k=10)
    print(f"Bandit with {b.k} arms; best arm is #{b.optimal_action} "
          f"with true mean {b.q_star[b.optimal_action]:.2f}")
