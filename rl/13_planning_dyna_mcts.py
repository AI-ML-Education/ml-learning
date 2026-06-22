# Planning — Dyna-Q and Monte Carlo Tree Search (MCTS)
"""
WHAT/WHY:
So far we split the world into "model-based" (file 03, needs P and R) and "model-free"
(files 04–11, learn from samples). PLANNING is the bridge: use a MODEL of the environment —
either given or learned — to THINK AHEAD and improve decisions without extra real interaction.
Two landmark ideas:

  - DYNA-Q: learn a model from experience, then use it to generate IMAGINED experience that
    accelerates a model-free learner. (Integrates learning + planning.) Runnable on GridWorld.
  - MCTS: build a search tree by simulating rollouts, focusing on promising branches. The
    planning algorithm behind AlphaGo/AlphaZero (file 14). Runnable on Tic-Tac-Toe.

A MODEL answers "if I do a in s, what happens?" (next state + reward). With it, the agent can
simulate consequences in its head — like a chess player reading lines before moving.

================================ PART A: DYNA-Q ================================

1. THE IDEA: learn a model, then plan with it for free

Real environment steps can be scarce. Dyna-Q squeezes more out of each one:

    after each REAL step (s, a, r, s'):
        (1) DIRECT RL:  Q-learning update from the real transition          (file 05)
        (2) MODEL LEARN: record Model[s,a] = (r, s')                        (deterministic here)
        (3) PLANNING:    repeat n times:
                pick a random previously-seen (s, a)
                (r, s') = Model[s, a]                # imagined transition
                Q-learning update from the IMAGINED transition

So each real step triggers n extra "mental rehearsal" updates. With n planning steps, Dyna-Q
reaches a good policy in far FEWER real interactions than plain Q-learning.

INTUITION: after touching the stove once, you don't need to touch it 100 times — you REPLAY
the lesson in your head. (Experience replay in DQN is a model-FREE cousin of this idea.)

================================ PART B: MCTS ================================

2. THE IDEA: search where it matters

For large game trees you can't enumerate everything (file 03's DP is hopeless at Go's 10^170).
MCTS builds a search tree incrementally, spending simulations on the most promising moves.
Each simulation has FOUR phases:

  (1) SELECTION: from the root, descend by picking children that maximize UCT (a UCB variant):
          UCT(a) = Q(a) + c·sqrt( ln N(parent) / N(a) )
      Balances exploitation (high Q) vs exploration (rarely-visited a). (Same optimism as file 12.)
  (2) EXPANSION: at a leaf, add one (or more) new child node for an untried move.
  (3) SIMULATION (rollout): play out to the end with a default (e.g., random) policy; get the
      game outcome (+1 win / 0 draw / -1 loss).
  (4) BACKPROPAGATION: propagate that outcome up the path, updating N and Q of every node on it.

After a fixed budget of simulations, pick the root child with the highest visit count N (the
most-tested move — more robust than highest Q).

3. WHY VISIT COUNT, NOT VALUE?

The most-VISITED action is the one MCTS spent the most evidence on; it's less noisy than the raw
value estimate. AlphaZero even trains its policy network to imitate these visit-count
distributions (file 14).

4. FROM MCTS TO ALPHAZERO (the bridge to file 14)

Plain MCTS uses RANDOM rollouts to estimate a position's value — slow and weak. AlphaZero
replaces BOTH the rollout and the rollout policy with a NEURAL NET (value head v + policy head p):
  - p guides SELECTION (PUCT) so search focuses on good moves immediately.
  - v replaces the random rollout — evaluate a leaf instantly with the net, no playout.
That's the whole AlphaZero trick; this file builds the classical MCTS it upgrades.

---

MATH SUMMARY:
  Dyna-Q:  Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]   (from real AND imagined data)
  UCT:     select argmax_a  Q(a) + c·sqrt(ln N_parent / N_a)
  MCTS:    select → expand → simulate → backpropagate, ×budget; then play argmax_a N(a)

GOTCHAS:
- Dyna-Q's model here is deterministic (Model[s,a]=(r,s')); stochastic worlds need a
  distribution, or the imagined data is wrong.
- A WRONG model makes planning HARMFUL (you rehearse fantasies). Dyna-Q+ adds exploration
  bonuses to refresh stale model entries when the world changes.
- MCTS in two-player games must NEGATE values between plies (what's good for me is bad for you).
- UCT constant c trades search breadth vs depth; tune it.
- Random rollouts are high-variance — the reason AlphaZero swaps them for a value net.

INTUITION:
Planning = "use a model to rehearse in your imagination." Dyna-Q rehearses past transitions to
learn faster; MCTS rehearses futures to choose the best move now. Together they set up AlphaZero.
"""

import importlib.util
import math
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_here, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mdp = _load("mdp_module", "01_markov_decision_processes.py")
_envs = _load("envs", "_environments.py")
GridWorld = _mdp.GridWorld
TicTacToe = _envs.TicTacToe


# ============================ PART A: Dyna-Q ============================
def dyna_q(env, n_planning, n_episodes=50, alpha=0.5, gamma=None,
           eps=0.1, seed=0):
    """Tabular Dyna-Q. n_planning imagined updates per real step. Returns steps-per-episode."""
    gamma = env.gamma if gamma is None else gamma
    Q = np.zeros((env.n_states, env.n_actions))
    model = {}                                  # (s,a) -> (r, s', done)
    seen = []                                   # list of observed (s,a) keys
    rng = np.random.default_rng(seed)
    env.rng = np.random.default_rng(seed)
    steps_per_episode = []

    for _ in range(n_episodes):
        s = env.reset()
        steps = 0
        for _ in range(200):
            # ε-greedy action
            a = rng.integers(env.n_actions) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done = env.step(a)
            steps += 1

            # (1) direct RL from the real transition
            target = r if done else r + gamma * np.max(Q[s2])
            Q[s, a] += alpha * (target - Q[s, a])

            # (2) model learning
            if (s, a) not in model:
                seen.append((s, a))
            model[(s, a)] = (r, s2, done)

            # (3) planning: n imagined Q-updates from random remembered transitions
            for _ in range(n_planning):
                ps, pa = seen[rng.integers(len(seen))]
                pr, ps2, pdone = model[(ps, pa)]
                pt = pr if pdone else pr + gamma * np.max(Q[ps2])
                Q[ps, pa] += alpha * (pt - Q[ps, pa])

            s = s2
            if done:
                break
        steps_per_episode.append(steps)
    return steps_per_episode


# ============================ PART B: MCTS (UCT) ============================
class Node:
    """One game position in the MCTS tree."""

    def __init__(self, game, parent=None, move=None):
        self.game = game            # a TicTacToe state (current player = game.to_play)
        self.parent = parent
        self.move = move            # the move that led here (from parent)
        self.children = {}          # move -> Node
        self.N = 0                  # visit count
        self.W = 0.0                # total value (from the perspective of the player to move at parent)
        self.untried = game.legal_moves()

    def q(self):
        return self.W / self.N if self.N > 0 else 0.0


def uct_select(node, c=1.4):
    """Pick the child maximizing UCT."""
    return max(node.children.values(),
               key=lambda ch: ch.q() + c * math.sqrt(math.log(node.N) / ch.N))


def mcts_move(root_game, n_sims=400, c=1.4, rng=None):
    """Run MCTS from root_game and return the most-visited move."""
    rng = np.random.default_rng() if rng is None else rng
    root = Node(root_game.clone())

    for _ in range(n_sims):
        node = root
        # --- (1) SELECTION: descend fully-expanded nodes by UCT ---
        while not node.untried and node.children:
            node = uct_select(node, c)
        # --- (2) EXPANSION: add one new child for an untried move ---
        if node.untried:
            move = node.untried.pop(rng.integers(len(node.untried)))
            g = node.game.clone()
            g.play(move)
            child = Node(g, parent=node, move=move)
            node.children[move] = child
            node = child
        # --- (3) SIMULATION: random rollout to a terminal state ---
        g = node.game.clone()
        while not g.is_terminal():
            moves = g.legal_moves()
            g.play(moves[rng.integers(len(moves))])
        outcome = g.winner()        # +1 X wins, -1 O wins, 0 draw
        # --- (4) BACKPROPAGATION: update along the path, negating per ply ---
        n = node
        while n is not None:
            n.N += 1
            # value from the perspective of the player who just MOVED into node n
            mover = -n.game.to_play   # the player who made the move leading to n
            n.W += outcome * mover
            n = n.parent

    # choose the most-visited move (robust)
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]


def random_player(game, rng):
    moves = game.legal_moves()
    return moves[rng.integers(len(moves))]


def play_match(p_x, p_o, rng):
    """Play one game; p_x/p_o are functions (game, rng)->move. Return winner (+1/-1/0)."""
    g = TicTacToe()
    while not g.is_terminal():
        mover = p_x if g.to_play == 1 else p_o
        g.play(mover(g, rng))
    return g.winner()


if __name__ == "__main__":
    print("=== PART A: Dyna-Q — planning accelerates learning on GridWorld ===\n")
    print("Total REAL steps over the first 10 episodes (lower = learned faster),")
    print("averaged over 10 seeds. More planning per real step = fewer real steps needed.\n")
    for n_plan in [0, 5, 50]:
        totals = []
        for seed in range(10):
            steps = dyna_q(GridWorld(slip=0.0), n_planning=n_plan, n_episodes=10, seed=seed)
            totals.append(sum(steps))
        label = "Q-learning (no planning)" if n_plan == 0 else f"Dyna-Q ({n_plan} plan steps)"
        print(f"  {label:<28} | real steps (first 10 eps): {np.mean(totals):6.1f}")
    print("\nMore planning steps → fewer REAL interactions to find the short path.\n")

    print("=== PART B: MCTS plays Tic-Tac-Toe (no neural net, just search) ===\n")
    rng = np.random.default_rng(0)
    mcts_p = lambda g, r: mcts_move(g, n_sims=800, rng=r)
    rand_p = lambda g, r: random_player(g, r)

    # MCTS as X vs random O
    results = {"MCTS": 0, "random": 0, "draw": 0}
    for i in range(50):
        w = play_match(mcts_p, rand_p, np.random.default_rng(i))
        results["MCTS" if w == 1 else "random" if w == -1 else "draw"] += 1
    print(f"MCTS (X, 800 sims) vs Random (O), 50 games: "
          f"{results['MCTS']} wins, {results['draw']} draws, {results['random']} losses")

    # MCTS vs MCTS: TTT is a theoretical draw, but VANILLA MCTS uses RANDOM rollouts, which
    # misjudge sharp tactics — so a few games are still decisive. That weakness is exactly
    # what AlphaZero (file 14) removes by replacing rollouts with a learned value network.
    strong = lambda g, r: mcts_move(g, n_sims=1500, rng=r)
    draws = sum(play_match(strong, strong, np.random.default_rng(100 + i)) == 0 for i in range(20))
    print(f"Strong MCTS vs Strong MCTS (1500 sims), 20 games: {draws}/20 draws "
          f"(random rollouts aren't perfect — AlphaZero fixes this).")
    print("\nMCTS focuses search on promising lines. Next (file 14): replace random rollouts")
    print("with a NEURAL net (value + policy) trained by self-play = AlphaZero.")
