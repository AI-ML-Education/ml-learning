# AlphaZero & MuZero — Self-Play, Search + Learning Combined
"""
WHAT/WHY:
AlphaZero is the crown jewel of model-based RL: starting from ZERO human knowledge (just the
rules), it reached superhuman play in Go, Chess, and Shogi purely by PLAYING ITSELF. It fuses
two threads from this folder:
  - MCTS (file 13): look-ahead search to choose strong moves.
  - Deep RL (files 06–10): a neural net that learns from experience.

The key insight: SEARCH and LEARNING bootstrap each other.
  - The network guides the search (so it doesn't waste time on bad moves).
  - The search produces better moves than the raw network (it "looks ahead").
  - Those improved moves become TARGETS to train the network → the network gets better →
    the search gets better → ... a virtuous cycle, with NO human games as data.

We implement a compact but complete AlphaZero on Tic-Tac-Toe and watch it learn from self-play.

---

1. THE NETWORK: ONE BODY, TWO HEADS

    f_θ(state) → ( p, v )
  - POLICY head p: a probability over moves ("which moves look good here?").
  - VALUE  head v ∈ [-1, 1]: "who is winning from here?" (+1 current player wins, -1 loses).

This single network replaces BOTH expensive parts of classical MCTS (file 13):
  - p replaces the need to try every move equally (guides which to explore).
  - v replaces the RANDOM ROLLOUT — evaluate a leaf instantly, no playout to the end. (This is
    why vanilla MCTS's tactical blunders from file 13 disappear.)

---

2. MCTS WITH A NETWORK: PUCT

Selection uses the network's prior p(a) via the PUCT formula:

    score(a) = Q(a)  +  c_puct · P(a) · sqrt(ΣN) / (1 + N(a))

  - Q(a): mean value of action a from search so far (exploit).
  - P(a): the network's prior probability for a (guides exploration toward good moves).
  - The visit term explores less-tried moves; it shrinks as N(a) grows.

At a leaf, instead of a random rollout, we ASK THE NETWORK for (p, v), use p as the children's
priors, and BACK UP v along the path (negating each ply, as in file 13). After a budget of
simulations, the move is chosen ∝ visit counts N(a).

---

3. THE SELF-PLAY → TRAIN LOOP (the whole algorithm)

    repeat:
      SELF-PLAY: play games against yourself; at each move run MCTS and record
          (state, π = normalized visit counts, player_to_move)
        when the game ends with outcome z, label every stored state with z from THAT
        player's perspective.
      TRAIN: fit the network on the collected (state, π, z):
          policy loss = cross-entropy( p_θ(state), π )      # imitate the SEARCH's move dist
          value  loss = ( v_θ(state) − z )^2                # predict the game outcome
          loss = value_loss + policy_loss  (+ weight decay)
      (the improved network is used for the next round of self-play)

The targets are MCTS's improved policy (π) and the actual game result (z). No human data, no
reward shaping — just the rules and self-play. This is "policy iteration" (file 03) where MCTS
is the policy-improvement operator and the net is the function approximator.

---

4. EXPLORATION DETAILS THAT MATTER

  - DIRICHLET NOISE added to the root prior during self-play → forces the agent to try varied
    openings (otherwise it collapses to one line and never discovers better ones).
  - TEMPERATURE on the visit-count move selection: high early (explore different moves), then
    →0 (play the best) — for the first few moves of each self-play game.

---

5. MuZero — ALPHAZERO WITHOUT THE RULES (the next leap)

AlphaZero still needs a perfect SIMULATOR (it must know the game rules to expand the tree).
MuZero removes even that: it LEARNS a model and searches inside that learned model. Three
networks:
  - REPRESENTATION h: observation → latent state s⁰.
  - DYNAMICS g: (latent state, action) → (next latent state, predicted reward).
  - PREDICTION f: latent state → (policy, value).
MCTS runs entirely in LATENT space. MuZero never predicts pixels or true states — only the
quantities needed for planning (reward, value, policy). This lets it master Atari AND board
games with one algorithm, without being told the rules. It's the bridge from "games with known
rules" toward general model-based RL.

---

MATH SUMMARY:
  net:     f_θ(s) = (p, v)
  PUCT:    a = argmax_a  Q(a) + c·P(a)·√(ΣN)/(1+N(a))
  targets: π = visit-count distribution (policy), z = game outcome (value)
  loss:    L = (v_θ − z)^2 − π·log p_θ  + λ‖θ‖²

GOTCHAS:
- Value perspective: store/learn v from the CURRENT player's view; negate across plies.
- Use VISIT COUNTS (not Q) as the policy target — far more stable.
- Without Dirichlet noise / temperature, self-play collapses to a narrow set of games.
- Too few simulations → weak targets → the net learns slowly or wrongly.
- AlphaZero needs a perfect simulator; if you don't have one, that's MuZero's job.

INTUITION:
AlphaZero = "let search and a neural net teach each other from self-play." Search looks ahead
to produce moves better than the net's instinct; the net learns those moves so its instinct
improves; repeat. MuZero does the same after first LEARNING the rules it plans with.
"""

import importlib.util
import math
import os

import numpy as np
import torch
import torch.nn as nn

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("envs", os.path.join(_here, "_environments.py"))
_envs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_envs)
TicTacToe = _envs.TicTacToe


class AlphaZeroNet(nn.Module):
    """Tiny two-headed net: board(9) -> policy logits(9) and value scalar in [-1,1]."""

    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(9, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.policy_head = nn.Linear(hidden, 9)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.policy_head(h), torch.tanh(self.value_head(h)).squeeze(-1)


def encode(game):
    """Board from the CURRENT player's perspective (so the net is side-agnostic)."""
    return (game.board * game.to_play).astype(np.float32)


def net_predict(net, game):
    """Return (priors over legal moves as a dict, value in [-1,1]) from the current player's view."""
    with torch.no_grad():
        logits, v = net(torch.tensor(encode(game)).unsqueeze(0))
        logits = logits.squeeze(0).numpy()
        v = float(v.item())
    legal = game.legal_moves()
    mask = np.full(9, -1e9)
    mask[legal] = logits[legal]
    e = np.exp(mask - mask.max())
    probs = e / e.sum()
    return {m: float(probs[m]) for m in legal}, v


class AZNode:
    def __init__(self, prior):
        self.N = 0
        self.W = 0.0
        self.P = prior
        self.children = {}      # move -> AZNode
        self.game = None        # set when expanded

    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0


def az_mcts(net, root_game, n_sims=40, c_puct=1.5, dirichlet=0.3, eps=0.25, rng=None):
    """Network-guided MCTS. Returns visit-count distribution over moves at the root."""
    rng = np.random.default_rng() if rng is None else rng
    root = AZNode(prior=1.0)
    root.game = root_game.clone()
    priors, _ = net_predict(net, root.game)
    # add Dirichlet noise to root priors for exploration during self-play
    moves = list(priors.keys())
    noise = rng.dirichlet([dirichlet] * len(moves))
    for m, nz in zip(moves, noise):
        root.children[m] = AZNode(prior=(1 - eps) * priors[m] + eps * nz)

    for _ in range(n_sims):
        node = root
        path = [node]
        # --- SELECTION via PUCT until we reach an unexpanded node ---
        while node.children:
            total_N = sum(ch.N for ch in node.children.values())
            best, best_score = None, -1e18
            for m, ch in node.children.items():
                score = ch.Q() + c_puct * ch.P * math.sqrt(total_N + 1) / (1 + ch.N)
                if score > best_score:
                    best_score, best = score, m
            parent = node
            node = node.children[best]
            if node.game is None:           # lazily realize the child's game state
                g = parent.game.clone(); g.play(best)
                node.game = g
            path.append(node)
            if node.game.is_terminal() or not node.children:
                break

        # --- EVALUATION: terminal result, or the network's value (no rollout) ---
        g = node.game
        if g.is_terminal():
            w = g.winner()
            # value from the perspective of the player to move at g (who has no move => lost/drew)
            value = 0.0 if w == 0 else (1.0 if w == -g.to_play else -1.0)
        else:
            child_priors, value = net_predict(net, g)
            for m, p in child_priors.items():
                node.children[m] = AZNode(prior=p)

        # --- BACKPROP: negate value each ply going up ---
        for n in reversed(path):
            n.N += 1
            n.W += value
            value = -value

    counts = np.zeros(9)
    for m, ch in root.children.items():
        counts[m] = ch.N
    if counts.sum() == 0:                   # fallback (shouldn't happen)
        for m in root_game.legal_moves():
            counts[m] = 1
    return counts / counts.sum()


def self_play_game(net, n_sims, rng, temp_moves=3):
    """Play one self-play game; return list of (encoded_state, pi, player_to_move) + outcome."""
    g = TicTacToe()
    history = []
    move_num = 0
    while not g.is_terminal():
        pi = az_mcts(net, g, n_sims=n_sims, rng=rng)
        history.append((encode(g), pi, g.to_play))
        if move_num < temp_moves:           # temperature: sample early for variety
            move = rng.choice(9, p=pi)
        else:
            move = int(np.argmax(pi))       # then play the best
        g.play(int(move))
        move_num += 1
    z = g.winner()                          # +1 X, -1 O, 0 draw
    return history, z


def train_alphazero(iterations=15, games_per_iter=40, n_sims=60, epochs=4, lr=2e-3,
                    buffer_size=6000, seed=0):
    """The self-play → train loop with a replay buffer. Returns net and win-rate history."""
    from collections import deque
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = AlphaZeroNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    buffer = deque(maxlen=buffer_size)      # keep recent self-play across iterations

    history_winrate = []
    for it in range(iterations):
        # --- SELF-PLAY: add fresh games to the replay buffer ---
        for _ in range(games_per_iter):
            game_hist, z = self_play_game(net, n_sims, rng)
            for enc, pi, player in game_hist:
                buffer.append((enc, pi, z * player))    # outcome from that state's player's view

        states, target_pis, target_zs = zip(*buffer)
        S = torch.tensor(np.array(states), dtype=torch.float32)
        PI = torch.tensor(np.array(target_pis), dtype=torch.float32)
        Z = torch.tensor(np.array(target_zs), dtype=torch.float32)

        # --- TRAIN: policy = cross-entropy to MCTS visits, value = MSE to outcome ---
        n = len(S)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, 64):
                mb = idx[start:start + 64]
                logits, v = net(S[mb])
                logp = torch.log_softmax(logits, dim=1)
                policy_loss = -(PI[mb] * logp).sum(dim=1).mean()
                value_loss = nn.functional.mse_loss(v, Z[mb])
                loss = value_loss + policy_loss
                opt.zero_grad(); loss.backward(); opt.step()

        wr = evaluate_vs_random(net, n_games=100, seed=1000 + it)
        history_winrate.append(wr)
        print(f"  iter {it + 1:2d} | buffer {n:4d} | "
              f"raw-policy non-loss rate vs random: {wr:.0%}")
    return net, history_winrate


def evaluate_vs_random(net, n_games=100, seed=0):
    """Non-loss rate (win or draw) of the net's RAW policy vs a random opponent.

    Crucially we use the network's policy head DIRECTLY (no MCTS) so we measure what the
    NETWORK has learned, not what search can paper over. Search would mask the net's progress.
    """
    rng = np.random.default_rng(seed)
    non_losses = 0
    for i in range(n_games):
        g = TicTacToe()
        az_is_x = (i % 2 == 0)              # alternate sides for fairness
        while not g.is_terminal():
            az_turn = (g.to_play == 1) == az_is_x
            if az_turn:
                priors, _ = net_predict(net, g)             # raw policy head, no search
                move = max(priors, key=priors.get)
            else:
                legal = g.legal_moves()
                move = int(legal[rng.integers(len(legal))])
            g.play(int(move))
        w = g.winner()
        az_won = (w == 1 and az_is_x) or (w == -1 and not az_is_x)
        if az_won or w == 0:
            non_losses += 1
    return non_losses / n_games


if __name__ == "__main__":
    print("=== AlphaZero learning Tic-Tac-Toe from self-play (zero human games) ===\n")
    print("Measuring the NETWORK's raw policy (no search) so its learning is visible.\n")
    untrained = AlphaZeroNet()
    base = evaluate_vs_random(untrained, n_games=100, seed=7)
    print(f"Untrained net raw-policy non-loss rate vs random: {base:.0%}\n")

    net, wr_hist = train_alphazero()

    print(f"\nUntrained → trained raw-policy non-loss rate vs random: {base:.0%} → {wr_hist[-1]:.0%}")
    print(f"Best raw-policy non-loss rate during training: {max(wr_hist):.0%}")

    # The DEPLOYED AlphaZero uses the net TOGETHER WITH search. Measure that combined player.
    rng = np.random.default_rng(123)
    non_loss = 0
    for i in range(40):
        g = TicTacToe(); az_is_x = (i % 2 == 0)
        while not g.is_terminal():
            if (g.to_play == 1) == az_is_x:
                pi = az_mcts(net, g, n_sims=80, eps=0.0, rng=rng)   # net + search
                move = int(np.argmax(pi))
            else:
                legal = g.legal_moves(); move = int(legal[rng.integers(len(legal))])
            g.play(move)
        w = g.winner()
        if (w == 1 and az_is_x) or (w == -1 and not az_is_x) or w == 0:
            non_loss += 1
    print(f"Deployed AlphaZero (net + MCTS) non-loss rate vs random: {non_loss / 40:.0%}")
    print("\nKey lesson: the raw NET learned strong instincts from self-play; the net + SEARCH")
    print("together are near-perfect — search and learning amplifying each other.")
    print("\nThis is policy iteration (file 03) at scale: MCTS improves the policy, the net")
    print("learns it, repeat. MuZero does the same after LEARNING the rules to plan with.")
    print("Next (file 15): RLHF — the RL pipeline that aligns large language models.")
