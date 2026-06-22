# Value Functions & the Bellman Equations — The Math RL Is Built On
r"""
WHAT/WHY:
A "value function" answers the question every RL agent secretly asks: "How good is this
situation?" If we knew the true value of every state (or every state-action pair), acting
optimally would be easy — just move toward higher value. Almost all of RL is about
*estimating* these value functions.

The Bellman equations are the recursive relationships that value functions must satisfy.
They are THE central equations of RL: every algorithm in files 03–05 (and DQN, etc.) is a
way of solving or approximating a Bellman equation.

---

1. TWO VALUE FUNCTIONS

State-value function V_π(s):
    "Expected return starting from state s, then following policy π forever."
        V_π(s) = E_π [ G_t | s_t = s ]

Action-value function Q_π(s, a):
    "Expected return starting from s, taking action a now, then following π afterward."
        Q_π(s, a) = E_π [ G_t | s_t = s, a_t = a ]

Relationship between them:
        V_π(s) = Σ_a π(a|s) · Q_π(s, a)          (average Q over the policy's actions)
        Q_π(s, a) = Σ_{s'} P(s'|s,a) [ R + γ·V_π(s') ]   (one step, then bootstrap on V)

WHY TWO? V tells you how good a state is; Q tells you how good each *action* is, which is
what you need to actually choose actions without a model. Q is "action-ready".

---

2. THE BELLMAN EXPECTATION EQUATION (for a FIXED policy π — "prediction")

The value of a state = immediate reward + discounted value of where you land.

    V_π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ·V_π(s') ]

    Q_π(s,a) = Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ Σ_{a'} π(a'|s') Q_π(s',a') ]

This is a *self-referential* (recursive) definition: V appears on both sides. That
recursion is called BOOTSTRAPPING — estimating a value from other estimated values.

INTUITION: "My position's worth = what I earn next step + (discounted) worth of my next
position." Like valuing a job by next month's paycheck plus the value of still having the
job afterward.

---

3. THE BELLMAN OPTIMALITY EQUATION (for the BEST policy π* — "control")

Instead of averaging over a policy, take the MAX over actions (act greedily):

    V*(s)   = max_a  Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ·V*(s') ]

    Q*(s,a) =        Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ· max_{a'} Q*(s',a') ]

Once you have Q*, the optimal policy is trivial — be greedy:

    π*(s) = argmax_a  Q*(s, a)

This is the whole game: find Q* (or V*), then act greedily with respect to it.

KEY DIFFERENCE:
  - Expectation equation: Σ_a π(a|s) ...   (average — evaluates a given policy)
  - Optimality equation:  max_a ...        (best — finds the optimal policy)

---

4. WHY THE BELLMAN EQUATION IS SOLVABLE (the contraction idea)

Define the Bellman OPTIMALITY OPERATOR T applied to any value estimate V:

    (T V)(s) = max_a Σ_{s'} P(s'|s,a) [ R + γ V(s') ]

Fact: T is a γ-contraction in max-norm. Repeatedly applying T (V ← T V) shrinks the error
by a factor γ each time, so V converges to the unique fixed point V*. This is exactly why
Value Iteration (file 03) works — it just hammers V with T until it stops changing.

    ||T V1 - T V2||∞  ≤  γ · ||V1 - V2||∞

---

5. THE BACKUP DIAGRAM (how all algorithms differ)

Every value-update is a "backup": pull information from successor states back to the
current one. Algorithms differ in (a) how deep they look and (b) whether they average or
take the max:

    Dynamic Programming : full-width (all s'), uses the model, bootstraps  -> file 03
    Monte Carlo         : full-depth (whole episode), sample, NO bootstrap -> file 04
    Temporal Difference : one step, sample, bootstraps                     -> file 05

This single table is the skeleton of classical RL. Keep it nearby.

---

6. THE TD ERROR (a preview of files 04–05)

When we can't use the full model, we sample one transition (s, a, r, s') and form the
Bellman equation as a *target* vs. our current *estimate*. The gap is the TD error:

    δ = [ r + γ·V(s') ]  -  V(s)
        \_____target____/   \_est_/

We then nudge our estimate toward the target:  V(s) ← V(s) + α·δ.
This one line is the engine of almost all model-free RL.

---

MATH SUMMARY (the four equations to know cold):
    V_π(s)   = Σ_a π(a|s) Σ_{s'} P[ R + γ V_π(s') ]          (Bellman expectation, V)
    Q_π(s,a) = Σ_{s'} P[ R + γ Σ_{a'} π(a'|s') Q_π(s',a') ]  (Bellman expectation, Q)
    V*(s)    = max_a Σ_{s'} P[ R + γ V*(s') ]                (Bellman optimality, V)
    Q*(s,a)  = Σ_{s'} P[ R + γ max_{a'} Q*(s',a') ]          (Bellman optimality, Q)

GOTCHAS:
- Mixing up V and Q (and which one you can act greedily on without a model — it's Q).
- Forgetting the γ on the bootstrapped term.
- Using max for evaluation (that's optimality, not the value of YOUR policy).
- Terminal states: V(terminal) = 0 by definition; never bootstrap past them.

INTUITION:
A value function is a "cached estimate of future reward". The Bellman equation says that
estimate must be self-consistent: today's value = today's reward + tomorrow's value.
RL = make the cache self-consistent (and, for control, greedy).
"""

import numpy as np


def bellman_expectation_demo():
    """Numerically verify the Bellman expectation equation on a tiny 3-state chain.

    Chain:  A --right--> B --right--> C(terminal, reward +1)
    Going 'left' is a self-loop (stay) with reward 0. Step reward is 0 except entering C.
    Policy: always go right (deterministic). We solve V_π exactly by linear algebra and
    confirm it satisfies V = r + γ P V.
    """
    gamma = 0.9
    # states: 0=A, 1=B, 2=C(terminal)
    # Under "always right": A->B (r=0), B->C (r=+1), C is terminal (absorbing, V=0).
    # Bellman: V(A) = 0 + γ V(B);  V(B) = 1 + γ V(C);  V(C) = 0
    V = np.zeros(3)
    V[2] = 0.0
    V[1] = 1.0 + gamma * V[2]
    V[0] = 0.0 + gamma * V[1]

    print("Bellman expectation on a 3-state chain (policy = always 'right'):")
    print(f"  V(C) = {V[2]:.4f}   (terminal)")
    print(f"  V(B) = 1 + γ·V(C) = {V[1]:.4f}")
    print(f"  V(A) = 0 + γ·V(B) = {V[0]:.4f}")

    # Self-consistency check: plug back into the equation.
    lhs_A = V[0]
    rhs_A = 0.0 + gamma * V[1]
    print(f"  check: V(A) == 0 + γ·V(B)?  {lhs_A:.4f} == {rhs_A:.4f}  -> {np.isclose(lhs_A, rhs_A)}\n")
    return V


def bellman_optimality_demo():
    """Show optimality (max) vs expectation (average) on one state with two actions.

    State s has two actions:
      - action 'safe' : reward 0 now, leads to a state worth 5.
      - action 'risky': reward +2 now, leads to a state worth 1.
    γ = 1 for simplicity.
    """
    gamma = 1.0
    V_next = {"safe": 5.0, "risky": 1.0}
    R = {"safe": 0.0, "risky": 2.0}

    Q = {a: R[a] + gamma * V_next[a] for a in R}        # Q(s,a) per action
    print("Bellman optimality vs expectation at one state:")
    print(f"  Q(s, safe)  = 0 + 1·5 = {Q['safe']:.1f}")
    print(f"  Q(s, risky) = 2 + 1·1 = {Q['risky']:.1f}")

    # Optimality: take the max.
    v_star = max(Q.values())
    best = max(Q, key=Q.get)
    print(f"  V*(s) = max_a Q(s,a) = {v_star:.1f}  -> optimal action: '{best}'")

    # Expectation under a uniform random policy: average.
    v_uniform = np.mean(list(Q.values()))
    print(f"  V_random(s) = average Q = {v_uniform:.1f}  (a 50/50 policy is worse)\n")
    print("  Lesson: 'max' picks the best action; 'average' just evaluates a given policy.\n")


if __name__ == "__main__":
    print("=== Value functions & Bellman equations ===\n")
    bellman_expectation_demo()
    bellman_optimality_demo()
    print("Next (file 03): use the Bellman OPTIMALITY equation as an iterative update")
    print("to actually SOLVE the GridWorld MDP — that's Dynamic Programming.")
