# Causal Language Model Objective (Next Token Prediction)
"""
WHAT/WHY:
Train a model to predict a token t+1 from tokens ≤ t.
This makes generation straightforward and avoids peeking into the future.

MATH:
For a token sequence x₁:T, maximize the log-likelihood objective:

    sum_{t=1}^{T-1} log p(x_{t+1} | x_{≤t})

Loss is the average cross-entropy between logits at position t and label x_{t+1}.

SHAPES:
Inputs (token IDs): [B, T] → logits [B, T, V], where,
- B is batch_size i.e. number of sequences in one training batch
- T sequence length i.e. number tokens per sequence

Example: if B=3 and T=4, inputs might look like:
[[12, 47,  8, 99],
 [77, 22, 11,  4],   # padded
 [ 9,  3, 10, 14]]

Each row is sequence of token IDs.

Outputs (logits): [B, T, V]
-> a 3D tensor of unnormalized scores (“logits”) for each token position.

- V is the vocabulary size.

For each of the B x T tokens, the model predicts a probability distribution over all V possible next tokens.
Example: if B=4, T=6, V=50,000, then logits has shape (4, 6, 50,000).

LABELS:
same shape as Inputs, but shifted left by 1; ignore index (e.g., -100) for padded positions.
-> because for language modeling we predict the next token.

Example:
Input sequence: [12, 47, 8, 99, 54, 2]
Label sequence: [47, 8, 99, 54, 2, -100]

Notice:
- Every position t is trained to predict the next token t+1.
- The last position has no "next token", so we set it to ignore index (e.g. -100 in PyTorch) so the loss isn't computed there.
- Similarly, padding tokens (like 0) are masked with -100 in the labels.

WHY THIS SETUP?
Because the training objective is:
maximize  sum_{t=1}^{T-1} log p(x_{t+1} | x_{≤t})
- The input gives the context up to time t.
- The label is the ground-truth next token.
- Cross-entropy loss is applied between logits[:, t, :] and labels[:, t], except where label = -100.

PYTORCH NOTES:
- Use F.cross_entropy(logits.transpose(1,2), labels, ignore_index=-100).
- Teacher forcing: model consumes ground-truth tokens, not its own predictions, during training.

GOTCHAS: Off-by-one label shift; forgetting to ignore padding.
"""

# Analogy with Time Series Forecasting (Lagged Features)
"""
The causal setup of LLM training is analogous to autoregressive time series forecasting.

In time series forecasting:
- We create lagged features for each horizon.
  For example, shift by 1 step for horizon=1, by 2 steps for horizon=2, etc.
- If we use the past n steps (e.g., 12 months), we form a feature vector of length n.
- The target is the series shifted by 1 step, while the last element is empty (no label), 
  similar to the -100 ignore index in LLM training.
- This setup naturally models horizon-1 prediction.

Parallels with LLMs:
- Prepare batches of token sequences and convert them into token IDs.
- Shift (lag) them by 1 position: inputs are tokens up to t, labels are token at t+1.
- Since token IDs are categorical, we use multi-class classification with softmax 
  to generate probabilities over the vocabulary.
- At inference, sample a token from the predicted distribution, 
  append it back to the input, and predict the next one — 
  exactly like recursive forecasting in time series.

NOTE:
This analogy captures the mechanics of causal language modeling.
However, unlike simple regression/classification models, LLMs use more complex architectures 
(e.g., self-attention) to capture long-range dependencies and context.
"""
