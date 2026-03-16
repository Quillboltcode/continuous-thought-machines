"""
CTM Innovation Proposals
========================
Five concrete architectural innovations for the Continuous Thought Machine (CTM),
each with drop-in PyTorch implementations and rationale grounded in the paper.

Innovations:
  1. Gated Synchronization Highway (GSH)
  2. Hierarchical NLM Ensembles
  3. Predictive Synchrony Loss (PSL)
  4. Sparse Adaptive Neuron Pairing (SANP)
  5. Cross-Tick Contrastive Synchronization (CTCS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# =============================================================================
# INNOVATION 1: Gated Synchronization Highway (GSH)
# =============================================================================
#
# PROBLEM: In the current CTM, the synchronisation signal flows identically
# into both the attention query path (action) and the output prediction path
# (out). There is no mechanism for the model to decide *how much* of the
# synchronisation history should influence each downstream pathway at each tick.
# This means past synchronisation states can drown out recent dynamics.
#
# PROPOSAL: Add a learned gating mechanism that modulates the synchronisation
# representation before it enters the attention and output heads. The gate is
# conditioned on the *current* activated state, allowing the model to
# dynamically weight the temporal structure of synchronisation.
#
# MATHEMATICAL INTUITION:
#   s_gated = gate(z^t) ⊙ s^t_synch
#   gate(z^t) = σ(W_gate · z^t + b_gate)
#
# WHERE IT PLUGS IN (ctm.py forward loop):
#   Replace direct use of synchronisation_action / synchronisation_out with
#   their gated versions before q_proj and output_projector.
#
# BENEFIT: Gives the model a fast-path to suppress stale synchronisation
# signals. Analogous to the forget gate in LSTMs, but operating on the
# synchronisation representation rather than the hidden state.


class GatedSynchronizationHighway(nn.Module):
    """
    Gates the synchronisation representation using the current activated state.

    Args:
        d_model: dimensionality of the activated_state (z^t)
        synch_size: dimensionality of the synchronisation representation
    """

    def __init__(self, d_model: int, synch_size: int):
        super().__init__()
        # Projects activated state down to synch space as a sigmoid gate
        self.gate_proj = nn.Sequential(nn.Linear(d_model, synch_size), nn.Sigmoid())
        # Optional residual to prevent gate from completely zeroing synch
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self, synchronisation: torch.Tensor, activated_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            synchronisation: (B, synch_size) — raw synchronisation representation
            activated_state: (B, d_model)    — current post-activations z^t

        Returns:
            gated_synch: (B, synch_size)
        """
        gate = self.gate_proj(activated_state)
        # Gated signal + small residual to preserve gradient flow
        gated = gate * synchronisation + self.residual_scale * synchronisation
        return gated


# How to integrate into CTM.__init__:
#
#   self.gsh_action = GatedSynchronizationHighway(d_model, synch_representation_size_action)
#   self.gsh_out    = GatedSynchronizationHighway(d_model, synch_representation_size_out)
#
# How to integrate into CTM.forward (inside the loop):
#
#   synchronisation_action, ... = self.compute_synchronisation(...)
#   synchronisation_action = self.gsh_action(synchronisation_action, activated_state)
#   q = self.q_proj(synchronisation_action).unsqueeze(1)
#   ...
#   synchronisation_out, ... = self.compute_synchronisation(...)
#   synchronisation_out = self.gsh_out(synchronisation_out, activated_state)
#   current_prediction = self.output_projector(synchronisation_out)


# =============================================================================
# INNOVATION 2: Hierarchical NLM Ensembles (HNE)
# =============================================================================
#
# PROBLEM: Every neuron currently uses an NLM of identical architecture (depth-1
# or depth-2 MLP with the *same* hidden width). This means each neuron integrates
# its pre-activation history on the *same* temporal scale. In the brain, neurons
# exhibit diverse integration time constants — fast neurons respond to recent
# spikes; slow neurons integrate over longer windows.
#
# PROPOSAL: Partition the D neurons into K groups. Each group uses a different
# NLM architecture varying in: (a) history window length M_k, and (b) hidden
# width d_k. Faster neurons get small M and small d; slower neurons get large M
# and large d. This creates a spectral decomposition of integration timescales
# across the latent space.
#
# MATHEMATICAL STRUCTURE:
#   For group k with neurons G_k ⊂ {1,...,D}:
#       z^{t+1}_d = g_{θ_k}(A^t_{d, -M_k:})   ∀ d ∈ G_k
#
# BENEFIT: The synchronisation matrix S^t then naturally captures cross-frequency
# coupling (fast-to-slow, slow-to-fast pairs), which maps beautifully onto
# neuroscientific cross-frequency coupling (theta-gamma, etc.) and may enable
# richer temporal representations without a parameter blowup.


class HierarchicalNLMEnsemble(nn.Module):
    """
    Partitions D neurons into K temporal-scale groups, each with its own NLM.

    Args:
        d_model:         total number of neurons D
        max_memory:      longest memory window (for pre-allocation)
        group_configs:   list of dicts, each with keys:
                           'n_neurons': int  — how many neurons in this group
                           'memory':    int  — M_k for this group
                           'hidden':    int  — hidden width of this group's NLM
    """

    def __init__(self, d_model: int, max_memory: int, group_configs: list):
        super().__init__()
        assert sum(g["n_neurons"] for g in group_configs) == d_model, (
            "Group neuron counts must sum to d_model"
        )

        self.d_model = d_model
        self.max_memory = max_memory
        self.group_configs = group_configs

        # Build one NLM per group using einsum-style grouped linear layers
        self.nlms = nn.ModuleList()
        self.group_slices = []
        start = 0
        for cfg in group_configs:
            n = cfg["n_neurons"]
            m = cfg["memory"]
            h = cfg["hidden"]
            end = start + n
            self.group_slices.append((start, end, m))
            # Each NLM: SuperLinear equivalent using einsum grouped linear
            nlm = nn.Sequential(
                _GroupedLinear(in_dims=m, out_dims=2 * h, N=n),
                nn.GLU(dim=-1),
                _GroupedLinear(in_dims=h, out_dims=2, N=n),
                nn.GLU(dim=-1),
                _Squeeze(-1),
            )
            self.nlms.append(nlm)
            start = end

    def forward(self, state_trace: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_trace: (B, D, max_memory) — full pre-activation history

        Returns:
            activated_state: (B, D)
        """
        parts = []
        for nlm, (start, end, m) in zip(self.nlms, self.group_slices):
            # Use only the last m steps of history for this group
            group_trace = state_trace[:, start:end, -m:]  # (B, n, m)
            group_out = nlm(group_trace)  # (B, n)
            parts.append(group_out)
        return torch.cat(parts, dim=1)  # (B, D)


class _GroupedLinear(nn.Module):
    """Lightweight per-neuron linear (equivalent to SuperLinear for standalone use)."""

    def __init__(self, in_dims, out_dims, N):
        super().__init__()
        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims))))

    def forward(self, x):
        # x: (B, N, in_dims)
        # w1: (in_dims, out_dims, N)
        # output: (B, N, out_dims)
        out = torch.einsum('BDM,MHD->BDH', x, self.w1) + self.b1
        return out.squeeze(-1)


class _Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


# Example group_configs for D=256 neurons, 3 temporal scales:
EXAMPLE_HNE_CONFIG = [
    {"n_neurons": 85, "memory": 5, "hidden": 8},  # fast neurons
    {
        "n_neurons": 85,
        "memory": 25,
        "hidden": 16,
    },  # medium neurons (matches paper default)
    {"n_neurons": 86, "memory": 64, "hidden": 32},  # slow neurons
]


# =============================================================================
# INNOVATION 3: Predictive Synchrony Loss (PSL)
# =============================================================================
#
# PROBLEM: The CTM is trained with a task loss at each tick + a certainty-
# weighted selection. This optimises task performance but provides no
# *intrinsic* signal to shape the quality of the synchronisation dynamics
# themselves. The representations could collapse or be highly correlated
# across neurons without the task loss penalising it.
#
# PROPOSAL: Add an auxiliary self-supervised loss that asks the model to predict
# the synchronisation at tick t+1 from the synchronisation at tick t. This is a
# predictive coding objective operating in synchronisation space. Combined with a
# decorrelation term (à la Barlow Twins) to prevent collapse.
#
# MATHEMATICAL FORM:
#   L_PSL = L_pred + λ * L_decorr
#
#   L_pred  = ||s^{t+1}_out - MLP(s^t_out)||^2  averaged over t
#   L_decorr = ||off-diagonal(C)||_F^2
#               where C_ij = corr(s_i, s_j) across batch
#
# BENEFIT: (a) Encourages the model to build temporally consistent
# representations, promoting richer internal dynamics. (b) The decorrelation
# term prevents mode collapse in synchronisation space. (c) The predictive
# signal provides a dense gradient at every tick rather than only at the
# most-certain tick.


class PredictiveSynchronyLoss(nn.Module):
    """
    Auxiliary loss that promotes temporally predictive, decorrelated synchronisation.

    Args:
        synch_size:    dimensionality of synchronisation_out
        hidden:        hidden dim of the prediction MLP
        lambda_decorr: weight for the decorrelation term
    """

    def __init__(self, synch_size: int, hidden: int = 64, lambda_decorr: float = 0.005):
        super().__init__()
        self.lambda_decorr = lambda_decorr
        # Small MLP to predict next tick's synchronisation from current
        self.predictor = nn.Sequential(
            nn.Linear(synch_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, synch_size),
        )

    def forward(self, synch_out_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            synch_out_sequence: (T, B, synch_size) — synchronisation_out at every tick,
                                collected during the CTM's forward pass.

        Returns:
            loss: scalar tensor
        """
        T = synch_out_sequence.size(0)
        if T < 2:
            return synch_out_sequence.new_tensor(0.0)

        # --- Predictive Loss ---
        current = synch_out_sequence[:-1]  # (T-1, B, synch_size)
        target = synch_out_sequence[1:]  # (T-1, B, synch_size)
        predicted = self.predictor(current)
        l_pred = F.mse_loss(predicted, target.detach())

        # --- Decorrelation Loss (Barlow Twins style) ---
        # Flatten time and batch
        flat = synch_out_sequence.view(
            -1, synch_out_sequence.size(-1)
        )  # (T*B, synch_size)
        flat_norm = (flat - flat.mean(0)) / (flat.std(0) + 1e-6)
        N = flat_norm.size(0)
        C = (flat_norm.T @ flat_norm) / N  # (synch_size, synch_size)
        # Off-diagonal penalty
        eye = torch.eye(C.size(0), device=C.device)
        l_decorr = ((C - eye) ** 2).sum() / C.numel()

        return l_pred + self.lambda_decorr * l_decorr


# How to integrate into training loop:
#
#   psl = PredictiveSynchronyLoss(synch_size=model.synch_representation_size_out).to(device)
#
#   # Inside the CTM forward, collect synch_out at every tick:
#   synch_out_list = []
#   for stepi in range(iterations):
#       ...
#       synchronisation_out, ... = model.compute_synchronisation(...)
#       synch_out_list.append(synchronisation_out)
#       ...
#   synch_out_seq = torch.stack(synch_out_list)  # (T, B, synch_size)
#
#   # In the training loss:
#   task_loss, _ = image_classification_loss(predictions, certainties, targets)
#   aux_loss = psl(synch_out_seq)
#   total_loss = task_loss + 0.1 * aux_loss


# =============================================================================
# INNOVATION 4: Sparse Adaptive Neuron Pairing (SANP)
# =============================================================================
#
# PROBLEM: Neuron pairs for synchronisation are chosen randomly at
# initialisation and frozen for the entire training run. This means the
# synchronisation representation is fixed in structure — if the randomly
# chosen pairs are poorly informative for a given task, the model cannot
# compensate. Furthermore, neuron pairs that contribute minimally to the
# gradient waste representational capacity.
#
# PROPOSAL: Make neuron pairing *adaptive* during training via a learned soft
# assignment. Each output slot in the synchronisation vector maintains a
# learned distribution over pairs. During training, the top-K pairs by
# probability mass are selected (straight-through estimator for gradients).
# Periodically, low-importance pairs are pruned and replaced by sampling new
# candidates from the learned distribution.
#
# MATHEMATICAL FORM:
#   For each slot k, maintain logits w_k ∈ R^{D^2}.
#   During forward: pair_k = argmax(softmax(w_k))  [hard, straight-through]
#   s^t_k = z^t_{pair_k[0]} · z^t_{pair_k[1]}
#
# BENEFIT: The model can discover which neuron pairs carry the most task-
# relevant synchronisation signal, rather than relying on random luck at init.


class SparseAdaptiveNeuronPairing(nn.Module):
    """
    Learns which neuron pairs to track for synchronisation.

    At train time: soft selection with straight-through gradients.
    At eval time: hard top-1 selection.

    Args:
        d_model:     total neurons D
        n_pairs:     number of synchronisation pairs (D_out or D_action)
        temperature: softmax temperature for pair selection
        init_top_k:  fraction of D^2 pairs to keep in logits (for memory efficiency)
    """

    def __init__(
        self,
        d_model: int,
        n_pairs: int,
        temperature: float = 1.0,
        init_top_k: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_pairs = n_pairs
        self.temperature = temperature
        self.init_top_k = min(init_top_k, d_model * d_model)

        # We maintain a sparse candidate set of pairs for efficiency
        # Sample init_top_k candidate pairs at startup
        all_i = torch.randint(0, d_model, (self.init_top_k,))
        all_j = torch.randint(0, d_model, (self.init_top_k,))
        self.register_buffer("candidate_i", all_i)
        self.register_buffer("candidate_j", all_j)

        # Learnable logits over candidate pairs, shape (n_pairs, init_top_k)
        self.pair_logits = nn.Parameter(torch.zeros(n_pairs, self.init_top_k))

    def forward(self, activated_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activated_state: (B, D) — current post-activations

        Returns:
            synch: (B, n_pairs) — synchronisation using adaptive pairs
        """
        # Soft pair weights: (n_pairs, init_top_k)
        weights = F.softmax(self.pair_logits / self.temperature, dim=-1)

        # Gather all candidate pair products: (B, init_top_k)
        left = activated_state[:, self.candidate_i]  # (B, init_top_k)
        right = activated_state[:, self.candidate_j]  # (B, init_top_k)
        pair_products = left * right  # (B, init_top_k)

        if self.training:
            # Soft weighted sum for gradient flow: (B, n_pairs)
            synch = torch.einsum("bk,nk->bn", pair_products, weights)
        else:
            # Hard top-1 selection at inference
            best_pairs = self.pair_logits.argmax(dim=-1)  # (n_pairs,)
            synch = pair_products[:, best_pairs]  # (B, n_pairs)

        return synch

    @torch.no_grad()
    def refresh_candidates(self, keep_top: int = 100):
        """
        Prune low-weight candidates and sample fresh ones.
        Call periodically during training (e.g., every 10k steps).
        """
        # Find top-K pairs by max logit across output slots
        importance = self.pair_logits.max(0).values  # (init_top_k,)
        top_idx = importance.topk(keep_top).indices
        new_count = self.init_top_k - keep_top

        # Sample new random pairs for the pruned slots
        new_i = torch.randint(
            0, self.d_model, (new_count,), device=self.candidate_i.device
        )
        new_j = torch.randint(
            0, self.d_model, (new_count,), device=self.candidate_j.device
        )

        self.candidate_i = torch.cat([self.candidate_i[top_idx], new_i])
        self.candidate_j = torch.cat([self.candidate_j[top_idx], new_j])

        # Reset logits for new candidates
        new_logits = torch.zeros(
            self.n_pairs, self.init_top_k, device=self.pair_logits.device
        )
        new_logits[:, :keep_top] = self.pair_logits[:, top_idx].detach()
        self.pair_logits = nn.Parameter(new_logits)


# =============================================================================
# INNOVATION 5: Cross-Tick Contrastive Synchronization (CTCS)
# =============================================================================
#
# PROBLEM: The CTM currently has no explicit mechanism to ensure that
# synchronisation representations are *distinct across inputs*. Two very different
# inputs could drive similar synchronisation trajectories, causing representational
# collapse in the synchronisation space. The task loss alone may not be
# sufficient to prevent this for tasks without highly discriminative targets.
#
# PROPOSAL: Apply a contrastive loss across the synchronisation trajectory.
# For a batch of B inputs, treat the synchronisation at each tick t as an anchor.
# Positive pairs: same input at adjacent ticks t and t+1 (temporal consistency).
# Negative pairs: different inputs at the same tick t (instance discrimination).
# This is a temporal variant of SimCLR operating on the synchronisation time series.
#
# MATHEMATICAL FORM (InfoNCE):
#   L_CTCS = -mean_t [ log( exp(sim(s^t_i, s^{t+1}_i)/τ) /
#                            Σ_j exp(sim(s^t_i, s^t_j)/τ) ) ]
#
# BENEFIT: Forces the synchronisation space to be (a) temporally smooth for
# each input (promoting stable thought trajectories) and (b) discriminative
# across inputs (preventing representational collapse). This directly addresses
# the CTM's known limitation of needing more training for effective dynamics.


class CrossTickContrastiveLoss(nn.Module):
    """
    Contrastive loss over the synchronisation trajectory.

    Args:
        temperature:  InfoNCE temperature τ
        n_ticks_sample: number of adjacent tick pairs to sample per forward pass
                        (sampling avoids O(T^2) cost for large T)
    """

    def __init__(self, temperature: float = 0.07, n_ticks_sample: int = 8):
        super().__init__()
        self.temperature = temperature
        self.n_ticks_sample = n_ticks_sample

    def forward(self, synch_out_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            synch_out_sequence: (T, B, synch_size)

        Returns:
            loss: scalar
        """
        T, B, D = synch_out_sequence.shape
        if T < 2 or B < 2:
            return synch_out_sequence.new_tensor(0.0)

        # Sample random adjacent tick pairs
        tick_starts = torch.randperm(T - 1, device=synch_out_sequence.device)[
            : self.n_ticks_sample
        ]

        losses = []
        for t in tick_starts:
            anchor = synch_out_sequence[t]  # (B, D)  — tick t
            positive = synch_out_sequence[t + 1]  # (B, D)  — tick t+1 (same input)

            # L2 normalize
            anchor = F.normalize(anchor, dim=-1)
            positive = F.normalize(positive, dim=-1)

            # Similarity matrix: anchor_i vs all positives_j at tick t+1
            # Positive pair for sample i is the i-th row of positive
            sim_matrix = torch.mm(anchor, positive.T) / self.temperature  # (B, B)

            # Labels: each anchor's positive is on the diagonal
            labels = torch.arange(B, device=anchor.device)

            tick_loss = F.cross_entropy(sim_matrix, labels)
            losses.append(tick_loss)

        return torch.stack(losses).mean()


# =============================================================================
# INTEGRATION EXAMPLE: Extended CTM Forward Pass
# =============================================================================
# Below is a sketch showing how all five innovations plug into the CTM forward,
# using drop-in replacements with minimal changes to the original structure.


def extended_ctm_forward_sketch(
    model,
    x,
    track=False,
    gsh_action=None,
    gsh_out=None,
    psl=None,
    ctcs_loss=None,
    use_hierarchical_nlm=False,
):
    """
    Demonstration of how to integrate innovations 1, 3, 5 into the CTM forward pass.
    (Innovations 2 and 4 require changes to __init__ / model construction.)

    Args:
        model:           ContinuousThoughtMachine instance (unmodified)
        gsh_action:      GatedSynchronizationHighway for action path  [Innovation 1]
        gsh_out:         GatedSynchronizationHighway for output path  [Innovation 1]
        psl:             PredictiveSynchronyLoss module               [Innovation 3]
        ctcs_loss:       CrossTickContrastiveLoss module               [Innovation 5]

    Returns:
        predictions, certainties, synchronisation_out, aux_losses (dict)
    """
    B = x.size(0)
    device = x.device

    kv = model.compute_features(x)
    state_trace = model.start_trace.unsqueeze(0).expand(B, -1, -1)
    activated_state = model.start_activated_state.unsqueeze(0).expand(B, -1)

    predictions = torch.empty(B, model.out_dims, model.iterations, device=device)
    certainties = torch.empty(B, 2, model.iterations, device=device)

    model.decay_params_action.data = torch.clamp(model.decay_params_action, 0, 15)
    model.decay_params_out.data = torch.clamp(model.decay_params_out, 0, 15)
    r_action = torch.exp(-model.decay_params_action).unsqueeze(0).repeat(B, 1)
    r_out = torch.exp(-model.decay_params_out).unsqueeze(0).repeat(B, 1)

    decay_alpha_action = decay_beta_action = None
    _, decay_alpha_out, decay_beta_out = model.compute_synchronisation(
        activated_state, None, None, r_out, synch_type="out"
    )

    synch_out_history = []  # Collected for PSL [Innovation 3] and CTCS [Innovation 5]

    for stepi in range(model.iterations):
        # --- Action Synchronisation + Gate [Innovation 1] ---
        synchronisation_action, decay_alpha_action, decay_beta_action = (
            model.compute_synchronisation(
                activated_state,
                decay_alpha_action,
                decay_beta_action,
                r_action,
                synch_type="action",
            )
        )
        if gsh_action is not None:
            synchronisation_action = gsh_action(synchronisation_action, activated_state)

        # --- Attention ---
        q = model.q_proj(synchronisation_action).unsqueeze(1)
        attn_out, _ = model.attention(
            q, kv, kv, average_attn_weights=False, need_weights=True
        )
        attn_out = attn_out.squeeze(1)
        pre_synapse_input = torch.cat((attn_out, activated_state), dim=-1)

        # --- Synapses ---
        state = model.synapses(pre_synapse_input)
        state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

        # --- NLMs (Innovation 2 replaces model.trace_processor here) ---
        activated_state = model.trace_processor(state_trace)

        # --- Output Synchronisation + Gate [Innovation 1] ---
        synchronisation_out, decay_alpha_out, decay_beta_out = (
            model.compute_synchronisation(
                activated_state,
                decay_alpha_out,
                decay_beta_out,
                r_out,
                synch_type="out",
            )
        )
        if gsh_out is not None:
            synchronisation_out = gsh_out(synchronisation_out, activated_state)

        synch_out_history.append(synchronisation_out)

        # --- Predictions ---
        current_prediction = model.output_projector(synchronisation_out)
        current_certainty = model.compute_certainty(current_prediction)
        predictions[..., stepi] = current_prediction
        certainties[..., stepi] = current_certainty

    # --- Auxiliary Losses ---
    aux_losses = {}
    synch_out_seq = torch.stack(synch_out_history)  # (T, B, synch_size)

    if psl is not None:
        aux_losses["psl"] = psl(synch_out_seq)  # Innovation 3

    if ctcs_loss is not None:
        aux_losses["ctcs"] = ctcs_loss(synch_out_seq)  # Innovation 5

    return predictions, certainties, synchronisation_out, aux_losses


# =============================================================================
# QUICK UNIT TESTS
# =============================================================================


def _test_all_innovations():
    torch.manual_seed(42)
    B, D, T, synch_size, memory = 4, 128, 10, 64, 25

    print("Testing Innovation 1: Gated Synchronization Highway...")
    gsh = GatedSynchronizationHighway(d_model=D, synch_size=synch_size)
    synch = torch.randn(B, synch_size)
    z = torch.randn(B, D)
    out = gsh(synch, z)
    assert out.shape == (B, synch_size), f"GSH shape mismatch: {out.shape}"
    print(f"  ✓ GSH output shape: {tuple(out.shape)}")

    print("Testing Innovation 2: Hierarchical NLM Ensembles...")
    hne_memory = 50  # Must be >= max memory across groups
    hne = HierarchicalNLMEnsemble(
        d_model=D,
        max_memory=hne_memory,
        group_configs=[
            {"n_neurons": 42, "memory": 5, "hidden": 8},
            {"n_neurons": 43, "memory": 25, "hidden": 16},
            {"n_neurons": 43, "memory": 50, "hidden": 32},
        ],
    )
    trace = torch.randn(B, D, hne_memory)
    activated = hne(trace)
    assert activated.shape == (B, D), f"HNE shape mismatch: {activated.shape}"
    print(f"  ✓ HNE output shape: {tuple(activated.shape)}")

    print("Testing Innovation 3: Predictive Synchrony Loss...")
    psl = PredictiveSynchronyLoss(synch_size=synch_size, hidden=32)
    synch_seq = torch.randn(T, B, synch_size, requires_grad=True)
    loss = psl(synch_seq)
    loss.backward()
    assert loss.item() > 0
    print(f"  ✓ PSL loss: {loss.item():.4f}")

    print("Testing Innovation 4: Sparse Adaptive Neuron Pairing...")
    sanp = SparseAdaptiveNeuronPairing(d_model=D, n_pairs=synch_size, init_top_k=256)
    z = torch.randn(B, D)
    sanp.train()
    synch_out = sanp(z)
    assert synch_out.shape == (B, synch_size), f"SANP shape mismatch: {synch_out.shape}"
    sanp.eval()
    synch_out_eval = sanp(z)
    assert synch_out_eval.shape == (B, synch_size)
    print(f"  ✓ SANP output shape (train): {tuple(synch_out.shape)}")

    print("Testing Innovation 5: Cross-Tick Contrastive Synchronization...")
    ctcs = CrossTickContrastiveLoss(temperature=0.07, n_ticks_sample=4)
    synch_seq = torch.randn(T, B, synch_size, requires_grad=True)
    loss = ctcs(synch_seq)
    loss.backward()
    assert loss.item() > 0
    print(f"  ✓ CTCS loss: {loss.item():.4f}")

    print("\nAll innovations passed! ✓")


if __name__ == "__main__":
    _test_all_innovations()
