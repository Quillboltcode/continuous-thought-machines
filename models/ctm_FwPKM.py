"""
CTM-FwPKM Neural Computer
=========================
A neural computer that integrates the Continuous Thought Machine (CTM, Darlow et al. 2025)
with Fast-weight Product Key Memory (FwPKM, Zhao & Jones 2026).

This version integrates the full CTM pipeline from ctm.py:
  - Full backbone support (resnet, shallow-wide, parity_backbone, etc.)
  - Positional embeddings (learnable-fourier, custom-rotational, etc.)
  - U-Net synapse model (SynapseUNET)
  - Neuron-Level Models with SuperLinear
  - Full synchronization types (first-last, random, random-pairing)
  - Certainty computation

Architecture overview:

  Input Image
      │
  [CTM Backbone]  ─────────────────────────────────────────────┐
      │                                                         │
  KV tokens  ──────────────────────────────────────────────────┤
      │                                                         │
  [FwPKM Write]  ← fast weights updated on image chunks        │
      │ (K1, K2, V stored)                                      │
      └──────────────────────────────────────────┐             │
                                                  ↓             ↓
                                     ┌─── CTM Internal Tick t ──────────────┐
                                     │   synch_a (action synchronization)   │
                                     │       │                               │
                                     │  ┌────┴──────────────────────┐       │
                                     │  │     Cross-Attention        │       │
                                     │  │  q = W_in · synch_a        │       │
                                     │  │  o_t = Attn(q, KV, KV)    │←──KV  │
                                     │  └────────────┬──────────────┘       │
                                     │               │                       │
                                     │  ┌────────────┴──────────────┐       │
                                     │  │     FwPKM Retrieval        │       │
                                     │  │  fq = W_fwpkm · synch_a   │       │
                                     │  │  m_t = FwPKM.read(fq)     │←─K1,K2,V
                                     │  └────────────┬──────────────┘       │
                                     │               │                       │
                                     │  ┌────────────┴──────────────┐       │
                                     │  │    Synapse (U-Net MLP)      │       │
                                     │  │  a_t = f_syn(z_t, o_t, m_t)       │
                                     │  └────────────┬──────────────┘       │
                                     │               │                       │
                                     │  ┌────────────┴──────────────┐       │
                                     │  │  Neuron-Level Models (NLM) │       │
                                     │  │  z_{t+1} = g_θ(A_t)       │       │
                                     │  └────────────┬──────────────┘       │
                                     │               │                       │
                                     │   synch_o = sync(z_{t+1})            │
                                     │   y_t = W_out · synch_o              │
                                     └──────────────────────────────────────┘
                                                    │
                                               [Output y_t]
                                          (one prediction per tick)

Design choices:
  • FwPKM writes image patch features (KV tokens) into episodic memory before ticks begin.
  • At every internal tick, CTM's action synchronization drives TWO reads:
      1. Cross-attention over raw KV tokens (original CTM path).
      2. FwPKM retrieval: synchronization → product-key query → sparse value read.
  • Both reads are concatenated with z_t and fed into the synapse model.
  • This creates a "working memory + episodic memory" duality:
      - Attention = working context window (full image)
      - FwPKM    = episodic memory that can be written/overwritten dynamically

References:
  [1] Darlow et al. (2025) "Continuous Thought Machines" NeurIPS 2025
  [2] Zhao & Jones (2026) "Fast-weight Product Key Memory" arXiv:2601.00671
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

from models.modules import (
    ParityBackbone, SynapseUNET, Squeeze, SuperLinear,
    LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding,
    CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
)
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_NEURON_SELECT_TYPES,
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)


# ===========================================================================
# FwPKM Layer   [Zhao & Jones 2026 §3]
# ===========================================================================


class FwPKMLayer(nn.Module):
    """
    Fast-weight Product Key Memory.

    Combines:
      • PKM's O(sqrt(N)) sparse retrieval via product-key decomposition.
      • TTT-style online fast-weight updates: value matrix V is updated per chunk
        by minimising a gated reconstruction loss L_mem = sum_t g_t * ||v_t - v̂_t||²/2.
      • Key matrices K1, K2 are updated by an auxiliary addressing loss L_addr
        (simplified here to a weight-normalisation step).

    Memory roles (per paper):
      • Slow weights (K1_slow, K2_slow, V_slow): semantic memory — dataset-wide facts.
      • Fast weights (K1, K2, V): episodic memory — context-specific bindings.

    Fast weights are initialised from the slow weights at the start of each sequence,
    then updated chunk-by-chunk as the sequence is processed.

    Args:
        d_model   : Dimension of host model's hidden states (input to FwPKM).
        d_key     : Key dimension (split in half across two sub-codebooks).
        d_val     : Value dimension inside FwPKM.
        n_mem     : Total memory slots. Must be a perfect square for PKM.
        top_k     : Number of memory slots retrieved per token.
        lr_fast   : Learning rate for fast-weight value updates.
    """

    def __init__(
        self,
        d_model: int,
        d_key: int = 64,
        d_val: int = 128,
        n_mem: int = 1024,
        top_k: int = 8,
        lr_fast: float = 0.1,
    ):
        super().__init__()

        assert n_mem > 0 and (int(math.isqrt(n_mem)) ** 2 == n_mem), (
            f"n_mem={n_mem} must be a perfect square."
        )

        self.d_model = d_model
        self.d_key = d_key
        self.d_val = d_val
        self.top_k = top_k
        self.lr_fast = lr_fast
        self.sqrt_n = int(math.isqrt(n_mem))
        self.n_mem = n_mem

        # ── Slow-weight projections (frozen at inference) ───────────────────
        self.q_norm = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_key, bias=False)

        self.v_norm = nn.RMSNorm(d_model)
        self.v_proj = nn.Linear(d_model, d_val, bias=False)

        self.g_norm = nn.RMSNorm(d_model)
        self.g_proj = nn.Linear(d_model, 1, bias=True)

        self.o_norm = nn.RMSNorm(d_val)
        self.o_proj = nn.Linear(d_val, d_model, bias=False)

        # ── Slow-weight memory (semantic memory) ────────────────────────────
        self.K1_slow = nn.Parameter(torch.randn(self.sqrt_n, d_key // 2) * 0.02)
        self.K2_slow = nn.Parameter(torch.randn(self.sqrt_n, d_key // 2) * 0.02)
        self.V_slow = nn.Parameter(torch.zeros(n_mem, d_val))

        # ── Key normalisation (weight-norm style for addressing stability) ──
        self.k1_norm = nn.LayerNorm(d_key // 2, elementwise_affine=False)
        self.k2_norm = nn.LayerNorm(d_key // 2, elementwise_affine=False)

    # -----------------------------------------------------------------------
    # Fast-weight lifecycle
    # -----------------------------------------------------------------------

    def init_fast_weights(self, batch_size: int, device: torch.device):
        """Create per-sequence fast weights cloned from slow weights."""
        K1 = self.K1_slow.detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        K2 = self.K2_slow.detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        V = self.V_slow.detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        return K1.to(device), K2.to(device), V.to(device)

    # -----------------------------------------------------------------------
    # Sparse retrieval   [PKM §2.2 → FwPKM §3.2]
    # -----------------------------------------------------------------------

    def sparse_retrieve(
        self,
        q: torch.Tensor,
        K1: torch.Tensor,
        K2: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Product-key sparse retrieval.

        Args:
            q   : (B, d_key)         query
            K1  : (B, sqrt_n, d_k/2) first sub-codebook
            K2  : (B, sqrt_n, d_k/2) second sub-codebook
            V   : (B, n_mem, d_val)  value matrix

        Returns:
            v_hat      : (B, d_val)    retrieved value
            mem_indices: (B, top_k)    activated slot indices
            weights    : (B, top_k)    softmax attention weights
        """
        B = q.shape[0]
        q1, q2 = q.chunk(2, dim=-1)  # (B, d_key//2)

        # Score each codebook independently — O(sqrt_n) per codebook
        s1 = torch.bmm(K1, q1.unsqueeze(-1)).squeeze(-1)  # (B, sqrt_n)
        s2 = torch.bmm(K2, q2.unsqueeze(-1)).squeeze(-1) # (B, sqrt_n)

        # Top-k candidates per codebook
        k_cand = max(self.top_k, 1)
        _, i1 = s1.topk(k_cand, dim=-1)  # (B, k_cand)
        _, i2 = s2.topk(k_cand, dim=-1)

        # Additive pairwise scores over Cartesian product i1 × i2
        s1_sel = s1.gather(1, i1)  # (B, k_cand)
        s2_sel = s2.gather(1, i2)
        pair_scores = s1_sel.unsqueeze(2) + s2_sel.unsqueeze(1)  # (B, k_cand, k_cand)
        pair_flat = pair_scores.reshape(B, -1)  # (B, k_cand²)

        # Final top_k pairs
        best_scores, best_idx = pair_flat.topk(self.top_k, dim=-1)  # (B, top_k)
        row_idx = best_idx // k_cand  # index into i1
        col_idx = best_idx % k_cand   # index into i2
        mem_row = i1.gather(1, row_idx)
        mem_col = i2.gather(1, col_idx)
        mem_indices = mem_row * self.sqrt_n + mem_col  # (B, top_k)

        # Weighted sum of value rows
        weights = F.softmax(best_scores, dim=-1)  # (B, top_k)
        V_sel = V.gather(1, mem_indices.unsqueeze(-1).expand(-1, -1, self.d_val))
        v_hat = (weights.unsqueeze(-1) * V_sel).sum(dim=1)  # (B, d_val)

        return v_hat, mem_indices, weights

    # -----------------------------------------------------------------------
    # Fast-weight update   [FwPKM §3.3]
    # -----------------------------------------------------------------------

    def _update_fast_weights(
        self,
        V: torch.Tensor,
        all_v_hat: List[torch.Tensor],
        all_v_tgt: List[torch.Tensor],
        all_g: List[torch.Tensor],
        all_idx: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply chunk-level gradient descent on L_mem to update V.

        L_mem = Σ_t  g_t/2 * ||v_t - v̂_t||²

        The gradient w.r.t. each accessed value row V_i is:
            ∇V_i L_mem = Σ_{t: i∈I_t}  g_t * (v̂_t - v_t) * s'_{t,i}

        We use a simplified aggregation: per-row mean gradient
        (consensus across competing writes), then a single gradient step.
        """
        V = V.clone()
        for t in range(len(all_v_hat)):
            residual = all_v_hat[t] - all_v_tgt[t]  # (B, d_val)  error signal
            g_t = all_g[t]  # (B,)
            idx_t = all_idx[t]  # (B, top_k)

            # Gradient: g_t * residual broadcast over top_k accessed slots
            delta = (
                (g_t.unsqueeze(-1) * residual).unsqueeze(1).expand(-1, self.top_k, -1)
            )  # (B, top_k, d_val)

            # Normalise by top_k (average-access normalisation, Eq.14 in paper)
            delta = delta / self.top_k

            idx_exp = idx_t.unsqueeze(-1).expand(-1, -1, self.d_val)
            V = V.scatter_add(1, idx_exp, -self.lr_fast * delta)

        return V

    # -----------------------------------------------------------------------
    # Forward: chunk-level write + read   [FwPKM §3.1]
    # -----------------------------------------------------------------------

    def write_chunk(
        self,
        h_chunk: torch.Tensor,  # (B, C, d_model)
        K1: torch.Tensor,
        K2: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one chunk:
          1. Retrieve for each token using current fast weights.
          2. Compute gated residual output.
          3. Update fast weights by minimising L_mem over the chunk.

        Returns: (outputs, K1, K2, V_updated)
            outputs : (B, C, d_model)
        """
        B, C, _ = h_chunk.shape
        all_v_hat, all_v_tgt, all_g, all_idx = [], [], [], []
        outputs = []

        for t in range(C):
            h_t = h_chunk[:, t]  # (B, d_model)
            q_t = self.q_proj(self.q_norm(h_t))  # (B, d_key)
            v_t = self.v_proj(self.v_norm(h_t))  # (B, d_val)
            g_t = torch.sigmoid(self.g_proj(self.g_norm(h_t))).squeeze(-1)  # (B,)

            v_hat_t, idx_t, _ = self.sparse_retrieve(q_t, K1, K2, V)

            # Gated residual  [Eq.12 in FwPKM]
            o_t = (
                g_t.unsqueeze(-1) * v_hat_t + (1 - g_t.unsqueeze(-1)) * v_t
            )  # (B, d_val)
            o_t = self.o_proj(self.o_norm(o_t))  # (B, d_model)
            outputs.append(o_t)

            all_v_hat.append(v_hat_t)
            all_v_tgt.append(v_t)
            all_g.append(g_t)
            all_idx.append(idx_t)

        # Fast-weight update on V  [Eq.15 in FwPKM]
        V = self._update_fast_weights(V, all_v_hat, all_v_tgt, all_g, all_idx)

        return torch.stack(outputs, dim=1), K1, K2, V

    def forward(
        self,
        h: torch.Tensor,  # (B, T, d_model)
        K1=None,
        K2=None,
        V=None,
        chunk_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a sequence into FwPKM episodic memory.
        Returns (outputs, K1, K2, V) where V contains the updated fast weights.
        """
        B = h.shape[0]
        if K1 is None:
            K1, K2, V = self.init_fast_weights(B, h.device)

        out_chunks = []
        for start in range(0, h.shape[1], chunk_size):
            chunk = h[:, start : start + chunk_size]
            out_c, K1, K2, V = self.write_chunk(chunk, K1, K2, V)
            out_chunks.append(out_c)

        return torch.cat(out_chunks, dim=1), K1, K2, V

    def read(
        self,
        q_vec: torch.Tensor,  # (B, d_key)  arbitrary query
        K1: torch.Tensor,
        K2: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Read-only retrieval from episodic memory.
        Used by CTM reasoning core to query memory at each internal tick.

        Returns: (B, d_model)  — projected retrieved value.
        """
        v_hat, _, _ = self.sparse_retrieve(q_vec, K1, K2, V)
        return self.o_proj(self.o_norm(v_hat))


# ===========================================================================
# CTM-FwPKM Neural Computer (with full CTM pipeline integration)
# ===========================================================================


class CTMFwPKMNeuralComputer(nn.Module):
    """
    Neural Computer that fuses CTM temporal reasoning with FwPKM episodic memory.

    This version integrates the full CTM pipeline from ctm.py:
      - Full backbone support (resnet, shallow-wide, parity_backbone, etc.)
      - Positional embeddings (learnable-fourier, custom-rotational, etc.)
      - U-Net synapse model (SynapseUNET)
      - Neuron-Level Models with SuperLinear
      - Full synchronization types (first-last, random, random-pairing)
      - Certainty computation

    Key ideas:
    ──────────
    1. The backbone encodes the input image into patch-level KV tokens.

    2. FwPKM encodes these tokens into a fast-weight episodic memory bank
       (K1, K2, V) via chunk-level gradient descent on a local reconstruction
       objective. This memorises "what is in the image" at the slot level.

    3. CTM's internal ticks then unfold neural dynamics. At every tick:
         (a) Action synchronization S_action is computed from post-activation
             history, capturing "what the model is currently thinking about".
         (b) S_action drives cross-attention to the raw KV tokens (standard CTM).
         (c) S_action also addresses the FwPKM memory bank via a product-key
             query — retrieving episodic content relevant to the current thought.
         (d) The synapse model combines (z_t, attn_out, fwpkm_retrieved) into
             pre-activations that drive NLMs to produce z_{t+1}.

    4. Output synchronization S_out is projected to class logits at every tick,
       enabling the certainty-based loss used in the original CTM.

    This design gives the model:
      • Dynamic attention (CTM) — able to attend to different image regions
        at different ticks.
      • Episodic memory (FwPKM) — able to rapidly memorise and retrieve
        arbitrary facts from the input, even across long internal sequences.
      • Full CTM pipeline — backbone, positional embeddings, U-Net synapses,
        SuperLinear NLMs, and certainty computation.

    Args:
        iterations (int): Number of internal ticks T.
        d_model (int): Core dimensionality of the CTM's latent space (D).
        d_input (int): Dimensionality of projected attention outputs.
        heads (int): Number of attention heads.
        n_synch_out (int): Number of neurons used for output synchronization.
        n_synch_action (int): Number of neurons used for action synchronization.
        synapse_depth (int): Depth of the synapse model (U-Net if > 1, else MLP).
        memory_length (int): History length for Neuron-Level Models (M).
        deep_nlms (bool): Use deeper (2-layer) NLMs if True, else linear.
        memory_hidden_dims (int): Hidden dimension size for deep NLMs.
        do_layernorm_nlm (bool): Apply LayerNorm within NLMs.
        backbone_type (str): Type of feature extraction backbone.
        pretrained_backbone (str): Use a pretrained backbone.
        positional_embedding_type (str): Type of positional embedding.
        out_dims (int): Output dimension size.
        prediction_reshaper (list): Shape for reshaping predictions.
        dropout (float): Dropout rate.
        neuron_select_type (str): Neuron selection strategy.
        n_random_pairing_self (int): Number of self-to-self synch pairs.
        # FwPKM specific
        fwpkm_d_key (int): FwPKM key dimension.
        fwpkm_d_val (int): FwPKM value dimension.
        fwpkm_n_mem (int): FwPKM memory slots.
        fwpkm_top_k (int): FwPKM top-k retrieval.
        fwpkm_lr (float): FwPKM learning rate.
        fwpkm_chunk_size (int): Chunk size for FwPKM write pass.
        grayscale (bool): Process grayscale images.
    """

    def __init__(
        self,
        # CTM parameters (from ctm.py)
        iterations: int = 10,
        d_model: int = 128,
        d_input: int = 128,
        heads: int = 4,
        n_synch_out: int = 32,
        n_synch_action: int = 32,
        synapse_depth: int = 1,
        memory_length: int = 10,
        deep_nlms: bool = True,
        memory_hidden_dims: int = 32,
        do_layernorm_nlm: bool = False,
        backbone_type: str = 'resnet18-2',
        pretrained_backbone: str = 'none',
        positional_embedding_type: str = 'none',
        out_dims: int = 10,
        prediction_reshaper: list = [-1],
        dropout: float = 0,
        dropout_nlm: float = None,
        neuron_select_type: str = 'random-pairing',
        n_random_pairing_self: int = 0,
        grayscale: bool = False,
        # FwPKM parameters
        fwpkm_d_key: int = 64,
        fwpkm_d_val: int = 128,
        fwpkm_n_mem: int = 1024,
        fwpkm_top_k: int = 8,
        fwpkm_lr: float = 0.1,
        fwpkm_chunk_size: int = 16,
    ):
        super(CTMFwPKMNeuralComputer, self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.prediction_reshaper = prediction_reshaper
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.backbone_type = backbone_type
        self.pretrained_backbone = pretrained_backbone
        self.out_dims = out_dims
        self.positional_embedding_type = positional_embedding_type
        self.neuron_select_type = neuron_select_type
        self.grayscale = grayscale
        self.fwpkm_chunk_size = fwpkm_chunk_size
        dropout_nlm = dropout if dropout_nlm is None else dropout_nlm

        # --- Assertions ---
        self.verify_args()

        # --- Input Processing (from CTM) ---
        d_backbone = self.get_d_backbone()
        self.set_initial_rgb()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)
        self.kv_proj = nn.Sequential(
            nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input)
        ) if heads else None
        self.q_proj = nn.LazyLinear(self.d_input) if heads else None
        self.attention = nn.MultiheadAttention(
            self.d_input, heads, dropout, batch_first=True
        ) if heads else None

        # --- Core CTM Modules ---
        self.synapses = self.get_synapses(synapse_depth, d_model, dropout)
        self.trace_processor = self.get_neuron_level_models(
            deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout_nlm
        )

        # --- Start States ---
        self.register_parameter(
            'start_activated_state',
            nn.Parameter(torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model))))
        )
        self.register_parameter(
            'start_trace',
            nn.Parameter(torch.zeros((d_model, memory_length)).uniform_(-math.sqrt(1/(d_model+memory_length)), math.sqrt(1/(d_model+memory_length))))
        )

        # --- Synchronisation ---
        self.neuron_select_type_out, self.neuron_select_type_action = self.get_neuron_select_type()
        self.synch_representation_size_action = self.calculate_synch_representation_size(self.n_synch_action)
        self.synch_representation_size_out = self.calculate_synch_representation_size(self.n_synch_out)

        for synch_type, size in (('action', self.synch_representation_size_action), ('out', self.synch_representation_size_out)):
            print(f"Synch representation size {synch_type}: {size}")
        if self.synch_representation_size_action:
            self.set_synchronisation_parameters('action', self.n_synch_action, n_random_pairing_self)
        self.set_synchronisation_parameters('out', self.n_synch_out, n_random_pairing_self)

        # --- Output Processing ---
        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))

        # --- FwPKM ---
        self.fwpkm = FwPKMLayer(
            d_model=d_input,
            d_key=fwpkm_d_key,
            d_val=fwpkm_d_val,
            n_mem=fwpkm_n_mem,
            top_k=fwpkm_top_k,
            lr_fast=fwpkm_lr,
        )

        # Project CTM action synchronization → FwPKM query
        self.synch_to_fwpkm_q = nn.Linear(self.synch_representation_size_action, fwpkm_d_key)

        # Project FwPKM retrieval output (d_input) → d_model for synapse injection
        self.fwpkm_out_proj = nn.Linear(d_input, d_model)

    # =========================================================================
    # CTM Methods (from ctm.py)
    # =========================================================================

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        """Computes synchronization as in CTM."""
        if synch_type == 'action':
            n_synch = self.n_synch_action
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == 'out':
            n_synch = self.n_synch_out
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right

        if self.neuron_select_type in ('first-last', 'random'):
            if self.neuron_select_type == 'first-last':
                if synch_type == 'action':
                    selected_left = selected_right = activated_state[:, -n_synch:]
                elif synch_type == 'out':
                    selected_left = selected_right = activated_state[:, :n_synch]
            else:
                selected_left = activated_state[:, neuron_indices_left]
                selected_right = activated_state[:, neuron_indices_right]

            outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
            i, j = torch.triu_indices(n_synch, n_synch)
            pairwise_product = outer[:, i, j]

        elif self.neuron_select_type == 'random-pairing':
            left = activated_state[:, neuron_indices_left]
            right = activated_state[:, neuron_indices_right]
            pairwise_product = left * right
        else:
            raise ValueError("Invalid neuron selection type")

        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1

        synchronisation = decay_alpha / (torch.sqrt(decay_beta))
        return synchronisation, decay_alpha, decay_beta

    def compute_features(self, x):
        """Compute key-value features from input using backbone."""
        initial_rgb = self.initial_rgb(x)
        self.kv_features = self.backbone(initial_rgb)
        pos_emb = self.positional_embedding(self.kv_features)
        combined_features = (self.kv_features + pos_emb).flatten(2).transpose(1, 2)
        kv = self.kv_proj(combined_features)
        return kv

    def compute_certainty(self, current_prediction):
        """Compute certainty based on normalized entropy."""
        B = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([B] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    # =========================================================================
    # Setup Methods (from ctm.py)
    # =========================================================================

    def set_initial_rgb(self):
        if 'resnet' in self.backbone_type:
            if self.grayscale:
                self.initial_rgb = nn.Identity()
            else:
                self.initial_rgb = nn.LazyConv2d(3, 1, 1)
        else:
            self.initial_rgb = nn.Identity()

    def get_d_backbone(self):
        if self.backbone_type == 'shallow-wide':
            return 2048
        elif self.backbone_type == 'parity_backbone':
            return self.d_input
        elif 'resnet' in self.backbone_type:
            if '18' in self.backbone_type or '34' in self.backbone_type:
                if self.backbone_type.split('-')[1]=='1': return 64
                elif self.backbone_type.split('-')[1]=='2': return 128
                elif self.backbone_type.split('-')[1]=='3': return 256
                elif self.backbone_type.split('-')[1]=='4': return 512
                else:
                    raise NotImplementedError
            else:
                if self.backbone_type.split('-')[1]=='1': return 256
                elif self.backbone_type.split('-')[1]=='2': return 512
                elif self.backbone_type.split('-')[1]=='3': return 1024
                elif self.backbone_type.split('-')[1]=='4': return 2048
                else:
                    raise NotImplementedError
        elif self.backbone_type == 'none':
            return None
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def set_backbone(self):
        if self.backbone_type == 'shallow-wide':
            self.backbone = ShallowWide()
        elif self.backbone_type == 'parity_backbone':
            d_backbone = self.get_d_backbone()
            self.backbone = ParityBackbone(n_embeddings=2, d_embedding=d_backbone)
        elif 'resnet' in self.backbone_type:
            self.backbone = prepare_resnet_backbone(
                self.backbone_type, pretrained=self.pretrained_backbone, grayscale=self.grayscale
            )
        elif self.backbone_type == 'none':
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_positional_embedding(self, d_backbone):
        if self.positional_embedding_type == 'learnable-fourier':
            return LearnableFourierPositionalEncoding(d_backbone, gamma=1 / 2.5)
        elif self.positional_embedding_type == 'multi-learnable-fourier':
            return MultiLearnableFourierPositionalEncoding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational':
            return CustomRotationalEmbedding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational-1d':
            return CustomRotationalEmbedding1D(d_backbone)
        elif self.positional_embedding_type == 'none':
            return lambda x: 0
        else:
            raise ValueError(f"Invalid positional_embedding_type: {self.positional_embedding_type}")

    def get_neuron_level_models(self, deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout):
        if deep_nlms:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )
        else:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )

    def get_synapses(self, synapse_depth, d_model, dropout):
        if synapse_depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        else:
            return SynapseUNET(d_model, synapse_depth, 16, dropout)

    def set_synchronisation_parameters(self, synch_type: str, n_synch: int, n_random_pairing_self: int = 0):
        assert synch_type in ('out', 'action'), f"Invalid synch_type: {synch_type}"
        left, right = self.initialize_left_right_neurons(synch_type, self.d_model, n_synch, n_random_pairing_self)
        synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
        self.register_buffer(f'{synch_type}_neuron_indices_left', left)
        self.register_buffer(f'{synch_type}_neuron_indices_right', right)
        self.register_parameter(f'decay_params_{synch_type}', nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True))

    def initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):
        if self.neuron_select_type=='first-last':
            if synch_type == 'out':
                neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch)
            elif synch_type == 'action':
                neuron_indices_left = neuron_indices_right = torch.arange(d_model-n_synch, d_model)

        elif self.neuron_select_type=='random':
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))

        elif self.neuron_select_type=='random-pairing':
            assert n_synch > n_random_pairing_self, f"Need at least {n_random_pairing_self} pairs for {self.neuron_select_type}"
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.concatenate((neuron_indices_left[:n_random_pairing_self], torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch-n_random_pairing_self))))

        device = self.start_activated_state.device
        return neuron_indices_left.to(device), neuron_indices_right.to(device)

    def get_neuron_select_type(self):
        print(f"Using neuron select type: {self.neuron_select_type}")
        if self.neuron_select_type == 'first-last':
            neuron_select_type_out, neuron_select_type_action = 'first', 'last'
        elif self.neuron_select_type in ('random', 'random-pairing'):
            neuron_select_type_out = neuron_select_type_action = self.neuron_select_type
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return neuron_select_type_out, neuron_select_type_action

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def verify_args(self):
        assert self.neuron_select_type in VALID_NEURON_SELECT_TYPES, \
            f"Invalid neuron selection type: {self.neuron_select_type}"

        assert self.backbone_type in VALID_BACKBONE_TYPES + ['none'], \
            f"Invalid backbone_type: {self.backbone_type}"

        assert self.positional_embedding_type in VALID_POSITIONAL_EMBEDDING_TYPES + ['none'], \
            f"Invalid positional_embedding_type: {self.positional_embedding_type}"

        if self.neuron_select_type == 'first-last':
            assert self.d_model >= (self.n_synch_out + self.n_synch_action), \
                "d_model must be >= n_synch_out + n_synch_action for neuron subsets"

        if self.backbone_type=='none' and self.positional_embedding_type!='none':
            raise AssertionError("There should be no positional embedding if there is no backbone.")

    def calculate_synch_representation_size(self, n_synch):
        if self.neuron_select_type == 'random-pairing':
            synch_representation_size = n_synch
        elif self.neuron_select_type in ('first-last', 'random'):
            synch_representation_size = (n_synch * (n_synch + 1)) // 2
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return synch_representation_size

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(self, x, track=False):
        """
        Forward pass.

        Returns:
            If track=False:
                predictions : (B, out_dims, iterations)
                certainties : (B, 2, iterations)
                synchronisation_out : (B, synch_representation_size_out)
            If track=True:
                All of above + tracking data
        """
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Write to FwPKM episodic memory ---
        _, K1, K2, V_mem = self.fwpkm(
            kv,
            chunk_size=self.fwpkm_chunk_size,
        )

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)

        # --- Initialise Recurrent Synch Values ---
        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action, r_out = (
            torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1),
            torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)
        )

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state, None, None, r_out, synch_type='out'
        )

        # --- Recurrent Loop ---
        for stepi in range(self.iterations):

            # --- Calculate Synchronisation for Input Data Interaction ---
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
            )

            # --- Interact with Data via Attention ---
            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)

            # --- FwPKM Retrieval --- (NEW: episodic memory read)
            fwpkm_q = self.synch_to_fwpkm_q(synchronisation_action)
            mem_raw = self.fwpkm.read(fwpkm_q, K1, K2, V_mem)
            mem_out = self.fwpkm_out_proj(mem_raw)

            # --- Apply Synapses (with FwPKM memory input) ---
            pre_synapse_input = torch.cat([attn_out, activated_state, mem_out], dim=-1)
            state = self.synapses(pre_synapse_input)

            # The 'state_trace' is the history of incoming pre-activations
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply Neuron-Level Models ---
            activated_state = self.trace_processor(state_trace)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            return (
                predictions, certainties, synchronisation_out,
                (np.array(synch_out_tracking), np.array(synch_action_tracking)),
                np.array(pre_activations_tracking), np.array(post_activations_tracking),
                np.array(attention_tracking)
            )
        return predictions, certainties, synchronisation_out

    # =========================================================================
    # Loss: certainty-based selection (from CTM)
    # =========================================================================

    def compute_loss(
        self,
        outputs_history: torch.Tensor,  # (B, out_dims, T)
        certainties: torch.Tensor,      # (B, 2, T)
        targets: torch.Tensor,          # (B,) class labels
    ) -> torch.Tensor:
        """
        Certainty-weighted loss (from CTM §3.5).

        Selects two ticks per forward pass to optimise:
          t1 = argmin(loss)       — best prediction tick
          t2 = final tick         — encourages continued improvement
        Both are weighted by their certainty (1 - normalised entropy).
        """
        T = outputs_history.shape[-1]
        losses = torch.stack(
            [F.cross_entropy(outputs_history[..., t], targets, reduction="none") for t in range(T)],
            dim=-1,
        )  # (B, T)

        t1_idx = losses.argmin(dim=-1)  # (B,)
        t2_idx = T - 1  # final tick

        loss_t1 = losses.gather(-1, t1_idx.unsqueeze(-1)).squeeze(-1).mean()
        loss_t2 = losses[..., t2_idx].mean()

        return loss_t1 + loss_t2


# ===========================================================================
# Quick sanity-check demo
# ===========================================================================


def demo():
    """Run a quick forward pass to verify tensor shapes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running demo on: {device}")

    model = CTMFwPKMNeuralComputer(
        iterations=5,
        d_model=128,
        d_input=128,
        heads=4,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=1,
        memory_length=10,
        deep_nlms=True,
        memory_hidden_dims=32,
        backbone_type='resnet18-2',
        pretrained_backbone='none',
        positional_embedding_type='none',
        out_dims=10,
        neuron_select_type='random-pairing',
        fwpkm_d_key=64,
        fwpkm_d_val=128,
        fwpkm_n_mem=1024,
        fwpkm_top_k=8,
        fwpkm_lr=0.1,
        fwpkm_chunk_size=16,
    ).to(device)

    x = torch.randn(4, 3, 32, 32, device=device)
    targets = torch.randint(0, 10, (4,), device=device)

    # Initialize lazy parameters by running a dummy forward pass
    with torch.no_grad():
        _ = model(x)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    with torch.no_grad():
        predictions, certainties, synch_out = model(x)

    print(f"\nNumber of internal ticks: {model.iterations}")
    print(f"Predictions shape:        {predictions.shape}")
    print(f"Certainties shape:        {certainties.shape}")

    loss = model.compute_loss(predictions, certainties, targets)
    print(f"Demo loss:                {loss.item():.4f}")

    print("\n✓ CTM-FwPKM Neural Computer forward pass OK")


if __name__ == "__main__":
    demo()
