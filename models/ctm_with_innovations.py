"""
CTM with Innovations
===================
Extended Continuous Thought Machine with five architectural innovations:
    1. Gated Synchronization Highway (GSH)
    2. Hierarchical NLM Ensembles (HNE)
    3. Predictive Synchrony Loss (PSL)
    4. Sparse Adaptive Neuron Pairing (SANP)
    5. Cross-Tick Contrastive Synchronization (CTCS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.ctm import ContinuousThoughtMachine
from models.modules import SuperLinear, Squeeze as ModuleSqueeze
from models.constants import (
    VALID_NEURON_SELECT_TYPES,
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)


class GatedSynchronizationHighway(nn.Module):
    """
    Gates the synchronisation representation using the current activated state.

    Args:
        d_model: dimensionality of the activated_state (z^t)
        synch_size: dimensionality of the synchronisation representation
    """

    def __init__(self, d_model: int, synch_size: int):
        super().__init__()
        self.gate_proj = nn.Sequential(nn.Linear(d_model, synch_size), nn.Sigmoid())
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self, synchronisation: torch.Tensor, activated_state: torch.Tensor
    ) -> torch.Tensor:
        gate = self.gate_proj(activated_state)
        gated = gate * synchronisation + self.residual_scale * synchronisation
        return gated


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

    def __init__(self, d_model: int, max_memory: int, group_configs: list, dropout: float = 0):
        super().__init__()
        assert sum(g["n_neurons"] for g in group_configs) == d_model, (
            "Group neuron counts must sum to d_model"
        )

        self.d_model = d_model
        self.max_memory = max_memory
        self.group_configs = group_configs

        self.nlms = nn.ModuleList()
        self.group_slices = []
        start = 0
        for cfg in group_configs:
            n = cfg["n_neurons"]
            m = cfg["memory"]
            h = cfg["hidden"]
            end = start + n
            self.group_slices.append((start, end, m))
            nlm = nn.Sequential(
                SuperLinear(in_dims=m, out_dims=2 * h, N=n),
                nn.GLU(dim=-1),
                SuperLinear(in_dims=h, out_dims=2, N=n),
                nn.GLU(dim=-1),
                ModuleSqueeze(-1),
            )
            self.nlms.append(nlm)
            start = end

    def forward(self, state_trace: torch.Tensor) -> torch.Tensor:
        parts = []
        for nlm, (start, end, m) in zip(self.nlms, self.group_slices):
            group_trace = state_trace[:, start:end, -m:]
            group_out = nlm(group_trace)
            parts.append(group_out)
        return torch.cat(parts, dim=1)


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

        all_i = torch.randint(0, d_model, (self.init_top_k,))
        all_j = torch.randint(0, d_model, (self.init_top_k,))
        self.register_buffer("candidate_i", all_i)
        self.register_buffer("candidate_j", all_j)

        self.pair_logits = nn.Parameter(torch.zeros(n_pairs, self.init_top_k))

    def forward(self, activated_state: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.pair_logits / self.temperature, dim=-1)

        left = activated_state[:, self.candidate_i]
        right = activated_state[:, self.candidate_j]
        pair_products = left * right

        if self.training:
            synch = torch.einsum("bk,nk->bn", pair_products, weights)
        else:
            best_pairs = self.pair_logits.argmax(dim=-1)
            synch = pair_products[:, best_pairs]

        return synch

    @torch.no_grad()
    def refresh_candidates(self, keep_top: int = 100):
        importance = self.pair_logits.max(0).values
        top_idx = importance.topk(keep_top).indices
        new_count = self.init_top_k - keep_top

        new_i = torch.randint(
            0, self.d_model, (new_count,), device=self.candidate_i.device
        )
        new_j = torch.randint(
            0, self.d_model, (new_count,), device=self.candidate_j.device
        )

        self.candidate_i = torch.cat([self.candidate_i[top_idx], new_i])
        self.candidate_j = torch.cat([self.candidate_j[top_idx], new_j])

        new_logits = torch.zeros(
            self.n_pairs, self.init_top_k, device=self.pair_logits.device
        )
        new_logits[:, :keep_top] = self.pair_logits[:, top_idx].detach()
        self.pair_logits = nn.Parameter(new_logits)


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
        self.predictor = nn.Sequential(
            nn.Linear(synch_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, synch_size),
        )

    def forward(self, synch_out_sequence: torch.Tensor) -> torch.Tensor:
        T = synch_out_sequence.size(0)
        if T < 2:
            return synch_out_sequence.new_tensor(0.0)

        current = synch_out_sequence[:-1]
        target = synch_out_sequence[1:]
        predicted = self.predictor(current)
        l_pred = F.mse_loss(predicted, target.detach())

        flat = synch_out_sequence.view(-1, synch_out_sequence.size(-1))
        flat_norm = (flat - flat.mean(0)) / (flat.std(0) + 1e-6)
        N = flat_norm.size(0)
        C = (flat_norm.T @ flat_norm) / N
        eye = torch.eye(C.size(0), device=C.device)
        l_decorr = ((C - eye) ** 2).sum() / C.numel()

        return l_pred + self.lambda_decorr * l_decorr


class CrossTickContrastiveLoss(nn.Module):
    """
    Contrastive loss over the synchronisation trajectory.

    Args:
        temperature:  InfoNCE temperature τ
        n_ticks_sample: number of adjacent tick pairs to sample per forward pass
    """

    def __init__(self, temperature: float = 0.07, n_ticks_sample: int = 8):
        super().__init__()
        self.temperature = temperature
        self.n_ticks_sample = n_ticks_sample

    def forward(self, synch_out_sequence: torch.Tensor) -> torch.Tensor:
        T, B, D = synch_out_sequence.shape
        if T < 2 or B < 2:
            return synch_out_sequence.new_tensor(0.0)

        tick_starts = torch.randperm(T - 1, device=synch_out_sequence.device)[
            : self.n_ticks_sample
        ]

        losses = []
        for t in tick_starts:
            anchor = synch_out_sequence[t]
            positive = synch_out_sequence[t + 1]

            anchor = F.normalize(anchor, dim=-1)
            positive = F.normalize(positive, dim=-1)

            sim_matrix = torch.mm(anchor, positive.T) / self.temperature

            labels = torch.arange(B, device=anchor.device)

            tick_loss = F.cross_entropy(sim_matrix, labels)
            losses.append(tick_loss)

        return torch.stack(losses).mean()


class CTMWithInnovations(ContinuousThoughtMachine):
    """
    Extended CTM with all five innovations integrated.
    
    Innovations:
        1. Gated Synchronization Highway (GSH) - gates synch with activated_state
        2. Hierarchical NLM Ensembles (HNE) - replaces trace_processor
        3. Predictive Synchrony Loss (PSL) - auxiliary loss module
        4. Sparse Adaptive Neuron Pairing (SANP) - replaces static neuron pairing
        5. Cross-Tick Contrastive Synchronization (CTCS) - auxiliary loss module

    Additional Args:
        use_gsh (bool): Enable Gated Synchronization Highway
        use_hne (bool): Enable Hierarchical NLM Ensembles
        use_sanp (bool): Enable Sparse Adaptive Neuron Pairing
        hne_group_configs (list): Group configs for HNE (required if use_hne=True)
        sanp_init_top_k (int): Number of candidate pairs for SANP
    """

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 backbone_type,
                 pretrained_backbone,
                 positional_embedding_type,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 neuron_select_type='random-pairing',
                 n_random_pairing_self=0,
                 grayscale=False,
                 use_gsh=False,
                 use_hne=False,
                 use_sanp=False,
                 hne_group_configs=None,
                 sanp_init_top_k=1000,
                 ):
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=heads,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            backbone_type=backbone_type,
            pretrained_backbone=pretrained_backbone,
            positional_embedding_type=positional_embedding_type,
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=dropout,
            dropout_nlm=dropout_nlm,
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=n_random_pairing_self,
            grayscale=grayscale,
        )

        self.use_gsh = use_gsh
        self.use_hne = use_hne
        self.use_sanp = use_sanp

        if use_gsh:
            self.gsh_action = GatedSynchronizationHighway(
                d_model, self.synch_representation_size_action
            )
            self.gsh_out = GatedSynchronizationHighway(
                d_model, self.synch_representation_size_out
            )

        if use_hne:
            assert hne_group_configs is not None, "hne_group_configs required when use_hne=True"
            self.trace_processor = HierarchicalNLMEnsemble(
                d_model=d_model,
                max_memory=memory_length,
                group_configs=hne_group_configs,
                dropout=dropout_nlm if dropout_nlm else dropout
            )

        if use_sanp:
            self.sanp_action = SparseAdaptiveNeuronPairing(
                d_model=d_model,
                n_pairs=n_synch_action,
                init_top_k=sanp_init_top_k
            )
            self.sanp_out = SparseAdaptiveNeuronPairing(
                d_model=d_model,
                n_pairs=n_synch_out,
                init_top_k=sanp_init_top_k
            )

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        if synch_type == 'action':
            n_synch = self.n_synch_action
        elif synch_type == 'out':
            n_synch = self.n_synch_out
        else:
            raise ValueError(f"Invalid synch_type: {synch_type}")

        if self.use_sanp:
            if synch_type == 'action':
                pairwise_product = self.sanp_action(activated_state)
            else:
                pairwise_product = self.sanp_out(activated_state)
        else:
            if synch_type == 'action':
                neuron_indices_left = self.action_neuron_indices_left
                neuron_indices_right = self.action_neuron_indices_right
            else:
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

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        kv = self.compute_features(x)

        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)

        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')

        synch_out_history = []

        for stepi in range(self.iterations):
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
            )

            if self.use_gsh:
                synchronisation_action = self.gsh_action(synchronisation_action, activated_state)

            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.cat((attn_out, activated_state), dim=-1)

            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            activated_state = self.trace_processor(state_trace)

            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )

            if self.use_gsh:
                synchronisation_out = self.gsh_out(synchronisation_out, activated_state)

            synch_out_history.append(synchronisation_out)

            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        if track:
            return predictions, certainties, (np.array(synch_out_tracking), np.array(synch_action_tracking)), np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking)
        
        synch_out_seq = torch.stack(synch_out_history) if synch_out_history else None
        return predictions, certainties, synchronisation_out, synch_out_seq


def _test_all_innovations():
    torch.manual_seed(42)
    B, D, T, synch_size, memory = 4, 128, 10, 64, 50

    print("Testing Innovation 1: Gated Synchronization Highway...")
    gsh = GatedSynchronizationHighway(d_model=D, synch_size=synch_size)
    synch = torch.randn(B, synch_size)
    z = torch.randn(B, D)
    out = gsh(synch, z)
    assert out.shape == (B, synch_size), f"GSH shape mismatch: {out.shape}"
    print(f"  ✓ GSH output shape: {tuple(out.shape)}")

    print("Testing Innovation 2: Hierarchical NLM Ensembles...")
    hne = HierarchicalNLMEnsemble(
        d_model=D,
        max_memory=memory,
        group_configs=[
            {"n_neurons": 42, "memory": 5, "hidden": 8},
            {"n_neurons": 43, "memory": 25, "hidden": 16},
            {"n_neurons": 43, "memory": 50, "hidden": 32},
        ],
    )
    trace = torch.randn(B, D, memory)
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

    print("\nTesting CTMWithInnovations model integration...")
    
    hne_group_configs = [
        {"n_neurons": 42, "memory": 5, "hidden": 8},
        {"n_neurons": 43, "memory": 25, "hidden": 16},
        {"n_neurons": 43, "memory": 50, "hidden": 32},
    ]
    
    model = CTMWithInnovations(
        iterations=5,
        d_model=D,
        d_input=128,
        heads=4,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=1,
        memory_length=50,
        deep_nlms=True,
        memory_hidden_dims=16,
        do_layernorm_nlm=False,
        backbone_type='none',
        pretrained_backbone='none',
        positional_embedding_type='none',
        out_dims=10,
        dropout=0.1,
        use_gsh=True,
        use_hne=True,
        use_sanp=False,
        hne_group_configs=hne_group_configs,
        sanp_init_top_k=256,
    )
    
    x = torch.randn(B, 3, 32, 32)
    model.train()
    predictions, certainties, synch_out, synch_out_seq = model(x)
    assert predictions.shape == (B, 10, 5), f"Predictions shape mismatch: {predictions.shape}"
    assert certainties.shape == (B, 2, 5), f"Certainties shape mismatch: {certainties.shape}"
    assert synch_out_seq is not None, "synch_out_seq should be collected"
    assert synch_out_seq.shape == (5, B, 32), f"synch_out_seq shape mismatch: {synch_out_seq.shape}"
    print(f"  ✓ CTMWithInnovations forward pass successful")
    print(f"    predictions shape: {tuple(predictions.shape)}")
    print(f"    certainties shape: {tuple(certainties.shape)}")
    print(f"    synch_out_seq shape: {tuple(synch_out_seq.shape)}")
    
    model.eval()
    with torch.no_grad():
        predictions_eval, certainties_eval, synch_out_eval, synch_out_seq_eval = model(x)
    print(f"  ✓ CTMWithInnovations eval pass successful")
    
    print("\nTesting auxiliary losses with collected synch_out_seq...")
    psl = PredictiveSynchronyLoss(synch_size=32, hidden=32)
    ctcs = CrossTickContrastiveLoss(temperature=0.07, n_ticks_sample=4)
    
    psl_loss = psl(synch_out_seq)
    ctcs_loss = ctcs(synch_out_seq)
    print(f"  ✓ PSL loss: {psl_loss.item():.4f}")
    print(f"  ✓ CTCS loss: {ctcs_loss.item():.4f}")

    print("\nAll innovations passed! ✓")


if __name__ == "__main__":
    _test_all_innovations()
