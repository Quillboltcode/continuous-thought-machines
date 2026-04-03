import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.ctm import ContinuousThoughtMachine
from models.modules import SuperLinear, Squeeze


class SyncGatedMessagePassing(nn.Module):
    """
    Synchronization-Gated Graph Neural Network (SG-GNN) for CTM.

    Uses the synchronization matrix S^t as a dynamic adjacency matrix to gate information
    flow between neurons. Based on the Communication-through-Coherence (CTC) hypothesis.

    a_i^{t+1} = NLM_i( sum_{j in N} sigma(S^t_ij) * phi(z_j^t, o^t) )

    Args:
        d_model: Number of neurons (hidden dimension).
        d_input: Dimension of external input (attention output).
        message_hidden: Hidden dimension for message passing MLP.
        use_learnable_decay: Whether to use learnable sync decay.
    """

    def __init__(self, d_model, d_input, message_hidden=None, use_learnable_decay=True):
        super().__init__()
        self.d_model = d_model
        self.d_input = d_input
        self.message_hidden = message_hidden or d_model // 2

        self.use_learnable_decay = use_learnable_decay

        self.message_net = nn.Sequential(
            nn.Linear(d_model + d_input, self.message_hidden),
            nn.GELU(),
            nn.Linear(self.message_hidden, d_model),
        )

        if use_learnable_decay:
            self.register_parameter('sync_decay', nn.Parameter(torch.ones(1) * 0.1))

    def forward(self, activated_state, synchronisation_out, attn_expanded):
        """
        Args:
            activated_state: (B, d_model) - post-activations z_j^t
            synchronisation_out: (B, synch_size) - sync representation
            attn_expanded: (B, d_input) - external attention input o^t

        Returns:
            new_state: (B, d_model) - updated pre-activations a^{t+1}
        """
        d_model = self.d_model

        gate_weights = F.sigmoid(synchronisation_out)
        if gate_weights.size(-1) != d_model:
            gate_weights = gate_weights.repeat(1, d_model // gate_weights.size(-1))

        message_input = torch.cat([activated_state, attn_expanded], dim=-1)

        messages = self.message_net(message_input)

        gated_messages = messages * gate_weights

        new_state = gated_messages

        return new_state


class ContinuousThoughtMachineSGD(ContinuousThoughtMachine):
    """
    CTM with Sync-Graph Dynamics (SGD).

    Replaces the static SynapseModel with a Synchronization-Gated GNN that uses
    the synchronization matrix as a dynamic adjacency matrix for information routing.

    Based on the Communication-through-Coherence (CTC) hypothesis from neuroscience.
    """

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 synapse_depth=1,
                 memory_length=25,
                 deep_nlms=True,
                 memory_hidden_dims=64,
                 do_layernorm_nlm=False,
                 backbone_type='resnet18-4',
                 pretrained_backbone='none',
                 positional_embedding_type='none',
                 out_dims=10,
                 prediction_reshaper=[-1],
                 dropout=0.0,
                 dropout_nlm=None,
                 neuron_select_type='random-pairing',
                 n_random_pairing_self=0,
                 grayscale=False,
                 use_sync_gated=True,
                 message_hidden=None,
                 **kwargs):
        
        self.use_sync_gated = use_sync_gated
        self.message_hidden = message_hidden

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

        if use_sync_gated:
            self.synapse = SyncGatedMessagePassing(
                d_model=d_model,
                d_input=d_input,
                message_hidden=message_hidden,
                use_learnable_decay=True
            )

    def get_synapses(self, synapse_depth, d_model, dropout):
        if self.use_sync_gated:
            return SyncGatedMessagePassing(
                d_model=d_model,
                d_input=d_input,
                message_hidden=self.message_hidden,
                use_learnable_decay=True
            )
        return super().get_synapses(synapse_depth, d_model, dropout)

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []
        sync_graph_tracking = []

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

        for stepi in range(self.iterations):
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
            )

            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)

            if self.use_sync_gated:
                state, decay_alpha_out, decay_beta_out = self.synapse(
                    activated_state,
                    synchronisation_action,
                    attn_out,
                    decay_alpha_out,
                    decay_beta_out
                )
                state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            else:
                pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)
                state = self.synapses(pre_synapse_input)
                state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            activated_state = self.trace_processor(state_trace)

            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )

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
            return predictions, certainties, (synch_out_tracking, synch_action_tracking), pre_activations_tracking, post_activations_tracking, attention_tracking
        return predictions, certainties, synchronisation_out