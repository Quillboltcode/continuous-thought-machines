import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F

from models.modules import ParityBackbone, SynapseUNET, Squeeze, SuperLinear, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_NEURON_SELECT_TYPES,
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)

from utils.losses import PonderLoss, LoopLMLoss, CTMGatedLoss


class CTMGated(nn.Module):
    """
    Continuous Thought Machine with Gated Early Exit (CTM-Gated).
    
    Extends the CTM with a gating mechanism that allows early termination of the thinking process.
    The gate can be based on:
    - Certainty threshold: Exit when model's certainty exceeds a threshold
    - Certainty argmax: Exit at the step with highest certainty (most aligned with CTM loss)
    - Ponder loss: KL divergence with geometric distribution (like PonderNet)
    - Normal distribution: Exit based on normal distribution CDF
    - Learned gate: A learnable halting probability
    - Improvement predictor: Learn to predict future improvement and exit when improvement < threshold
    
    Args:
        exit_strategy (str): Strategy for early exit. Options:
            - 'certainty': Exit when certainty > threshold
            - 'ponder': Use PonderNet-style halting probabilities
            - 'normal': Use normal distribution for halting
            - 'learned': Learn halting probabilities
            - 'improvement': Learn to predict improvement and exit when predicted improvement < threshold
            - 'none': No early exit (standard CTM behavior)
        exit_threshold (float): Threshold for certainty-based exit [0, 1]
        lambda_p (float): Geometric distribution parameter for ponder loss
        beta (float): Regularization weight for ponder loss
        use_halt_logits (bool): Whether to use learnable halting logits
        min_steps (int): Minimum number of thinking steps before early exit
        max_iterations (int): Maximum number of thinking steps (same as iterations)
        improvement_threshold (float): Threshold for improvement-based exit [0, 1]
        stable_ticks (int): Number of consecutive ticks certainty must exceed threshold before halting (certainty-momentum)
        
    Inherits most args from CTM:
        iterations (int): Number of internal 'thought' ticks (T, in paper).
        d_model (int): Core dimensionality of the CTM's latent space.
        d_input (int): Dimensionality of projected attention outputs.
        heads (int): Number of attention heads.
        n_synch_out (int): Number of neurons used for output synchronisation.
        n_synch_action (int): Number of neurons used for action/attention synchronisation.
        synapse_depth (int): Depth of the synapse model.
        memory_length (int): History length for Neuron-Level Models.
        deep_nlms (bool): Use deeper (2-layer) NLMs if True.
        memory_hidden_dims (int): Hidden dimension size for deep NLMs.
        do_layernorm_nlm (bool): Apply LayerNorm within NLMs.
        backbone_type (str): Type of feature extraction backbone.
        pretrained_backbone (str): Use a pretrained backbone.
        positional_embedding_type (str): Type of positional embedding.
        out_dims (int): Output dimension size.
        prediction_reshaper (list): Shape for reshaping predictions.
        dropout (float): Dropout rate.
        neuron_select_type (str): Neuron selection strategy.
        n_random_pairing_self (int): Number of neurons for self-to-self synch.
        grayscale (bool): Use grayscale images.
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
                  exit_strategy='certainty',
                  exit_threshold=0.9,
                  lambda_p=0.2,
                  beta=0.01,
                   use_halt_logits=False,
                   min_steps=1,
                   improvement_threshold=0.01,
                   stable_ticks=1,
                  ):
        super(CTMGated, self).__init__()

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
        self.memory_length = memory_length
        self.grayscale = grayscale
        
        self.exit_strategy = exit_strategy
        self.exit_threshold = exit_threshold
        self.lambda_p = lambda_p
        self.beta = beta
        self.min_steps = min_steps
        self.improvement_threshold = improvement_threshold
        self.stable_ticks = stable_ticks
        
        dropout_nlm = dropout if dropout_nlm is None else dropout_nlm

        self.verify_args()

        d_backbone = self.get_d_backbone()
        self.set_initial_rgb()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)
        self.kv_proj = nn.Sequential(nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input)) if heads else None
        self.q_proj = nn.LazyLinear(self.d_input) if heads else None
        self.attention = nn.MultiheadAttention(self.d_input, heads, dropout, batch_first=True) if heads else None
        
        self.synapses = self.get_synapses(synapse_depth, d_model, dropout)
        self.trace_processor = self.get_neuron_level_models(deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout_nlm)

        self.register_parameter('start_activated_state', nn.Parameter(torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model)))))
        self.register_parameter('start_trace', nn.Parameter(torch.zeros((d_model, memory_length)).uniform_(-math.sqrt(1/(d_model+memory_length)), math.sqrt(1/(d_model+memory_length)))))

        self.neuron_select_type_out, self.neuron_select_type_action = self.get_neuron_select_type()
        self.synch_representation_size_action = self.calculate_synch_representation_size(self.n_synch_action)
        self.synch_representation_size_out = self.calculate_synch_representation_size(self.n_synch_out)
        
        for synch_type, size in (('action', self.synch_representation_size_action), ('out', self.synch_representation_size_out)):
            print(f"Synch representation size {synch_type}: {size}")
        if self.synch_representation_size_action:
            self.set_synchronisation_parameters('action', self.n_synch_action, n_random_pairing_self)
        self.set_synchronisation_parameters('out', self.n_synch_out, n_random_pairing_self)

        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))
        
        if exit_strategy == 'learned' or use_halt_logits:
            self.halt_logits = nn.Linear(self.synch_representation_size_out, 1)
            nn.init.zeros_(self.halt_logits.weight)
            nn.init.zeros_(self.halt_logits.bias)
        
        if exit_strategy == 'improvement':
            self.improvement_predictor = nn.Sequential(
                nn.Linear(self.d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
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
        initial_rgb = self.initial_rgb(x)
        self.kv_features = self.backbone(initial_rgb)
        pos_emb = self.positional_embedding(self.kv_features)
        combined_features = (self.kv_features + pos_emb).flatten(2).transpose(1, 2)
        kv = self.kv_proj(combined_features)
        return kv

    def compute_certainty(self, current_prediction):
        B = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([B] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    def compute_halt_probability(self, synchronisation_out, step, total_steps):
        B = synchronisation_out.size(0)
        
        if self.exit_strategy == 'learned' or hasattr(self, 'halt_logits'):
            halt_logits = self.halt_logits(synchronisation_out).squeeze(-1)
            halt_prob = torch.sigmoid(halt_logits)
            
        elif self.exit_strategy == 'ponder':
            progress = step / max(total_steps - 1, 1)
            halt_prob = self.lambda_p * torch.ones(B, device=synchronisation_out.device)
            
        elif self.exit_strategy == 'normal':
            progress = torch.tensor(step / max(total_steps - 1, 1), device=synchronisation_out.device)
            mean = 0.5
            std = 0.25
            cdf_value = 0.5 * (1 + torch.erf((progress - mean) / (std * math.sqrt(2))))
            halt_prob = cdf_value * torch.ones(B, device=synchronisation_out.device)
            
        elif self.exit_strategy == 'certainty':
            halt_prob = None
            
        elif self.exit_strategy == 'improvement':
            halt_prob = None
            
        else:
            halt_prob = None
            
        return halt_prob

    def compute_predicted_improvement(self, activated_state):
        if hasattr(self, 'improvement_predictor'):
            predicted_improvement = self.improvement_predictor(activated_state).squeeze(-1)
            return predicted_improvement
        return None

    def should_exit(self, step, certainties, halt_prob=None, activated_state=None, stable_count=None):
        B = certainties.size(0)
        
        if self.exit_strategy == 'none':
            return torch.zeros(B, dtype=torch.bool, device=certainties.device)
        
        elif self.exit_strategy == 'certainty':
            certainty_values = certainties[:, 1, step]
            should_exit = (certainty_values > self.exit_threshold) & (step >= self.min_steps)
            
            if self.stable_ticks > 1 and stable_count is not None:
                should_exit = should_exit & (stable_count >= self.stable_ticks)
            
        elif self.exit_strategy in ('ponder', 'normal', 'learned'):
            if halt_prob is None:
                return torch.zeros(B, dtype=torch.bool, device=certainties.device)
            if self.training:
                rand_val = torch.rand(B, device=certainties.device)
                should_exit = (halt_prob > rand_val) & (step >= self.min_steps)
            else:
                should_exit = (halt_prob > 0.5) & (step >= self.min_steps)
            
        elif self.exit_strategy == 'improvement':
            if activated_state is not None and hasattr(self, 'improvement_predictor'):
                predicted_improvement = self.compute_predicted_improvement(activated_state)
                should_exit = (predicted_improvement < self.improvement_threshold) & (step >= self.min_steps)
            else:
                should_exit = torch.zeros(B, dtype=torch.bool, device=certainties.device)
            
        else:
            should_exit = torch.zeros(B, dtype=torch.bool, device=certainties.device)
            
        return should_exit

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
            self.backbone = prepare_resnet_backbone(self.backbone_type, pretrained=self.pretrained_backbone, grayscale=self.grayscale)
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

        return neuron_indices_left, neuron_indices_right

    def get_neuron_select_type(self):
        print(f"Using neuron select type: {self.neuron_select_type}")
        if self.neuron_select_type == 'first-last':
            neuron_select_type_out, neuron_select_type_action = 'first', 'last'
        elif self.neuron_select_type in ('random', 'random-pairing'):
            neuron_select_type_out = neuron_select_type_action = self.neuron_select_type
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return neuron_select_type_out, neuron_select_type_action

    def verify_args(self):
        assert self.exit_strategy in ['certainty', 'ponder', 'normal', 'learned', 'none', 'improvement'], \
            f"Invalid exit_strategy: {self.exit_strategy}"
        
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

    def compute_ponder_loss(self, predictions, certainties, targets):
        B, C, T = predictions.size()
        
        if hasattr(self, 'halt_logits'):
            halt_probs = torch.sigmoid(self.halt_logits(predictions.mean(dim=2)))
            halt_probs = halt_probs.unsqueeze(-1).expand(-1, -1, T)
        else:
            ones = torch.ones(B, 1, device=predictions.device)
            if self.exit_strategy == 'ponder':
                base_prob = self.lambda_p * ones.squeeze(-1)
            else:
                base_prob = 0.1 * ones.squeeze(-1)
            progress = torch.arange(T, device=predictions.device).float() / max(T - 1, 1)
            if self.exit_strategy == 'normal':
                mean, std = 0.5, 0.25
                base_prob = 0.5 * (1 + torch.erf((progress - mean) / (std * math.sqrt(2))))
            base_prob = base_prob.unsqueeze(0).expand(B, -1)
            halt_probs = base_prob
        
        ones = torch.ones(B, 1, device=predictions.device)
        cum_not_halted = torch.cumprod(torch.cat((ones, 1.0 - halt_probs[:, :-1]), dim=1), dim=1)
        p_t = halt_probs * cum_not_halted
        p_t[:, -1] = 1.0 - p_t[:, :-1].sum(dim=1)
        
        expanded_targets = targets.unsqueeze(1).expand(-1, T)
        ce_per_step = F.cross_entropy(predictions, expanded_targets, reduction='none')
        loss_rec = (p_t * ce_per_step).sum(dim=1).mean()
        
        steps = torch.arange(1, T + 1, device=predictions.device).float()
        geo_prior = torch.pow(1 - self.lambda_p, steps - 1) * self.lambda_p
        geo_prior = geo_prior / geo_prior.sum()
        
        kl_div = (p_t * (torch.log(p_t + 1e-8) - torch.log(geo_prior + 1e-8))).sum(dim=1).mean()

        total_loss = loss_rec + self.beta * kl_div
        
        return total_loss, p_t

    def compute_improvement_predictor_loss(self, predictions, targets, activated_states_list):
        if not hasattr(self, 'improvement_predictor') or activated_states_list is None:
            return None
        
        B, C, T = predictions.size()
        if T < 2 or len(activated_states_list) < T - 1:
            return None
        
        targets_expanded = targets.unsqueeze(1).expand(-1, T)
        step_losses = F.cross_entropy(predictions, targets_expanded, reduction='none')
        
        actual_improvements = torch.zeros(B, T - 1, device=predictions.device)
        for t in range(T - 1):
            actual_improvements[:, t] = step_losses[:, t] - step_losses[:, t + 1]
        
        predicted_improvements = torch.zeros(B, T - 1, device=predictions.device)
        for t in range(T - 1):
            activated_state = activated_states_list[t]
            predicted_improvements[:, t] = self.improvement_predictor(activated_state).squeeze(-1)
        
        loss = F.mse_loss(predicted_improvements, actual_improvements.clamp(0, 1))
        
        return loss

    def forward(self, x, track=False, use_early_exit=True):
        B = x.size(0)
        device = x.device

        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []
        exit_steps = torch.full((B,), self.iterations - 1, dtype=torch.long, device=device)
        halt_probs_all = []
        activated_states_all = []

        kv = self.compute_features(x)

        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)

        decay_alpha_action, decay_beta_action = None, None
        r_action = torch.exp(-self.decay_params_action.clamp(0, 15)).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.decay_params_out.clamp(0, 15)).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')

        all_halted = torch.zeros(B, dtype=torch.bool, device=device)
        stable_count = torch.zeros(B, dtype=torch.long, device=device)
        
        for stepi in range(self.iterations):
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')

            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            activated_state = self.trace_processor(state_trace)

            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            if use_early_exit and not all_halted.all():
                if self.stable_ticks > 1:
                    certainty_above_threshold = certainties[:, 1, stepi] > self.exit_threshold
                    if stepi == 0:
                        stable_count = certainty_above_threshold.long()
                    else:
                        stable_count = torch.where(
                            certainty_above_threshold,
                            stable_count + 1,
                            torch.zeros_like(stable_count)
                        )
                else:
                    stable_count = None
                
                halt_prob = self.compute_halt_probability(synchronisation_out, stepi, self.iterations)
                
                if halt_prob is not None:
                    halt_probs_all.append(halt_prob)
                
                should_exit = self.should_exit(stepi, certainties, halt_prob, activated_state, stable_count)
                newly_halted = should_exit & ~all_halted
                exit_steps[newly_halted] = stepi
                all_halted = all_halted | should_exit
                
                if not self.training and all_halted.all():
                    # Only break if not tracking, otherwise we need full tracking history
                    if not track:
                        # Slice tensors to actual steps taken if needed? 
                        # Actually just breaking is fine, the rest of predictions will be garbage/empty for these indices but they should have exited anyway.
                        break
            
            activated_states_all.append(activated_state.detach())

            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        if track:
            return predictions, certainties, (np.array(synch_out_tracking), np.array(synch_action_tracking)), np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking), exit_steps
        
        return predictions, certainties, synchronisation_out, exit_steps, activated_states_all

    @torch.no_grad()
    def adaptive_forward(self, x, threshold=None, max_ticks=None):
        """
        Runs CTM ticks per-sample, halting each independently.
        Returns predictions, halt ticks, and the actual outputs used.
        
        Args:
            x: Input tensor [B, ...]
            threshold: Override exit threshold for certainty-based exit
            max_ticks: Maximum number of thinking steps
            
        Returns:
            final_predictions: [B, C] - prediction at each sample's halt tick
            halt_ticks: [B] - the tick at which each sample halted
            all_predictions: [B, C, T] - predictions at all ticks
            all_certainties: [B, 2, T] - certainties at all ticks
        """
        if threshold is None:
            threshold = self.exit_threshold
        if max_ticks is None:
            max_ticks = self.iterations
            
        B = x.size(0)
        device = x.device
        
        kv = self.compute_features(x)
        
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)
        
        predictions = torch.empty(B, self.out_dims, max_ticks, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, max_ticks, device=device, dtype=torch.float32)
        
        decay_alpha_action, decay_beta_action = None, None
        r_action = torch.exp(-self.decay_params_action.clamp(0, 15)).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.decay_params_out.clamp(0, 15)).unsqueeze(0).repeat(B, 1)
        
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')
        
        halted = torch.zeros(B, dtype=torch.bool, device=device)
        halt_tick = torch.full((B,), max_ticks - 1, dtype=torch.long, device=device)
        stable_count = torch.zeros(B, dtype=torch.long, device=device)
        
        for t in range(max_ticks):
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')
            
            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, _ = self.attention(q, kv, kv, average_attn_weights=False, need_weights=False)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)
            
            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            
            activated_state = self.trace_processor(state_trace)
            
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')
            
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)
            
            predictions[..., t] = current_prediction
            certainties[..., t] = current_certainty
            
            if self.stable_ticks > 1:
                certainty_above_threshold = certainties[:, 1, t] > threshold
                stable_count = torch.where(
                    certainty_above_threshold,
                    stable_count + 1,
                    torch.zeros_like(stable_count)
                )
            
            newly_halted = (~halted) & (certainties[:, 1, t] >= threshold)
            if self.stable_ticks > 1:
                newly_halted = newly_halted & (stable_count >= self.stable_ticks)
            newly_halted = newly_halted & (t >= self.min_steps)
            
            halt_tick[newly_halted] = t
            halted |= newly_halted
            
            if halted.all():
                break
        
        final_predictions = predictions[torch.arange(B), :, halt_tick.clamp(max=predictions.size(-1) - 1)]
        
        return final_predictions, halt_tick, predictions, certainties
