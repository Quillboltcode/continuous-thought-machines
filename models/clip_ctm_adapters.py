import torch
import torch.nn as nn

from models.ctm import ContinuousThoughtMachine


class BottleneckAdapter(nn.Module):
    """Small bottleneck MLP adapter with residual connection."""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.fc(x)


class CLIPCTMAdapter(ContinuousThoughtMachine):
    """
    CLIP-CTM with configurable adapter insertion points (A-F variants).

    Supports inserting learnable bottleneck adapters at six different points
    in the CTM architecture while keeping CLIP completely frozen.

    Variants:
        CLIPCTMAdapterHookA (BackboneAdapter): On `kv` features after projection.
        CLIPCTMAdapterHookB (AttentionAdapter): On `attn_out` after attention computation.
        CLIPCTMAdapterHookC (SynapseAdapter): Inside SynapseUNET at the bottleneck.
        CLIPCTMAdapterHookD (StateAdapter): On `activated_state` (highest leverage per parameter).
        CLIPCTMAdapterHookE (ActionAdapter): On `synchronisation_action` representation.
        CLIPCTMAdapterHookF (OutputAdapter): On `synchronisation_out` representation (task head).

    Args:
        clip_model_name: HuggingFace CLIP model identifier.
        use_text_features: Concatenate frozen text-token embeddings.
        text_prompts: List of class-descriptor strings.
        use_intermediate_layer: Add learnable per-patch positional embedding.
        clip_dtype: Precision for CLIP parameters.
        adapter_hooks: List of enabled adapter hooks (e.g., ['A'], ['A', 'D'], ['A','B','D','E','F']).
        adapter_reduction: Bottleneck reduction ratio for adapters.
        **ctm_kwargs: Forwarded to ContinuousThoughtMachine.
    """

    def __init__(self,
                 clip_model_name="openai/clip-vit-base-patch32",
                 use_text_features=False,
                 text_prompts=None,
                 use_intermediate_layer=False,
                 clip_dtype="float16",
                 adapter_hooks=None,
                 adapter_reduction=4,
                 **ctm_kwargs):
        from transformers import CLIPModel, CLIPProcessor

        self.clip_model_name = clip_model_name
        self.use_text_features = use_text_features
        self.use_intermediate_layer = use_intermediate_layer
        self.adapter_hooks = adapter_hooks if adapter_hooks else ['A']
        self.adapter_reduction = adapter_reduction

        ctm_kwargs['backbone_type'] = 'none'
        ctm_kwargs['positional_embedding_type'] = 'none'

        d_input = ctm_kwargs['d_input']
        d_model = ctm_kwargs['d_model']

        super().__init__(**ctm_kwargs)

        if isinstance(clip_dtype, str):
            _dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }
            clip_dtype = _dtype_map.get(clip_dtype, torch.float32)

        self.clip_model = CLIPModel.from_pretrained(clip_model_name, torch_dtype=clip_dtype)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        vm = self.clip_model.vision_model
        vc = self.clip_model.config.vision_config
        self.n_patches = (vc.image_size // vc.patch_size) ** 2
        self.clip_dim = vc.hidden_size
        self.clip_dtype = clip_dtype

        self.clip_processor = None
        self.text_proj = None
        if use_text_features:
            if text_prompts is None:
                raise ValueError("text_prompts must be provided when use_text_features=True")

            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            text_tokens = self._extract_text_tokens(text_prompts)
            self.register_buffer('text_tokens', text_tokens)

            tc = self.clip_model.config.text_config
            if tc.hidden_size != self.clip_dim:
                self.text_proj = nn.Linear(tc.hidden_size, self.clip_dim, bias=False)

        self.patch_pe = None
        if use_intermediate_layer:
            self.patch_pe = nn.Embedding(self.n_patches, d_input)

        self.adapter_A = None
        self.adapter_B = None
        self.adapter_C = None
        self.adapter_D = None
        self.adapter_E = None
        self.adapter_F = None

        if 'A' in self.adapter_hooks:
            self.adapter_A = BottleneckAdapter(d_input, reduction=adapter_reduction)

        if 'B' in self.adapter_hooks:
            self.adapter_B = BottleneckAdapter(d_input, reduction=adapter_reduction)

        if 'C' in self.adapter_hooks:
            self.synapse_bottleneck_adapter = BottleneckAdapter(d_model, reduction=adapter_reduction)

        if 'D' in self.adapter_hooks:
            self.adapter_D = BottleneckAdapter(d_model, reduction=adapter_reduction)

        if 'E' in self.adapter_hooks:
            self.adapter_E = BottleneckAdapter(self.synch_representation_size_action, reduction=adapter_reduction)

        if 'F' in self.adapter_hooks:
            self.adapter_F = BottleneckAdapter(self.synch_representation_size_out, reduction=adapter_reduction)

        with torch.no_grad():
            n_tok = self.n_patches
            if use_text_features and hasattr(self, 'text_tokens'):
                n_tok += self.text_tokens.shape[0]
            self.kv_proj(torch.zeros(1, n_tok, self.clip_dim, dtype=torch.float32))

    def compute_features(self, x):
        """Extract frozen CLIP tokens, project into CTM's d_input space."""
        B = x.size(0)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            tokens = self._extract_vit_patch_tokens(x).float()

            if self.use_text_features:
                text = self.text_tokens.unsqueeze(0).expand(B, -1, -1).float()
                if self.text_proj is not None:
                    text = self.text_proj(text)
                tokens = torch.cat([tokens, text], dim=1)

        kv = self.kv_proj(tokens)

        if 'A' in self.adapter_hooks and self.adapter_A is not None:
            kv = kv + self.adapter_A(kv)

        if self.patch_pe is not None:
            positions = torch.arange(kv.size(1), device=kv.device)
            kv = kv + self.patch_pe(positions).unsqueeze(0)

        return kv

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

        for stepi in range(self.iterations):
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')

            if 'E' in self.adapter_hooks and self.adapter_E is not None:
                synchronisation_action = synchronisation_action + self.adapter_E(synchronisation_action)

            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)

            if 'B' in self.adapter_hooks and self.adapter_B is not None:
                attn_out = attn_out + self.adapter_B(attn_out)

            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

            if 'C' in self.adapter_hooks and hasattr(self, 'synapse_bottleneck_adapter'):
                state = self.synapses(pre_synapse_input)
                bottleneck = state
                adapted_bottleneck = self.synapse_bottleneck_adapter(bottleneck)
                state = adapted_bottleneck
            else:
                state = self.synapses(pre_synapse_input)

            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            activated_state = self.trace_processor(state_trace)

            if 'D' in self.adapter_hooks and self.adapter_D is not None:
                activated_state = activated_state + self.adapter_D(activated_state)

            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            if 'F' in self.adapter_hooks and self.adapter_F is not None:
                synchronisation_out = synchronisation_out + self.adapter_F(synchronisation_out)

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

    def _extract_vit_patch_tokens(self, x):
        import torch.nn.functional as F

        vm = self.clip_model.vision_model
        x = x.to(dtype=vm.embeddings.patch_embedding.weight.dtype)

        target = vm.embeddings.image_size
        if x.shape[-1] != target or x.shape[-2] != target:
            x = F.interpolate(x, size=(target, target), mode='bilinear', align_corners=False)

        hidden_states = vm.embeddings(x)
        hidden_states = vm.pre_layrnorm(hidden_states)

        hidden_states = vm.encoder(inputs_embeds=hidden_states).last_hidden_state
        hidden_states = vm.post_layernorm(hidden_states)

        return hidden_states[:, 1:, :]

    def _extract_text_tokens(self, text_prompts):
        inputs = self.clip_processor(text=text_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(next(self.clip_model.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model.text_model(**inputs)
            hidden = outputs.last_hidden_state

        return hidden.reshape(-1, hidden.shape[-1])


class CLIPCTMAdapterHookA(CLIPCTMAdapter):
    """CLIP-CTM with BackboneAdapter on kv features after projection."""
    def __init__(self, adapter_reduction=4, **ctm_kwargs):
        adapter_hooks = ['A']
        super().__init__(adapter_hooks=adapter_hooks, adapter_reduction=adapter_reduction, **ctm_kwargs)


class CLIPCTMAdapterHookB(CLIPCTMAdapter):
    """CLIP-CTM with AttentionAdapter on attn_out after attention computation."""
    def __init__(self, adapter_reduction=4, **ctm_kwargs):
        adapter_hooks = ['B']
        super().__init__(adapter_hooks=adapter_hooks, adapter_reduction=adapter_reduction, **ctm_kwargs)


class CLIPCTMAdapterHookC(CLIPCTMAdapter):
    """CLIP-CTM with SynapseAdapter inside SynapseUNET at the bottleneck."""
    def __init__(self, adapter_reduction=4, **ctm_kwargs):
        adapter_hooks = ['C']
        super().__init__(adapter_hooks=adapter_hooks, adapter_reduction=adapter_reduction, **ctm_kwargs)


class CLIPCTMAdapterHookD(CLIPCTMAdapter):
    """CLIP-CTM with StateAdapter on activated_state (highest leverage per parameter)."""
    def __init__(self, adapter_reduction=4, **ctm_kwargs):
        adapter_hooks = ['D']
        super().__init__(adapter_hooks=adapter_hooks, adapter_reduction=adapter_reduction, **ctm_kwargs)


class CLIPCTMAdapterHookE(CLIPCTMAdapter):
    """CLIP-CTM with ActionAdapter on synchronisation_action representation."""
    def __init__(self, adapter_reduction=4, **ctm_kwargs):
        adapter_hooks = ['E']
        super().__init__(adapter_hooks=adapter_hooks, adapter_reduction=adapter_reduction, **ctm_kwargs)


class CLIPCTMAdapterHookF(CLIPCTMAdapter):
    """CLIP-CTM with OutputAdapter on synchronisation_out representation (task head)."""
    def __init__(self, adapter_reduction=4, **ctm_kwargs):
        adapter_hooks = ['F']
        super().__init__(adapter_hooks=adapter_hooks, adapter_reduction=adapter_reduction, **ctm_kwargs)