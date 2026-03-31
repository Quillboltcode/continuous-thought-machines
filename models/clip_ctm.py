import torch
import torch.nn as nn

from models.ctm import ContinuousThoughtMachine


class CLIPCTM(ContinuousThoughtMachine):
    """
    CLIP-CTM: CTM reasoning loop as a trainable adapter over frozen CLIP.

    Overrides ``compute_features`` to extract spatial patch tokens from a frozen
    HuggingFace CLIP vision encoder (ViT-based), then projects them through the
    CTM's ``kv_proj`` into its latent reasoning space.  All CTM internals
    (synapses, NLMs, synchronisation, attention) remain fully trainable.

    Requires: ``pip install transformers`` (already in requirements.txt).

    Args:
        clip_model_name: HuggingFace CLIP model identifier
            (``"openai/clip-vit-base-patch32"``,
            ``"openai/clip-vit-base-patch16"``,
            ``"openai/clip-vit-large-patch14"``, etc.).
        use_text_features: Concatenate frozen text-token embeddings onto image
            patch tokens before projection.
        text_prompts: List of class-descriptor strings used when
            *use_text_features* is ``True``.
        use_intermediate_layer: Add a learnable per-patch positional embedding
            *after* ``kv_proj`` (in CTM's ``d_input`` space).
        clip_dtype: Precision for CLIP parameters (``"float32"`` or ``"float16"``).
        **ctm_kwargs: Forwarded to
            :class:`~models.ctm.ContinuousThoughtMachine`.  The constructor
            silently forces ``backbone_type="none"`` and
            ``positional_embedding_type="none"``.
    """

    def __init__(self,
                 clip_model_name="openai/clip-vit-base-patch32",
                 use_text_features=False,
                 text_prompts=None,
                 use_intermediate_layer=False,
                 clip_dtype="float16",
                 **ctm_kwargs):
        from transformers import CLIPModel, CLIPProcessor

        self.clip_model_name = clip_model_name
        self.use_text_features = use_text_features
        self.use_intermediate_layer = use_intermediate_layer

        # ---- Force CLIP-adapter overrides on parent CTM kwargs ----
        ctm_kwargs['backbone_type'] = 'none'
        ctm_kwargs['positional_embedding_type'] = 'none'

        d_input = ctm_kwargs['d_input']

        # ---- Parent constructor (sets up all CTM internals) ----
        super().__init__(**ctm_kwargs)

        # ---- Resolve dtype ----
        if isinstance(clip_dtype, str):
            _dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }
            clip_dtype = _dtype_map.get(clip_dtype, torch.float32)

        # ---- Load frozen CLIP (ViT vision + optional text) ----
        self.clip_model = CLIPModel.from_pretrained(clip_model_name, torch_dtype=clip_dtype)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        vm = self.clip_model.vision_model
        vc = self.clip_model.config.vision_config
        self.n_patches = (vc.image_size // vc.patch_size) ** 2
        self.clip_dim = vc.hidden_size
        self.clip_dtype = clip_dtype

        # ---- Optional: frozen text-token embeddings ----
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

        # ---- Optional: learnable per-patch PE in CTM d_input space ----
        self.patch_pe = None
        if use_intermediate_layer:
            self.patch_pe = nn.Embedding(self.n_patches, d_input)

        # ---- Initialize lazy modules (kv_proj, q_proj, attention) ----
        with torch.no_grad():
            vc = self.clip_model.config.vision_config
            n_tok = (vc.image_size // vc.patch_size) ** 2
            if use_text_features and hasattr(self, 'text_tokens'):
                n_tok += self.text_tokens.shape[0]
            dummy = torch.zeros(1, n_tok, self.clip_dim)
            self.kv_proj(dummy)
            dummy_q = torch.zeros(1, self.d_model)
            self.q_proj(dummy_q)

    # ------------------------------------------------------------------
    # compute_features — the single override that wires CLIP → CTM
    # ------------------------------------------------------------------

    def compute_features(self, x):
        """Extract frozen CLIP tokens, project into CTM's ``d_input`` space."""
        B = x.size(0)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            tokens = self._extract_vit_patch_tokens(x)  # (B, N_patches, clip_dim)

            if self.use_text_features:
                text = self.text_tokens.unsqueeze(0).expand(B, -1, -1)
                if self.text_proj is not None:
                    text = self.text_proj(text.float())
                tokens = torch.cat([tokens, text], dim=1)  # image first, text second

        kv = self.kv_proj(tokens)  # LazyLinear handles CLIP dim → d_input

        if self.patch_pe is not None:
            positions = torch.arange(kv.size(1), device=kv.device)
            kv = kv + self.patch_pe(positions).unsqueeze(0)

        return kv

    # ------------------------------------------------------------------
    # Private extraction helpers
    # ------------------------------------------------------------------

    def _extract_vit_patch_tokens(self, x):
        """
        Walk CLIP ViT: ``patch_embedding → cat CLS + add PE →
        pre_layrnorm → encoder → post_layernorm``, drop CLS token.

        Resizes input to CLIP's native resolution if the spatial dims don't
        match (e.g. CIFAR 32×32 → 224×224).

        Returns:
            Tensor of shape ``(B, N_patches, clip_dim)``.
        """
        import torch.nn.functional as F

        vm = self.clip_model.vision_model
        x = x.to(dtype=vm.embeddings.patch_embedding.weight.dtype)

        # Resize to CLIP's expected resolution if needed
        target = vm.embeddings.image_size
        if x.shape[-1] != target or x.shape[-2] != target:
            x = F.interpolate(x, size=(target, target), mode='bilinear', align_corners=False)

        # Patch embedding + CLS + positional encoding
        hidden_states = vm.embeddings(x)
        hidden_states = vm.pre_layrnorm(hidden_states)

        # Transformer encoder
        hidden_states = vm.encoder(inputs_embeds=hidden_states).last_hidden_state
        hidden_states = vm.post_layernorm(hidden_states)

        # Drop CLS token, keep patch tokens
        return hidden_states[:, 1:, :]

    def _extract_text_tokens(self, text_prompts):
        """
        Encode *text_prompts* through the frozen CLIP text transformer.

        Returns **all** token positions (not just the EOS embedding), reshaped
        to ``(n_prompts * 77, text_hidden_size)``.
        """
        inputs = self.clip_processor(text=text_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(next(self.clip_model.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model.text_model(**inputs)
            hidden = outputs.last_hidden_state  # (n_prompts, seq_len, text_hidden_size)

        return hidden.reshape(-1, hidden.shape[-1])  # (n_prompts * 77, text_hidden_size)
