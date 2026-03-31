import torch
from models.ctm import ContinuousThoughtMachine
from models.ctm_gated import CTMGated, CTMGatedLoss
from models.ctm_with_innovations import CTMWithInnovations
from models.clip_ctm import CLIPCTM
from models.lstm import LSTMBaseline
from models.ff import FFBaseline


def create_model(args, prediction_reshaper=None):
    """
    Create a model based on arguments.

    Args:
        args: Namespace with model configuration
        prediction_reshaper: Shape for reshaping predictions

    Returns:
        model: Created model
        loss_fn: Loss function (for CTM-Gated), None otherwise
    """
    if prediction_reshaper is None:
        prediction_reshaper = [-1]

    model = None
    loss_fn = None

    if args.model == "ctm":
        model = ContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            backbone_type=args.backbone_type,
            pretrained_backbone=args.pretrained_backbone,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            dropout_nlm=args.dropout_nlm,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
            grayscale=args.grayscale,
        )
    elif args.model == "ctm_gated":
        model = CTMGated(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            backbone_type=args.backbone_type,
            pretrained_backbone=args.pretrained_backbone,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            dropout_nlm=args.dropout_nlm,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
            grayscale=args.grayscale,
            exit_strategy=args.exit_strategy,
            exit_threshold=args.exit_threshold,
            lambda_p=args.lambda_p,
            beta=args.beta,
            min_steps=args.min_steps,
        )
        loss_fn = CTMGatedLoss(
            loss_type=args.loss_type,
            lambda_p=args.lambda_p,
            beta=args.beta,
            use_most_certain=True,
            max_steps=args.iterations
        )
    elif args.model == "ctm_with_innovations":
        model = CTMWithInnovations(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            backbone_type=args.backbone_type,
            pretrained_backbone=args.pretrained_backbone,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            dropout_nlm=args.dropout_nlm,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
            grayscale=args.grayscale,
            use_gsh=args.use_gsh,
            use_hne=args.use_hne,
            use_sanp=args.use_sanp,
            hne_group_configs=args.hne_group_configs,
            sanp_init_top_k=args.sanp_init_top_k,
        )
    elif args.model == "clip_ctm":
        model = CLIPCTM(
            clip_model_name=args.clip_model_name,
            use_text_features=args.use_text_features,
            use_intermediate_layer=args.use_intermediate_layer,
            clip_dtype=args.clip_dtype,
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            dropout_nlm=args.dropout_nlm,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
            pretrained_backbone=args.pretrained_backbone,
            grayscale=args.grayscale,
        )
    elif args.model == "lstm":
        model = LSTMBaseline(
            num_layers=args.num_layers,
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
        )
    elif args.model == "ff":
        model = FFBaseline(
            d_model=args.d_model,
            backbone_type=args.backbone_type,
            out_dims=args.out_dims,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    return model, loss_fn


def get_model_family(model_type):
    """Get the model family for conditional logic."""
    return model_type
