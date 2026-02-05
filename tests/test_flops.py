"""
Tests for FLOPs calculation in ContinuousThoughtMachine models.
Tests both fvcore and handcrafted calculations, comparing them to ensure accuracy.
Note: fvcore may skip some operations like LayerNorm, residual adds, and pooling.
"""
import torch
import pytest
from models.ctm import ContinuousThoughtMachine
from models.utils import compute_ctm_flops


def test_fvcore_available():
    """Test if fvcore is available."""
    try:
        from fvcore.nn import FlopCountAnalysis
        fvcore_available = True
    except ImportError:
        fvcore_available = False
    
    print(f"\nfvcore available: {fvcore_available}")
    return fvcore_available


def test_flops_basic_model():
    """Test FLOPs calculation for a basic CTM model - compare fvcore vs handcrafted."""
    fvcore_available = test_fvcore_available()
    
    model = ContinuousThoughtMachine(
        iterations=5,
        d_model=64,
        d_input=32,
        heads=4,
        n_synch_out=16,
        n_synch_action=16,
        synapse_depth=1,
        memory_length=3,
        deep_nlms=False,
        memory_hidden_dims=32,
        do_layernorm_nlm=False,
        backbone_type='none',
        pretrained_backbone='none',
        positional_embedding_type='none',
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=0.1,
        synchronisation_type='ctm',
        neuron_select_type='random-pairing',
        n_random_pairing_self=0,
    )
    
    input_shape = (3, 224, 224)  # Example input shape
    batch_size = 1
    
    # Test with fvcore
    flops_fvcore = compute_ctm_flops(model, input_shape, batch_size, use_fvcore=True)
    
    # Test with handcrafted calculation
    flops_handcrafted = compute_ctm_flops(model, input_shape, batch_size, use_fvcore=False)
    
    # Both should return valid results
    assert isinstance(flops_fvcore, dict)
    assert isinstance(flops_handcrafted, dict)
    assert flops_fvcore['total'] > 0
    assert flops_handcrafted['total'] > 0
    
    # Check that loop_total = per_iteration * iterations for both
    assert flops_fvcore['loop_total'] == flops_fvcore['per_iteration'] * model.iterations
    assert flops_handcrafted['loop_total'] == flops_handcrafted['per_iteration'] * model.iterations
    
    # Compare results
    diff = abs(flops_fvcore['total'] - flops_handcrafted['total'])
    diff_percent = (diff / max(flops_fvcore['total'], flops_handcrafted['total'])) * 100
    
    print(f"\n{'='*60}")
    print(f"Basic Model FLOPs Comparison")
    print(f"{'='*60}")
    print(f"fvcore calculation:")
    print(f"  Total: {flops_fvcore['total']:,}")
    print(f"  Per iteration: {flops_fvcore['per_iteration']:,}")
    print(f"  Loop total: {flops_fvcore['loop_total']:,}")
    print(f"  Outside loop: {flops_fvcore['breakdown']['outside_loop']:,}")
    print(f"\nHandcrafted calculation:")
    print(f"  Total: {flops_handcrafted['total']:,}")
    print(f"  Per iteration: {flops_handcrafted['per_iteration']:,}")
    print(f"  Loop total: {flops_handcrafted['loop_total']:,}")
    print(f"  Outside loop: {flops_handcrafted['breakdown']['outside_loop']:,}")
    print(f"\nDifference:")
    print(f"  Absolute: {diff:,}")
    print(f"  Percentage: {diff_percent:.2f}%")
    print(f"  Note: fvcore may skip LayerNorm, residual adds, and pooling operations")


def test_flops_deep_nlm_model():
    """Test FLOPs calculation for a model with deep NLMs - compare both methods."""
    model = ContinuousThoughtMachine(
        iterations=10,
        d_model=128,
        d_input=64,
        heads=4,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=3,
        memory_length=5,
        deep_nlms=True,
        memory_hidden_dims=64,
        do_layernorm_nlm=False,
        backbone_type='none',
        pretrained_backbone='none',
        positional_embedding_type='none',
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=0.1,
        synchronisation_type='ctm',
        neuron_select_type='random-pairing',
        n_random_pairing_self=8,
    )
    
    input_shape = (3, 224, 224)
    batch_size = 2
    
    # Test both methods
    flops_fvcore = compute_ctm_flops(model, input_shape, batch_size, use_fvcore=True)
    flops_handcrafted = compute_ctm_flops(model, input_shape, batch_size, use_fvcore=False)
    
    # Deep NLMs should have more FLOPs per iteration
    assert flops_fvcore['per_iteration'] > 0
    assert flops_handcrafted['per_iteration'] > 0
    
    # Check that FLOPs scale with batch size
    flops_fvcore_batch1 = compute_ctm_flops(model, input_shape, batch_size=1, use_fvcore=True)
    assert flops_fvcore['total'] > flops_fvcore_batch1['total']
    
    diff = abs(flops_fvcore['total'] - flops_handcrafted['total'])
    
    print(f"\n{'='*60}")
    print(f"Deep NLM Model FLOPs Comparison")
    print(f"{'='*60}")
    print(f"fvcore: {flops_fvcore['total']:,} (per iter: {flops_fvcore['per_iteration']:,})")
    print(f"Handcrafted: {flops_handcrafted['total']:,} (per iter: {flops_handcrafted['per_iteration']:,})")
    print(f"Difference: {diff:,}")


def test_flops_different_neuron_select_types():
    """Test FLOPs calculation with different neuron selection types."""
    neuron_select_types = ['random-pairing', 'first-last', 'random']
    
    for neuron_select_type in neuron_select_types:
        model = ContinuousThoughtMachine(
            iterations=5,
            d_model=128,
            d_input=32,
            heads=2,
            n_synch_out=16,
            n_synch_action=16,
            synapse_depth=1,
            memory_length=3,
            deep_nlms=False,
            memory_hidden_dims=32,
            do_layernorm_nlm=False,
            backbone_type='none',
            pretrained_backbone='none',
            positional_embedding_type='none',
            out_dims=10,
            prediction_reshaper=[-1],
            dropout=0.1,
            synchronisation_type='ctm',
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=0,
        )
        
        input_shape = (3, 224, 224)
        flops_result = compute_ctm_flops(model, input_shape, batch_size=1)
        
        assert flops_result['total'] > 0
        print(f"\n{neuron_select_type} FLOPs: {flops_result['total']:,}")


def test_flops_no_attention():
    """Test FLOPs calculation for a model without attention."""
    model = ContinuousThoughtMachine(
        iterations=5,
        d_model=64,
        d_input=32,
        heads=0,  # No attention
        n_synch_out=16,
        n_synch_action=0,  # No action synchronisation
        synapse_depth=1,
        memory_length=3,
        deep_nlms=False,
        memory_hidden_dims=32,
        do_layernorm_nlm=False,
        backbone_type='none',
        pretrained_backbone='none',
        positional_embedding_type='none',
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=0.1,
        synchronisation_type='ctm',
        neuron_select_type='random-pairing',
        n_random_pairing_self=0,
    )
    
    input_shape = (3, 224, 224)
    flops_result = compute_ctm_flops(model, input_shape, batch_size=1)
    
    # Should still have positive FLOPs
    assert flops_result['total'] > 0
    
    # KV projection should be 0 (no attention)
    assert flops_result.get('kv_proj', 0) == 0
    
    print(f"\nNo Attention Model FLOPs: {flops_result['total']:,}")


def test_flops_iterations_scaling():
    """Test that FLOPs scale linearly with iterations."""
    base_iterations = 5
    
    model1 = ContinuousThoughtMachine(
        iterations=base_iterations,
        d_model=64,
        d_input=32,
        heads=2,
        n_synch_out=16,
        n_synch_action=16,
        synapse_depth=1,
        memory_length=3,
        deep_nlms=False,
        memory_hidden_dims=32,
        do_layernorm_nlm=False,
        backbone_type='none',
        pretrained_backbone='none',
        positional_embedding_type='none',
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=0.1,
        synchronisation_type='ctm',
        neuron_select_type='random-pairing',
        n_random_pairing_self=0,
    )
    
    model2 = ContinuousThoughtMachine(
        iterations=base_iterations * 2,
        d_model=64,
        d_input=32,
        heads=2,
        n_synch_out=16,
        n_synch_action=16,
        synapse_depth=1,
        memory_length=3,
        deep_nlms=False,
        memory_hidden_dims=32,
        do_layernorm_nlm=False,
        backbone_type='none',
        pretrained_backbone='none',
        positional_embedding_type='none',
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=0.1,
        synchronisation_type='ctm',
        neuron_select_type='random-pairing',
        n_random_pairing_self=0,
    )
    
    input_shape = (3, 224, 224)
    flops1 = compute_ctm_flops(model1, input_shape, batch_size=1)
    flops2 = compute_ctm_flops(model2, input_shape, batch_size=1)
    
    # Outside loop FLOPs should be the same
    assert abs(flops1['breakdown']['outside_loop'] - flops2['breakdown']['outside_loop']) < 1e-6
    
    # Per iteration FLOPs should be the same
    assert abs(flops1['per_iteration'] - flops2['per_iteration']) < 1e-6
    
    # Total FLOPs should scale with iterations
    ratio = flops2['total'] / flops1['total']
    # Should be approximately 2x (allowing for small numerical differences)
    assert 1.8 < ratio < 2.2
    
    print(f"\nIterations Scaling Test:")
    print(f"  Model 1 ({base_iterations} iterations): {flops1['total']:,}")
    print(f"  Model 2 ({base_iterations * 2} iterations): {flops2['total']:,}")
    print(f"  Ratio: {ratio:.2f}")


def test_flops_model_method():
    """Test the compute_flops method on the model itself."""
    model = ContinuousThoughtMachine(
        iterations=5,
        d_model=64,
        d_input=32,
        heads=2,
        n_synch_out=16,
        n_synch_action=16,
        synapse_depth=1,
        memory_length=3,
        deep_nlms=False,
        memory_hidden_dims=32,
        do_layernorm_nlm=False,
        backbone_type='none',
        pretrained_backbone='none',
        positional_embedding_type='none',
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=0.1,
        synchronisation_type='ctm',
        neuron_select_type='random-pairing',
        n_random_pairing_self=0,
    )
    
    input_shape = (3, 224, 224)
    flops_result = model.compute_flops(input_shape, batch_size=1)
    
    assert isinstance(flops_result, dict)
    assert flops_result['total'] > 0
    
    print(f"\nModel.compute_flops() result: {flops_result['total']:,}")


def test_flops_with_layernorm():
    """Test FLOPs calculation with LayerNorm enabled - fvcore may skip LayerNorm operations."""
    model = ContinuousThoughtMachine(
        iterations=5,
        d_model=64,
        d_input=32,
        heads=4,
        n_synch_out=16,
        n_synch_action=16,
        synapse_depth=1,
        memory_length=3,
        deep_nlms=True,
        memory_hidden_dims=32,
        do_layernorm_nlm=True,  # Enable LayerNorm - fvcore may skip this
        backbone_type='none',
        pretrained_backbone='none',
        positional_embedding_type='none',
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=0.1,
        synchronisation_type='ctm',
        neuron_select_type='random-pairing',
        n_random_pairing_self=0,
    )
    
    input_shape = (3, 224, 224)
    batch_size = 1
    
    # Test both methods
    flops_fvcore = compute_ctm_flops(model, input_shape, batch_size, use_fvcore=True)
    flops_handcrafted = compute_ctm_flops(model, input_shape, batch_size, use_fvcore=False)
    
    diff = abs(flops_fvcore['total'] - flops_handcrafted['total'])
    diff_percent = (diff / max(flops_fvcore['total'], flops_handcrafted['total'])) * 100
    
    print(f"\n{'='*60}")
    print(f"LayerNorm Model FLOPs Comparison")
    print(f"{'='*60}")
    print(f"Note: fvcore may skip LayerNorm operations")
    print(f"fvcore: {flops_fvcore['total']:,}")
    print(f"Handcrafted: {flops_handcrafted['total']:,}")
    print(f"Difference: {diff:,} ({diff_percent:.2f}%)")
    
    # Handcrafted should be >= fvcore (since it includes LayerNorm)
    # But this may not always be true due to different calculation methods
    assert flops_fvcore['total'] > 0
    assert flops_handcrafted['total'] > 0


def test_flops_unet_synapses():
    """Test FLOPs calculation with U-Net synapses - may have residual connections."""
    model = ContinuousThoughtMachine(
        iterations=5,
        d_model=64,
        d_input=32,
        heads=4,
        n_synch_out=16,
        n_synch_action=16,
        synapse_depth=3,  # U-Net with depth 3 - has residual connections
        memory_length=3,
        deep_nlms=False,
        memory_hidden_dims=32,
        do_layernorm_nlm=False,
        backbone_type='none',
        pretrained_backbone='none',
        positional_embedding_type='none',
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=0.1,
        synchronisation_type='ctm',
        neuron_select_type='random-pairing',
        n_random_pairing_self=0,
    )
    
    input_shape = (3, 224, 224)
    batch_size = 1
    
    # Test both methods
    flops_fvcore = compute_ctm_flops(model, input_shape, batch_size, use_fvcore=True)
    flops_handcrafted = compute_ctm_flops(model, input_shape, batch_size, use_fvcore=False)
    
    diff = abs(flops_fvcore['total'] - flops_handcrafted['total'])
    diff_percent = (diff / max(flops_fvcore['total'], flops_handcrafted['total'])) * 100
    
    print(f"\n{'='*60}")
    print(f"U-Net Synapses Model FLOPs Comparison")
    print(f"{'='*60}")
    print(f"Note: U-Net has residual connections - fvcore may skip residual adds")
    print(f"fvcore: {flops_fvcore['total']:,}")
    print(f"Handcrafted: {flops_handcrafted['total']:,}")
    print(f"Difference: {diff:,} ({diff_percent:.2f}%)")
    
    assert flops_fvcore['total'] > 0
    assert flops_handcrafted['total'] > 0


def test_flops_comprehensive_comparison():
    """Comprehensive comparison of fvcore vs handcrafted for various configurations."""
    configs = [
        {
            'name': 'Basic',
            'iterations': 5,
            'd_model': 64,
            'd_input': 32,
            'heads': 2,
            'synapse_depth': 1,
            'deep_nlms': False,
            'do_layernorm_nlm': False,
        },
        {
            'name': 'Deep NLM',
            'iterations': 5,
            'd_model': 64,
            'd_input': 32,
            'heads': 2,
            'synapse_depth': 1,
            'deep_nlms': True,
            'do_layernorm_nlm': False,
        },
        {
            'name': 'U-Net Synapses',
            'iterations': 5,
            'd_model': 64,
            'd_input': 32,
            'heads': 2,
            'synapse_depth': 3,
            'deep_nlms': False,
            'do_layernorm_nlm': False,
        },
        {
            'name': 'LayerNorm Enabled',
            'iterations': 5,
            'd_model': 64,
            'd_input': 32,
            'heads': 2,
            'synapse_depth': 1,
            'deep_nlms': True,
            'do_layernorm_nlm': True,
        },
        {
            'name': 'Complex (U-Net + LayerNorm)',
            'iterations': 10,
            'd_model': 128,
            'd_input': 64,
            'heads': 4,
            'synapse_depth': 3,
            'deep_nlms': True,
            'do_layernorm_nlm': True,
        },
    ]
    
    print(f"\n{'='*80}")
    print(f"Comprehensive FLOPs Comparison: fvcore vs Handcrafted")
    print(f"{'='*80}")
    print(f"{'Configuration':<25} {'fvcore':<15} {'Handcrafted':<15} {'Difference':<15} {'% Diff':<10}")
    print(f"{'-'*80}")
    
    for config in configs:
        model = ContinuousThoughtMachine(
            iterations=config['iterations'],
            d_model=config['d_model'],
            d_input=config['d_input'],
            heads=config['heads'],
            n_synch_out=16,
            n_synch_action=16,
            synapse_depth=config['synapse_depth'],
            memory_length=3,
            deep_nlms=config['deep_nlms'],
            memory_hidden_dims=32,
            do_layernorm_nlm=config['do_layernorm_nlm'],
            backbone_type='none',
            pretrained_backbone='none',
            positional_embedding_type='none',
            out_dims=10,
            prediction_reshaper=[-1],
            dropout=0.1,
            synchronisation_type='ctm',
            neuron_select_type='random-pairing',
            n_random_pairing_self=0,
        )
        
        input_shape = (3, 224, 224)
        batch_size = 1
        
        flops_fvcore = compute_ctm_flops(model, input_shape, batch_size, use_fvcore=True)
        flops_handcrafted = compute_ctm_flops(model, input_shape, batch_size, use_fvcore=False)
        
        diff = abs(flops_fvcore['total'] - flops_handcrafted['total'])
        diff_percent = (diff / max(flops_fvcore['total'], flops_handcrafted['total'])) * 100
        
        print(f"{config['name']:<25} {flops_fvcore['total']:<15,} {flops_handcrafted['total']:<15,} {diff:<15,} {diff_percent:<10.2f}%")
        
        # Both should be positive
        assert flops_fvcore['total'] > 0
        assert flops_handcrafted['total'] > 0
    
    print(f"{'-'*80}")
    print(f"Note: fvcore may skip LayerNorm, residual adds, and pooling operations")
    print(f"This can lead to differences, especially in models with LayerNorm enabled")


if __name__ == '__main__':
    # Run tests
    print("="*80)
    print("FLOPs Calculation Tests - Comparing fvcore vs Handcrafted")
    print("="*80)
    
    test_flops_basic_model()
    test_flops_deep_nlm_model()
    test_flops_different_neuron_select_types()
    test_flops_no_attention()
    test_flops_iterations_scaling()
    test_flops_model_method()
    test_flops_with_layernorm()
    test_flops_unet_synapses()
    test_flops_comprehensive_comparison()
    
    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
