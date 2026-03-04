import torch
import pytest
import numpy as np
from models.ctm_gated import CTMGated, CTMGatedLoss


@pytest.fixture
def base_gated_params():
    """Base parameters for CTMGated model."""
    return dict(
        iterations=50,
        d_model=64,
        d_input=16,
        heads=2,
        n_synch_out=8,
        n_synch_action=8,
        synapse_depth=1,
        memory_length=5,
        deep_nlms=True,
        memory_hidden_dims=8,
        do_layernorm_nlm=False,
        backbone_type="parity_backbone",
        pretrained_backbone=False,
        positional_embedding_type="custom-rotational-1d",
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=0.0,
        neuron_select_type="first-last",
        n_random_pairing_self=0,
        grayscale=False,
    )


@pytest.fixture
def sample_input(device):
    """Create sample input for testing."""
    batch_size = 8
    parity_length = 64
    return torch.randint(0, 2, (batch_size, parity_length), dtype=torch.float32, device=device) * 2 - 1


@pytest.fixture
def sample_targets(device):
    """Create sample targets for testing."""
    batch_size = 8
    return torch.randint(0, 10, (batch_size,), dtype=torch.long, device=device)


class TestEarlyExitFunctionality:
    """Test that early exit functionality works correctly."""

    def test_early_exit_certainty_strategy(self, base_gated_params, sample_input, device):
        """Test that certainty-based early exit can exit before max iterations."""
        model = CTMGated(
            **base_gated_params,
            exit_strategy='certainty',
            exit_threshold=0.7,
            min_steps=5,
        ).to(device)
        model.eval()

        with torch.no_grad():
            predictions, certainties, synch_out, exit_steps = model(sample_input, use_early_exit=True)

        max_iterations = base_gated_params['iterations']
        
        # Check that exit_steps are within valid range
        assert torch.all(exit_steps >= 0), "Exit steps should be non-negative"
        assert torch.all(exit_steps < max_iterations), f"Exit steps should be less than {max_iterations}"
        
        # The key test: some samples should exit early (before max_iterations - 1)
        early_exit_count = (exit_steps < max_iterations - 1).sum().item()
        
        # Print debug info
        print(f"\nCertainty strategy results:")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Exit steps: {exit_steps.tolist()}")
        print(f"  Early exits (< {max_iterations - 1}): {early_exit_count}/{len(exit_steps)}")
        print(f"  Mean exit step: {exit_steps.float().mean().item():.2f}")
        
        # Assertions to verify early exit is working
        assert early_exit_count > 0, (
            f"Expected at least some samples to exit early, "
            f"but all exited at step {max_iterations - 1}"
        )

    def test_early_exit_ponder_strategy(self, base_gated_params, sample_input, device):
        """Test that ponder-based early exit can exit before max iterations."""
        model = CTMGated(
            **base_gated_params,
            exit_strategy='ponder',
            lambda_p=0.1,
            beta=0.01,
            min_steps=3,
        ).to(device)
        model.eval()

        with torch.no_grad():
            predictions, certainties, synch_out, exit_steps = model(sample_input, use_early_exit=True)

        max_iterations = base_gated_params['iterations']
        
        # Check that exit_steps are within valid range
        assert torch.all(exit_steps >= 0), "Exit steps should be non-negative"
        assert torch.all(exit_steps < max_iterations), f"Exit steps should be less than {max_iterations}"
        
        # Count early exits
        early_exit_count = (exit_steps < max_iterations - 1).sum().item()
        
        print(f"\nPonder strategy results:")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Exit steps: {exit_steps.tolist()}")
        print(f"  Early exits (< {max_iterations - 1}): {early_exit_count}/{len(exit_steps)}")
        print(f"  Mean exit step: {exit_steps.float().mean().item():.2f}")
        
        assert early_exit_count > 0, (
            f"Expected at least some samples to exit early with ponder strategy"
        )

    def test_early_exit_normal_strategy(self, base_gated_params, sample_input, device):
        """Test that normal distribution-based early exit can exit before max iterations."""
        model = CTMGated(
            **base_gated_params,
            exit_strategy='normal',
            min_steps=1,
        ).to(device)
        model.eval()

        with torch.no_grad():
            predictions, certainties, synch_out, exit_steps = model(sample_input, use_early_exit=True)

        max_iterations = base_gated_params['iterations']
        
        # Check that exit_steps are within valid range
        assert torch.all(exit_steps >= 0), "Exit steps should be non-negative"
        assert torch.all(exit_steps < max_iterations), f"Exit steps should be less than {max_iterations}"
        
        # Count early exits
        early_exit_count = (exit_steps < max_iterations - 1).sum().item()
        
        print(f"\nNormal strategy results:")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Exit steps: {exit_steps.tolist()}")
        print(f"  Early exits (< {max_iterations - 1}): {early_exit_count}/{len(exit_steps)}")
        print(f"  Mean exit step: {exit_steps.float().mean().item():.2f}")
        
        assert early_exit_count > 0, (
            f"Expected at least some samples to exit early with normal strategy"
        )

    def test_early_exit_learned_strategy(self, base_gated_params, sample_input, device):
        """Test that learned gate early exit can exit before max iterations."""
        model = CTMGated(
            **base_gated_params,
            exit_strategy='learned',
            use_halt_logits=True,
            min_steps=5,
        ).to(device)
        model.eval()

        with torch.no_grad():
            predictions, certainties, synch_out, exit_steps = model(sample_input, use_early_exit=True)

        max_iterations = base_gated_params['iterations']
        
        # Check that exit_steps are within valid range
        assert torch.all(exit_steps >= 0), "Exit steps should be non-negative"
        assert torch.all(exit_steps < max_iterations), f"Exit steps should be less than {max_iterations}"
        
        # Count early exits
        early_exit_count = (exit_steps < max_iterations - 1).sum().item()
        
        print(f"\nLearned strategy results:")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Exit steps: {exit_steps.tolist()}")
        print(f"  Early exits (< {max_iterations - 1}): {early_exit_count}/{len(exit_steps)}")
        print(f"  Mean exit step: {exit_steps.float().mean().item():.2f}")
        
        assert early_exit_count > 0, (
            f"Expected at least some samples to exit early with learned strategy"
        )

    def test_no_early_exit_strategy(self, base_gated_params, sample_input, device):
        """Test that 'none' exit strategy always runs to max iterations."""
        model = CTMGated(
            **base_gated_params,
            exit_strategy='none',
        ).to(device)
        model.eval()

        with torch.no_grad():
            predictions, certainties, synch_out, exit_steps = model(sample_input, use_early_exit=True)

        max_iterations = base_gated_params['iterations']
        
        # All samples should exit at max_iterations - 1
        expected_exit = torch.full_like(exit_steps, max_iterations - 1)
        assert torch.all(exit_steps == expected_exit), (
            f"With exit_strategy='none', all samples should exit at step {max_iterations - 1}, "
            f"got {exit_steps.tolist()}"
        )

    def test_early_exit_disabled(self, base_gated_params, sample_input, device):
        """Test that use_early_exit=False prevents early exit."""
        model = CTMGated(
            **base_gated_params,
            exit_strategy='certainty',
            exit_threshold=0.5,  # Low threshold would normally cause early exit
            min_steps=1,
        ).to(device)
        model.eval()

        with torch.no_grad():
            predictions, certainties, synch_out, exit_steps = model(sample_input, use_early_exit=False)

        max_iterations = base_gated_params['iterations']
        
        # All samples should exit at max_iterations - 1 when use_early_exit=False
        expected_exit = torch.full_like(exit_steps, max_iterations - 1)
        assert torch.all(exit_steps == expected_exit), (
            f"With use_early_exit=False, all samples should exit at step {max_iterations - 1}"
        )

    def test_min_steps_respected(self, base_gated_params, sample_input, device):
        """Test that min_steps is respected - no exits before min_steps."""
        min_steps = 10
        model = CTMGated(
            **base_gated_params,
            exit_strategy='certainty',
            exit_threshold=0.99,  # Very high threshold to encourage early exit
            min_steps=min_steps,
        ).to(device)
        model.eval()

        with torch.no_grad():
            predictions, certainties, synch_out, exit_steps = model(sample_input, use_early_exit=True)

        # No sample should exit before min_steps
        assert torch.all(exit_steps >= min_steps), (
            f"Some samples exited before min_steps={min_steps}: {exit_steps.tolist()}"
        )


class TestEarlyExitShapeAndConsistency:
    """Test output shapes and consistency of early exit."""

    def test_exit_steps_shape(self, base_gated_params, sample_input, device):
        """Test that exit_steps has correct shape."""
        model = CTMGated(
            **base_gated_params,
            exit_strategy='certainty',
        ).to(device)

        predictions, certainties, synch_out, exit_steps = model(sample_input)

        batch_size = sample_input.size(0)
        assert exit_steps.shape == (batch_size,), (
            f"exit_steps should have shape ({batch_size},), got {exit_steps.shape}"
        )

    def test_predictions_shape_with_early_exit(self, base_gated_params, sample_input, device):
        """Test that predictions have correct shape even with early exit."""
        model = CTMGated(
            **base_gated_params,
            exit_strategy='certainty',
        ).to(device)

        predictions, certainties, synch_out, exit_steps = model(sample_input)

        batch_size = sample_input.size(0)
        out_dims = base_gated_params['out_dims']
        iterations = base_gated_params['iterations']

        assert predictions.shape == (batch_size, out_dims, iterations), (
            f"predictions should have shape ({batch_size}, {out_dims}, {iterations}), "
            f"got {predictions.shape}"
        )

    def test_exit_steps_different_per_sample(self, base_gated_params, sample_input, device):
        """Test that different samples can exit at different steps."""
        model = CTMGated(
            **base_gated_params,
            exit_strategy='ponder',
            lambda_p=0.2,
            min_steps=1,
        ).to(device)
        model.eval()

        # Run multiple times to get variety in exit steps
        all_same_count = 0
        for _ in range(5):
            with torch.no_grad():
                predictions, certainties, synch_out, exit_steps = model(sample_input, use_early_exit=True)
            
            # Check if all samples in this batch exited at the same step
            if torch.all(exit_steps == exit_steps[0]):
                all_same_count += 1

        # It's possible but unlikely that all samples exit at the same step every time
        # We allow some tolerance here
        assert all_same_count < 5, (
            "All samples consistently exit at the same step - "
            "this may indicate an issue with per-sample halting"
        )


class TestEarlyExitStatistics:
    """Collect statistics about early exit behavior."""

    @pytest.mark.parametrize("exit_strategy", ['certainty', 'ponder', 'normal', 'learned'])
    def test_early_exit_statistics(self, base_gated_params, sample_input, device, exit_strategy):
        """Collect and report statistics on early exit behavior."""
        
        kwargs = {'exit_strategy': exit_strategy}
        if exit_strategy == 'certainty':
            kwargs['exit_threshold'] = 0.8
            kwargs['min_steps'] = 5
        elif exit_strategy == 'ponder':
            kwargs['lambda_p'] = 0.15
            kwargs['min_steps'] = 3
        elif exit_strategy == 'learned':
            kwargs['use_halt_logits'] = True
            kwargs['min_steps'] = 5
        else:
            kwargs['min_steps'] = 1

        model = CTMGated(**base_gated_params, **kwargs).to(device)
        model.eval()

        # Run multiple times to collect statistics
        exit_step_counts = []
        for _ in range(10):
            with torch.no_grad():
                _, _, _, exit_steps = model(sample_input, use_early_exit=True)
            exit_step_counts.extend(exit_steps.cpu().tolist())

        max_iterations = base_gated_params['iterations']
        
        # Calculate statistics
        exit_steps_tensor = torch.tensor(exit_step_counts, dtype=torch.float32)
        mean_exit = exit_steps_tensor.mean().item()
        std_exit = exit_steps_tensor.std().item()
        min_exit = exit_steps_tensor.min().item()
        max_exit = exit_steps_tensor.max().item()
        early_exit_rate = (exit_steps_tensor < max_iterations - 1).float().mean().item()

        print(f"\n{'='*50}")
        print(f"Early Exit Statistics for '{exit_strategy}' strategy:")
        print(f"{'='*50}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Mean exit step: {mean_exit:.2f}")
        print(f"  Std exit step: {std_exit:.2f}")
        print(f"  Min exit step: {min_exit:.0f}")
        print(f"  Max exit step: {max_exit:.0f}")
        print(f"  Early exit rate: {early_exit_rate*100:.1f}%")
        print(f"{'='*50}")

        # Assert that early exit is actually happening
        assert early_exit_rate > 0, (
            f"No early exits detected for {exit_strategy} strategy - "
            f"all samples ran to max iterations"
        )

        # Mean exit should be less than max_iterations - 1 for effective early exit
        assert mean_exit < max_iterations - 1, (
            f"Mean exit step ({mean_exit:.2f}) is not significantly earlier "
            f"than max_iterations - 1 ({max_iterations - 1})"
        )
