import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


def halting_loss(halt_probs, lambda_halt=0.01):
    """
    halt_probs: [B, T] — probability of halting at each tick
    Encourages halting early while staying differentiable.
    """
    T = halt_probs.size(1)
    tick_indices = torch.arange(1, T+1, device=halt_probs.device).float()
    expected_ticks = (halt_probs * tick_indices).sum(1)
    return lambda_halt * expected_ticks.mean()


def certainty_efficiency_loss(certainties, alpha=0.05):
    """
    Penalises late-peaking certainty. Certainties: [B, 2, T].
    Returns a scalar that is higher when certainty peaks at later ticks.
    """
    C = certainties[:, 1, :]
    T = C.size(1)
    tick_weights = torch.arange(1, T+1, dtype=C.dtype, device=C.device)
    C_softmax = C.softmax(-1)
    mean_tick = (C_softmax * tick_weights).sum(-1)
    return alpha * mean_tick.mean()


def adaptive_halt_threshold(certainties, threshold=0.8, min_stable_ticks=3):
    """
    Returns the earliest tick where certainty was >= threshold
    for min_stable_ticks consecutive ticks. Falls back to argmax(certainty).
    """
    C = certainties[:, 1, :]
    B, T = C.shape
    halt_tick = C.argmax(-1).clone()

    for b in range(B):
        for t in range(T - min_stable_ticks + 1):
            window = C[b, t:t + min_stable_ticks]
            if (window >= threshold).all():
                halt_tick[b] = t
                break
    return halt_tick


class TickCurriculum:
    def __init__(self, T_min=5, T_max=50, warmup_iters=50000):
        self.T_min = T_min
        self.T_max = T_max
        self.warmup_iters = warmup_iters

    def current_T(self, iteration):
        frac = min(iteration / self.warmup_iters, 1.0)
        return int(self.T_min + (self.T_max - self.T_min) * 
                   (1 - math.cos(frac * math.pi)) / 2)


def compute_ctc_loss(predictions, targets, blank_label=0):
    """
    Computes the Connectionist Temporal Classification (CTC) loss.

    Args:
        predictions: A tensor of shape [B, C, L] representing the logits of the 
                     predicted sequences.  B is the batch size, C is the number
                     of classes (including the blank label), and L is the sequence
                     length of the predictions.
        targets: A tensor of shape [B, T] representing the target sequences.
                 B is the batch size and T is the target sequence length.
                 Note that T can vary within the batch.
        blank_label: The index of the blank label.  Defaults to 0.

    Returns:
        The CTC loss (a scalar tensor).
    """

    batch_size, num_classes, prediction_length = predictions.shape
    _, target_length = targets.shape

    # 1. Log softmax on predictions:  Crucially, CTC loss requires log probabilities.
    log_probs = F.log_softmax(predictions, dim=1)  # Shape: [B, C, L]

    # 2.  Prepare inputs for torch.nn.CTCLoss:
    #    a.  Convert log_probs to shape (L, B, C):  CTCLoss expects time first.
    log_probs = log_probs.permute(2, 0, 1)  # Shape: [L, B, C]

    #    b.  Get lengths of the predicted sequences (all L in this case).
    input_lengths = torch.full(size=(batch_size,), fill_value=prediction_length, dtype=torch.long)

    #    c.  Get lengths of the target sequences.
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long) # Handle variable target lengths

    # 3. Create the CTCLoss criterion.  `blank=blank_label` is essential!
    ctc_loss = torch.nn.CTCLoss(blank=blank_label, reduction='mean') # 'mean' for averaging over the batch

    # 4. Calculate the loss.  `targets` needs to be a concatenated tensor.
    #    We handle padding by only passing the valid lengths to CTCLoss.
    concatenated_targets = torch.cat(list(targets)) # Concatenate targets

    loss = ctc_loss(log_probs, concatenated_targets, input_lengths, target_lengths)

    return loss

def sort_loss(predictions, targets):
    """
    The sort task was used partly to show that ctc loss can work.
    """
    loss = compute_ctc_loss(predictions, targets, blank_label=predictions.shape[1]-1)
    return loss

def image_classification_loss(predictions, certainties, targets, use_most_certain=True):
    """
    Computes the maze loss with auto-extending cirriculum.

    Predictions are of shape: (B, class, internal_ticks),
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B]

    use_most_certain will select either the most certain point or the final point. 
    """
    targets_expanded = torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1)
    # Losses are of shape [B, internal_ticks]
    losses = nn.CrossEntropyLoss(reduction='none')(predictions, targets_expanded)
        
    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:  # Revert to final loss if set
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected)/2
    return loss, loss_index_2

def maze_loss(predictions, certainties, targets, cirriculum_lookahead=5, use_most_certain=True):
    """
    Computes the maze loss with auto-extending cirriculum.

    Predictions are of shape: (B, route_length, class, internal_ticks),
        where classes are in [0,1,2,3,4] for [Up, Down, Left, Right, Wait]
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B, route_length]

    cirriculum_lookahead: how far to look ahead in the auto-cirriculum

    use_most_certain will select either the most certain point or the final point. For baselines,
        the final point proved the only usable option. 
    
    """
    # Predictions reshaped to: [B*route_length, 5, internal_ticks]
    predictions_reshaped = predictions.flatten(0,1)
    # Targets reshaped to: [B*route_length, internal_ticks]
    targets_reshaped = torch.repeat_interleave(targets.unsqueeze(-1), 
                                               predictions.size(-1), -1).flatten(0,1).long()
    
    # Losses are of shape [B, route_length, internal_ticks]
    losses = nn.CrossEntropyLoss(reduction='none')(predictions_reshaped, targets_reshaped)
    losses = losses.reshape(predictions[:,:,0].shape)
    
    # Below is the code for auto-cirriculum
    # Find where correct, and make sure to always push +5 beyond that
    iscorrects = (predictions.argmax(2) == targets.unsqueeze(-1)).cumsum(1)
    correct_mask = (iscorrects == torch.arange(1, iscorrects.size(1)+1, device=iscorrects.device).reshape(1, -1, 1))
    correct_mask[:,0,:] = 1
    upto_where = correct_mask.cumsum(1).argmax(1).max(-1)[0]+cirriculum_lookahead
    loss_mask = torch.zeros_like(losses)
    for bi in range(predictions.size(0)):
        loss_mask[bi, :upto_where[bi]] = 1

    # Reduce losses along route dimension
    # Will now be of shape [B, internal_ticks]
    losses = (losses * loss_mask).sum(1)/(loss_mask.sum(1))

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1]
    loss_selected = losses[batch_indexer, loss_index_2]

    loss = ((loss_minimum_ce + loss_selected)/2).mean()
    return loss, loss_index_2, upto_where.detach().cpu().numpy()

def parity_loss(predictions, certainties, targets, use_most_certain=True):
    """
    Computes the parity loss.

    Predictions are of shape: (B, parity_sequence_length, class, internal_ticks),
        where classes are in [0,1,2,3,4] for [Up, Down, Left, Right, Wait]
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B, parity_sequence_length]

    use_most_certain will select either the most certain point or the final point. For baselines,
        the final point proved the only usable option. 
    """

    # Losses are of shape [B, parity_sequence_length, internal_ticks]
    losses = nn.CrossEntropyLoss(reduction='none')(predictions.flatten(0,1), 
                                                   torch.repeat_interleave(targets.unsqueeze(-1), 
                                                                           predictions.size(-1), -1).flatten(0,1).long()).reshape(predictions[:,:,0].shape)

    # Average the loss over the parity sequenece dimension
    losses = losses.mean(1)

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected)/2
    return loss, loss_index_2


def qamnist_loss(predictions, certainties, targets, use_most_certain=True):
    """
    Computes the qamnist loss over the last num_answer_steps steps.

    Predictions are of shape: (B, class, internal_ticks),
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B]
    num_answer_steps: number of steps to consider for the loss

    use_most_certain will select either the most certain point or the final point. 
    """

    losses = nn.CrossEntropyLoss(reduction='none')(predictions, 
                                                   torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1))
        
    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected)/2
    return loss, loss_index_2


class PonderLoss(nn.Module):
    """
    PonderNet loss with geometric prior regularization.
    
    From: "PonderNet: Learning to Ponder" (Banino et al., 2021)
    https://arxiv.org/abs/2107.05407
    
    Loss = ReconstructionLoss + beta * KL(p_halt || geometric_prior)
    
    The geometric prior encourages the model to halt after approximately 1/lambda_p steps.
    """
    def __init__(self, lambda_p: float = 0.2, beta: float = 0.01, max_steps: int = 10):
        """
        Args:
            lambda_p: Parameter for geometric prior (expected steps = 1/lambda_p)
            beta: Weight for regularization term
            max_steps: Maximum number of pondering steps (truncation point)
        """
        super().__init__()
        self.beta = beta
        self.lambda_p = lambda_p
        self.max_steps = max_steps
        
        steps = torch.arange(1, max_steps + 1, dtype=torch.float32)
        prior = (1 - lambda_p) ** (steps - 1) * lambda_p
        self.prior = prior / prior.sum()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, halt_probs: Optional[torch.Tensor] = None):
        """
        Args:
            predictions: Predictions [batch_size, num_classes, max_steps]
            targets: Targets [batch_size]
            halt_probs: Halt probabilities [batch_size, max_steps] - if None, uses uniform
            
        Returns:
            total_loss, recon_loss, reg_loss
        """
        if predictions.dim() == 3:
            predictions = predictions.permute(0, 2, 1)  # [B, T, C]
        
        B, T, C = predictions.shape
        
        if halt_probs is None:
            p = torch.ones(T, B, device=predictions.device) / T
        else:
            p = halt_probs.T  # [T, B]
        
        recon_loss = torch.tensor(0., device=predictions.device)
        for t in range(T):
            step_loss = F.cross_entropy(predictions[:, t, :], targets, reduction='mean')
            recon_loss = recon_loss + (p[t] * step_loss)
        
        prior = self.prior[:T].unsqueeze(1).expand(-1, B)
        
        p_normalized = p / (p.sum(dim=0, keepdim=True) + 1e-8)
        reg_loss = (p_normalized * torch.log(p_normalized / (prior + 1e-10) + 1e-10)).sum(dim=0).mean()
        
        total_loss = recon_loss + self.beta * reg_loss
        
        return total_loss, recon_loss.detach(), reg_loss.detach()


class LoopLMLoss(nn.Module):
    """
    LoopLM loss with uniform prior (entropy regularization).
    
    From: "Scaling Latent Reasoning via Looped Language Models" (Li et al., 2025)
    https://arxiv.org/abs/2502.10048
    
    Loss = E_p[L_step] + beta * KL(p || uniform) 
         = E_p[L_step] + beta * (log T_max - H(p))
    
    The uniform prior encourages the model to distribute reasoning steps evenly.
    """
    def __init__(self, beta: float = 0.01, max_steps: int = 4):
        """
        Args:
            beta: Weight for entropy regularization
            max_steps: Maximum number of recurrent steps (T_max)
        """
        super().__init__()
        self.beta = beta
        self.max_steps = max_steps
        self.log_uniform = torch.log(torch.tensor(max_steps, dtype=torch.float32))
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, halt_probs: Optional[torch.Tensor] = None):
        """
        Args:
            predictions: Predictions [batch_size, num_classes, max_steps]
            targets: Targets [batch_size]
            halt_probs: Halt probabilities [batch_size, max_steps] - if None, uses uniform
            
        Returns:
            total_loss, expected_loss, entropy_bonus
        """
        if predictions.dim() == 3:
            predictions = predictions.permute(0, 2, 1)  # [B, T, C]
        
        B, T, C = predictions.shape
        
        if halt_probs is None:
            p = torch.ones(B, T, device=predictions.device) / T
        else:
            p = halt_probs  # [B, T]
        
        step_losses = torch.zeros(B, T, device=predictions.device)
        for t in range(T):
            step_losses[:, t] = F.cross_entropy(predictions[:, t, :], targets, reduction='none')
        
        p_normalized = p / (p.sum(dim=1, keepdim=True) + 1e-10)
        
        expected_loss = (p_normalized * step_losses).sum(dim=1)
        
        entropy = -(p_normalized * torch.log(p_normalized + 1e-10)).sum(dim=1)
        kl_uniform = self.log_uniform - entropy
        
        total = expected_loss + self.beta * kl_uniform
        
        return total.mean(), expected_loss.mean(), entropy.mean()


class CTMGatedLoss(nn.Module):
    """
    Loss function for CTM-Gated model.
    
    Supports multiple loss types:
    - 'standard': Standard CTM loss using minimum CE + most certain CE
    - 'ponder': PonderNet-style loss with geometric prior (Banino et al., 2021)
    - 'loop': LoopLM-style loss with uniform prior (Li et al., 2025)
    """
    
    def __init__(self, loss_type='standard', lambda_p=0.2, beta=0.01, use_most_certain=True, max_steps=10):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_p = lambda_p
        self.beta = beta
        self.use_most_certain = use_most_certain
        self.max_steps = max_steps
        
    def forward(self, predictions, certainties, targets, exit_steps=None, halt_probs=None):
        """
        Args:
            predictions: [B, C, T] - model predictions at each step
            certainties: [B, 2, T] - certainty values (entropy, 1-entropy)
            targets: [B] - target class indices
            exit_steps: [B] - step at which each sample exited (for computing halt loss)
            halt_probs: [B, T] - halting probabilities at each step
        """
        if self.loss_type == 'standard':
            return self._standard_loss(predictions, certainties, targets)
        elif self.loss_type == 'ponder':
            return self._ponder_loss(predictions, targets, halt_probs)
        elif self.loss_type == 'loop':
            return self._loop_loss(predictions, targets, halt_probs)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _standard_loss(self, predictions, certainties, targets):
        """Standard CTM loss: use most certain exit (certainty argmax)"""
        B = predictions.size(0)
        T = predictions.size(2)
        
        targets_expanded = torch.repeat_interleave(targets.unsqueeze(-1), T, -1)
        losses = F.cross_entropy(predictions, targets_expanded, reduction='none')
        
        exit_index = certainties[:, 1].argmax(-1)
        
        if not self.use_most_certain:
            exit_index[:] = T - 1
        
        batch_indexer = torch.arange(B, device=predictions.device)
        loss = losses[batch_indexer, exit_index].mean()

        return loss, exit_index
    
    def _ponder_loss(self, predictions, targets, halt_probs):
        """
        PonderNet-style loss with geometric prior.
        
        From: "PonderNet: Learning to Ponder" (Banino et al., 2021)
        
        Uses soft halt probabilities instead of hard one-hot assignment.
        """
        B, C, T = predictions.size()
        
        if halt_probs is None:
            progress = torch.arange(T, device=predictions.device).float() / max(T - 1, 1)
            base_prob = self.lambda_p * torch.ones(T, device=predictions.device)
            base_prob = base_prob.unsqueeze(0).expand(B, -1)
            
            ones = torch.ones(B, 1, device=predictions.device)
            cum_not_halted = torch.cumprod(torch.cat((ones, 1.0 - base_prob[:, :-1]), dim=1), dim=1)
            p_t = base_prob * cum_not_halted
            p_t[:, -1] = 1.0 - p_t[:, :-1].sum(dim=1)
        else:
            p_t = halt_probs
        
        targets_expanded = targets.unsqueeze(1).expand(-1, T)
        ce_per_step = F.cross_entropy(predictions, targets_expanded, reduction='none')
        
        ones = torch.ones(B, 1, device=predictions.device)
        cum_not_halted = torch.cumprod(torch.cat((ones, 1.0 - p_t[:, :-1]), dim=1), dim=1)
        weighted_loss = p_t * cum_not_halted * ce_per_step
        loss_rec = weighted_loss.sum(dim=1).mean()
        
        steps = torch.arange(1, T + 1, device=predictions.device).float()
        geo_prior = torch.pow(1 - self.lambda_p, steps - 1) * self.lambda_p
        geo_prior = geo_prior / geo_prior.sum()
        
        p_normalized = p_t / (p_t.sum(dim=1, keepdim=True) + 1e-8)
        geo_prior_expanded = geo_prior.unsqueeze(0).expand(B, -1)
        kl_div = (p_normalized * (torch.log(p_normalized + 1e-8) - torch.log(geo_prior_expanded + 1e-8))).sum(dim=1).mean()
        
        total_loss = loss_rec + self.beta * kl_div
        
        exit_steps = p_t.argmax(dim=1) if halt_probs is not None else torch.full((B,), T - 1, dtype=torch.long, device=predictions.device)
        
        return total_loss, exit_steps
    
    def _loop_loss(self, predictions, targets, halt_probs):
        """
        LoopLM-style loss with uniform prior.
        
        From: "Scaling Latent Reasoning via Looped Language Models" (Li et al., 2025)
        
        Uses soft halt probabilities instead of hard one-hot assignment.
        """
        B, C, T = predictions.size()
        
        if halt_probs is None:
            p_t = torch.ones(B, T, device=predictions.device) / T
        else:
            p_t = halt_probs
        
        targets_expanded = targets.unsqueeze(1).expand(-1, T)
        ce_per_step = F.cross_entropy(predictions, targets_expanded, reduction='none')
        
        p_normalized = p_t / (p_t.sum(dim=1, keepdim=True) + 1e-8)
        loss_rec = (p_normalized * ce_per_step).sum(dim=1).mean()
        
        entropy = -(p_normalized * torch.log(p_normalized + 1e-8)).sum(dim=1).mean()
        
        log_T = torch.log(torch.tensor(T, device=predictions.device, dtype=torch.float32))
        kl_uniform = log_T - entropy
        
        total_loss = loss_rec + self.beta * kl_uniform
        
        exit_steps = p_t.argmax(dim=1) if halt_probs is not None else torch.full((B,), T - 1, dtype=torch.long, device=predictions.device)
        
        return total_loss, exit_steps