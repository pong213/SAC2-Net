"""
Optimization Objective Module
============================================================================
Combined loss function for multimodal micro-expression recognition:
    L_total = L_cls + lambda_sasa * L_sasa + lambda_rel * L_rel

Components:
    1. L_cls:  Class-weighted focal loss (handles class imbalance + sample difficulty)
    2. L_sasa: Semantic Anchoring Soft Alignment loss (KL divergence, from SASA module)
    3. L_rel:  Reliability regularization loss (encourages decisive reliability maps)

Usage:
    criterion = CombinedLoss(num_classes=5, class_counts=[32, 99, 45, 38, 33])
    loss, loss_dict = criterion(model_outputs, labels, au_labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


# =============================================================================
# Component 1: Class-Weighted Focal Loss
# =============================================================================
class ClassWeightedFocalLoss(nn.Module):
    """
    Focal loss with per-class frequency weighting.

    Combines two mechanisms:
    - Class weights: inversely proportional to class frequency, addressing
      the imbalanced category distribution in micro-expression datasets.
    - Focal modulation: (1 - p_t)^gamma downweights easy/well-classified
      samples so the model focuses on hard examples.

    The combined effect: rare classes get higher base weight, AND within each
    class, hard samples get further amplified.

    Args:
        num_classes: Number of emotion categories
        class_counts: List of sample counts per class (for computing weights).
                      If None, uniform weights are used.
        gamma: Focal loss focusing parameter (default: 2.0).
               gamma=0 reduces to standard weighted cross-entropy.
               gamma=2 is the standard choice from the original paper.
        label_smoothing: Label smoothing factor (default: 0.1).
                         Prevents overconfident predictions and improves
                         calibration, especially useful on small datasets.
        weight_scheme: How to compute class weights from counts.
                       "inverse_freq": w_c = N_total / (num_classes * N_c)
                       "effective_num": w_c based on effective number of samples
                                        (from Class-Balanced Loss, Cui et al. 2019)
    """

    def __init__(
            self,
            num_classes: int,
            class_counts: Optional[List[int]] = None,
            gamma: float = 2.0,
            label_smoothing: float = 0.1,
            weight_scheme: str = "effective_num",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        # Compute class weights
        if class_counts is not None:
            class_counts = torch.tensor(class_counts, dtype=torch.float32)
            if weight_scheme == "inverse_freq":
                weights = class_counts.sum() / (num_classes * class_counts)
            elif weight_scheme == "effective_num":
                # Effective Number of Samples (Cui et al., CVPR 2019)
                # beta is typically set close to 1, e.g., (N-1)/N
                beta = (class_counts.sum() - 1.0) / class_counts.sum()
                effective_num = 1.0 - beta ** class_counts
                weights = (1.0 - beta) / (effective_num + 1e-8)
            else:
                raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

            # Normalize weights so they sum to num_classes (preserves loss scale)
            weights = weights / weights.sum() * num_classes
            self.register_buffer("class_weights", weights)
        else:
            self.register_buffer("class_weights", torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, num_classes) raw model output (pre-softmax)
            targets: (B,) integer class labels

        Returns:
            loss: scalar focal loss
        """
        # Apply label smoothing to targets
        # Converts hard labels to soft: [0, 0, 1, 0, 0] -> [0.02, 0.02, 0.92, 0.02, 0.02]
        num_classes = self.num_classes
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Compute log-softmax (numerically stable)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
        probs = torch.exp(log_probs)  # (B, C)

        # Focal modulation: (1 - p_t)^gamma per class
        focal_weight = (1.0 - probs) ** self.gamma  # (B, C)

        # Per-sample loss (sum over classes for soft targets)
        loss_per_sample = -(focal_weight * smooth_targets * log_probs).sum(dim=-1)  # (B,)

        # Apply class weights (indexed by true label)
        weights = self.class_weights.to(logits.device)
        sample_weights = weights[targets]  # (B,)

        # Weighted mean
        loss = (loss_per_sample * sample_weights).mean()

        return loss


# =============================================================================
# Component 2: Reliability Regularization Loss
# =============================================================================
class ReliabilityRegLoss(nn.Module):

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, reliability_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            reliability_maps: List of (B, 1, H', W') tensors from CCF blocks

        Returns:
            loss: reliability regularization loss
        """
        total_loss = 0.0
        num_maps = len(reliability_maps)

        if num_maps == 0:
            return torch.tensor(0.0)

        for R in reliability_maps:
            loss = (R[:, :, 1:, :] - R[:, :, :-1, :]).abs().mean() + \
                    (R[:, :, :, 1:] - R[:, :, :, :-1]).abs().mean()
            total_loss = total_loss + loss

        return total_loss / num_maps


# =============================================================================
# Combined Loss Function
# =============================================================================
class CombinedLoss(nn.Module):
    """
    Combined optimization objective for multimodal micro-expression recognition.

    L_total = L_cls + lambda_sasa * L_sasa + lambda_rel * L_rel

    where:
        L_cls:  Class-weighted focal loss on emotion classification
        L_sasa: SASA contrastive alignment loss (computed externally)
        L_rel:  Reliability regularization

    Args:
        num_classes: Number of emotion categories
        class_counts: Sample counts per class for focal loss weighting.
                      Example for CASME II: [32, 99, 45, 38, 33]
        gamma: Focal loss gamma (default: 2.0)
        label_smoothing: Label smoothing factor (default: 0.1)
        lambda_cls: Weight for classification loss (dynamic: 0 -> 2)
        lambda_sasa: Weight for SASA alignment loss (dynamic: 2 -> 0)
        lambda_rel: Weight for reliability regularization loss (default: 0.01)

    Example:
        criterion = CombinedLoss(
            num_classes=5,
            class_counts=[32, 99, 45, 38, 33],
        )

        # In training loop:
        model_outputs = model(mag_images, flow_images, texts=texts)
        sasa_loss, _ = sasa_loss_function(
            model_outputs['f_mag_proj'],
            model_outputs['f_flow_proj'],
            model_outputs['f_text_proj'],
            au_labels,
        )
        total_loss, loss_dict = criterion(
            model_outputs, labels, sasa_loss=sasa_loss
        )
        total_loss.backward()
    """

    def __init__(
            self,
            num_classes: int,
            class_counts: Optional[List[int]] = None,
            gamma: float = 2.0,
            label_smoothing: float = 0.1,
            lambda_cls: float = 1.0,
            lambda_sasa: float = 1.0,
            lambda_rel: float = 0.1,
    ):
        super().__init__()

        self.lambda_cls = lambda_cls
        self.lambda_sasa = lambda_sasa
        self.lambda_rel = lambda_rel

        # Classification loss
        self.cls_loss = ClassWeightedFocalLoss(
            num_classes=num_classes,
            class_counts=class_counts,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )

        # Reliability regularization loss
        self.rel_loss = ReliabilityRegLoss()

    def forward(
            self,
            model_outputs: Dict[str, torch.Tensor],
            targets: torch.Tensor,
            sasa_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            model_outputs: Dict from MultimodalFERModel.forward() containing:
                - 'logits': (B, num_classes)
                - 'reliability_mag': List of (B, 1, H', W') per block
                - 'reliability_flow': List of (B, 1, H', W') per block
            targets: (B,) integer class labels
            sasa_loss: Pre-computed SASA loss (scalar tensor).
                       Computed externally because it needs au_labels.

        Returns:
            total_loss: Scalar loss for backpropagation
            loss_dict: Dictionary of individual loss components (for logging)
        """
        # --- 1. Classification loss ---
        L_cls = self.cls_loss(model_outputs['logits'], targets)

        # --- 2. SASA loss (pre-computed externally) ---
        if sasa_loss is not None:
            L_sasa = sasa_loss
        else:
            L_sasa = torch.tensor(0.0, device=targets.device)

        # --- 3. Reliability regularization loss ---
        R_all = model_outputs.get('reliability_mag', []) + \
                model_outputs.get('reliability_flow', [])

        if len(R_all) > 0:
            L_rel = self.rel_loss(R_all)
        else:
            L_rel = torch.tensor(0.0, device=targets.device)

        # --- Combine ---
        total_loss = self.lambda_cls * L_cls + self.lambda_sasa * L_sasa + self.lambda_rel * L_rel

        # Loss dict for logging (detached scalars)
        loss_dict = {
            'total': total_loss.item(),
            'cls': L_cls.item(),
            'sasa': L_sasa.item(),
            'rel': L_rel.item(),
            'cls_weighted': (self.lambda_cls * L_cls).item(),
            'sasa_weighted': (self.lambda_sasa * L_sasa).item(),
            'rel_weighted': (self.lambda_rel * L_rel).item(),
        }

        return total_loss, loss_dict


# =============================================================================
# Training & Evaluation Utilities
# =============================================================================

class LossLogger:
    """
    Tracks and reports loss components across training.

    Essential for tuning lambda values: if one loss component dominates
    the gradient, the other objectives are effectively ignored.

    Usage:
        logger = LossLogger()
        for batch in dataloader:
            loss, loss_dict = criterion(outputs, labels, sasa_loss)
            logger.update(loss_dict)
        logger.report(epoch=1)
    """

    def __init__(self):
        self.history = {}
        self.batch_count = 0

    def update(self, loss_dict: Dict[str, float]):
        for key, val in loss_dict.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(val)
        self.batch_count += 1

    def report(self, epoch: int):
        print(f"\n--- Epoch {epoch} Loss Summary ---")
        for key, vals in self.history.items():
            avg = sum(vals) / len(vals)
            print(f"  {key:<20s}: {avg:.4f}")
        # Show relative contribution to help tune lambdas
        if 'cls' in self.history and 'sasa_weighted' in self.history:
            avg_cls = sum(self.history['cls']) / len(self.history['cls'])
            avg_sasa = sum(self.history['sasa_weighted']) / len(self.history['sasa_weighted'])
            avg_rel = sum(self.history['rel_weighted']) / len(self.history['rel_weighted'])
            total = avg_cls + avg_sasa + avg_rel + 1e-8
            print(f"\n  Loss contribution ratios:")
            print(f"    cls:  {avg_cls / total * 100:.1f}%")
            print(f"    sasa: {avg_sasa / total * 100:.1f}%")
            print(f"    rel:  {avg_rel / total * 100:.1f}%")

    def reset(self):
        self.history = {}
        self.batch_count = 0


def train_one_epoch(
        model,
        dataloader,
        criterion,
        sasa_loss_fn,
        optimizer,
        scheduler=None,
        device='cuda',
        max_grad_norm=1.0,
):
    """
    Train for one epoch.

    Args:
        model: MultimodalFERModel
        dataloader: Training dataloader yielding
                    (mag_images, flow_images, texts, labels, au_labels)
        criterion: CombinedLoss instance
        sasa_loss_fn: Your SASA loss function
        optimizer: Optimizer
        scheduler: Optional LR scheduler (step per batch)
        device: Device string
        max_grad_norm: Gradient clipping norm (stabilizes training)

    Returns:
        loss_logger: LossLogger with per-batch loss history
        accuracy: Training accuracy for the epoch
    """
    model.train()
    logger = LossLogger()
    correct = 0
    total = 0

    for batch_idx, (mag_images, flow_images, texts, labels, au_labels) in enumerate(dataloader):
        mag_images = mag_images.to(device)
        flow_images = flow_images.to(device)
        labels = labels.to(device)

        # Forward pass (training mode: returns SASA features + logits + reliability)
        outputs = model(mag_images, flow_images, texts=texts)

        # Compute SASA loss externally (needs au_labels)
        sasa_loss, _ = sasa_loss_fn(
            outputs['f_mag_proj'],
            outputs['f_flow_proj'],
            outputs['f_text_proj'],
            au_labels,
        )

        # Compute combined loss
        total_loss, loss_dict = criterion(outputs, labels, sasa_loss=sasa_loss)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping (important for stability with multiple losses)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Track accuracy
        preds = outputs['logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Log losses
        logger.update(loss_dict)

    accuracy = correct / total if total > 0 else 0.0
    return logger, accuracy


@torch.no_grad()
def evaluate(model, dataloader, num_classes, device='cuda'):
    """
    Evaluate model on test/validation set.

    Note: Text is NOT needed during evaluation. The model in eval mode
    only requires visual inputs and returns logits.

    Args:
        model: MultimodalFERModel
        dataloader: Test dataloader yielding (mag_images, flow_images, labels)
        num_classes: Number of classes (for per-class metrics)
        device: Device string

    Returns:
        metrics: Dict with overall and per-class accuracy, plus predictions
    """
    model.eval()
    all_preds = []
    all_labels = []

    for mag_images, flow_images, labels in dataloader:
        mag_images = mag_images.to(device)
        flow_images = flow_images.to(device)

        # Eval mode: no texts needed, returns logits only
        outputs = model(mag_images, flow_images)
        preds = outputs['logits'].argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Overall accuracy
    overall_acc = (all_preds == all_labels).float().mean().item()

    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (all_preds[mask] == c).float().mean().item()
        else:
            per_class_acc[c] = 0.0

    # Unweighted Average Recall (UAR) - standard metric for imbalanced datasets
    uar = sum(per_class_acc.values()) / num_classes

    # Weighted F1 (via confusion matrix)
    # Note: for full F1 computation, use sklearn.metrics.f1_score in practice
    # This is a simplified version
    metrics = {
        'overall_acc': overall_acc,
        'uar': uar,
        'per_class_acc': per_class_acc,
        'predictions': all_preds,
        'labels': all_labels,
    }

    return metrics


# =============================================================================
# Complete Training Script Example
# =============================================================================

def example_training_loop():
    """
    Demonstrates the complete training pipeline with all loss components.
    """
    print("=" * 70)
    print("Complete Training Pipeline Verification")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 4
    num_classes = 5

    # --- Simulate model outputs (replace with real model in practice) ---
    model_outputs_train = {
        'logits': torch.randn(B, num_classes, device=device, requires_grad=True),
        'f_mag_proj': F.normalize(torch.randn(B, 512, device=device), dim=-1),
        'f_flow_proj': F.normalize(torch.randn(B, 512, device=device), dim=-1),
        'f_text_proj': F.normalize(torch.randn(B, 512, device=device), dim=-1),
        'reliability_mag': [torch.sigmoid(torch.randn(B, 1, 7, 7, device=device))],
        'reliability_flow': [torch.sigmoid(torch.randn(B, 1, 7, 7, device=device))],
    }

    labels = torch.randint(0, num_classes, (B,), device=device)

    # Simulated SASA loss (replace with your sasa_loss_function)
    fake_sasa_loss = torch.tensor(1.5, device=device, requires_grad=True)

    # --- Initialize criterion ---
    # Example class counts from a micro-expression dataset
    class_counts = [32, 99, 45, 38, 33]
    criterion = CombinedLoss(
        num_classes=num_classes,
        class_counts=class_counts,
        gamma=2.0,
        label_smoothing=0.1,
        lambda_sasa=0.1,
        lambda_rel=0.01,
    )

    print(f"\nClass weights (effective_num): {criterion.cls_loss.class_weights.tolist()}")
    print(f"Lambda SASA: {criterion.lambda_sasa}")
    print(f"Lambda REL:  {criterion.lambda_rel}")

    # --- Compute loss ---
    total_loss, loss_dict = criterion(model_outputs_train, labels, sasa_loss=fake_sasa_loss)

    print(f"\n--- Loss Components ---")
    for key, val in loss_dict.items():
        print(f"  {key:<20s}: {val:.4f}")

    # --- Verify gradient flow ---
    total_loss.backward()
    print(f"\n--- Gradient Check ---")
    print(f"  logits has grad: {model_outputs_train['logits'].grad is not None}")

    # --- Test without SASA (edge case) ---
    total_loss_no_sasa, loss_dict_no_sasa = criterion(model_outputs_train, labels, sasa_loss=None)
    print(f"\n--- Without SASA Loss ---")
    print(f"  Total: {loss_dict_no_sasa['total']:.4f}  (sasa={loss_dict_no_sasa['sasa']:.4f})")

    # --- Test LossLogger ---
    print(f"\n--- Loss Logger Demo ---")
    logger = LossLogger()
    for _ in range(5):
        _, ld = criterion(model_outputs_train, labels, sasa_loss=fake_sasa_loss)
        logger.update(ld)
    logger.report(epoch=1)

    # --- Gamma ablation preview ---
    print(f"\n--- Focal Gamma Effect (same inputs) ---")
    for gamma in [0.0, 0.5, 1.0, 2.0, 3.0]:
        fl = ClassWeightedFocalLoss(num_classes=num_classes, class_counts=class_counts, gamma=gamma)
        l = fl(model_outputs_train['logits'].detach(), labels)
        print(f"  gamma={gamma:.1f}: loss={l.item():.4f}")

    print(f"\n{'=' * 70}")
    print("All verifications passed!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    example_training_loop()
