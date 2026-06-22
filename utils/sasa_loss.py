"""
Semantic Anchoring Soft Alignment (SASA) Loss

A novel multimodal contrastive learning method for micro expression recognition
that uses text as a semantic anchor and AU-based soft labels for alignment.

Key Features:
    - Text serves as semantic anchor (visual modalities align TO text)
    - AU Jaccard similarity creates soft contrastive labels
    - Hierarchical AU handling (bilateral vs unilateral)
    - Only ONE hyperparameter: temperature τ

AU Pair Similarity Rules:
    - Exact match (AU4 vs AU4): 1.0
    - Bilateral vs Unilateral (AU4 vs L4): 0.7
    - Same side unilateral (L4 vs L4): 1.0
    - Opposite side unilateral (L4 vs R4): 0.6
    - Unrelated (AU4 vs AU12): 0.0

Author: Pong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union


# Type alias for AU labels (can be int or str)
AULabel = Union[int, str]


def parse_au_label(au_label: AULabel) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse an AU label into its base number and laterality.

    Args:
        au_label: AU label as int (4) or str ('4', 'l4', 'r4', 'L4', 'R4')

    Returns:
        Tuple of (base_au_number, laterality)
        - laterality is 'l', 'r', or None (bilateral)

    Examples:
        4     -> (4, None)   # Bilateral AU4
        '4'   -> (4, None)   # Bilateral AU4 (string)
        'l4'  -> (4, 'l')    # Left AU4
        'r4'  -> (4, 'r')    # Right AU4
        'L4'  -> (4, 'l')    # Left AU4 (case-insensitive)
    """
    if isinstance(au_label, int):
        return au_label, None

    au_str = str(au_label).lower().strip()

    if au_str.startswith('l'):
        try:
            return int(au_str[1:]), 'l'
        except ValueError:
            return None, None
    elif au_str.startswith('r'):
        try:
            return int(au_str[1:]), 'r'
        except ValueError:
            return None, None
    else:
        try:
            return int(au_str), None
        except ValueError:
            return None, None


def compute_pairwise_au_similarity(au_a: AULabel, au_b: AULabel) -> float:
    """
    Compute similarity between two individual AU labels.

    Similarity Rules:
        - Exact match (AU4 vs AU4): 1.0
        - Bilateral vs Unilateral (AU4 vs L4): 0.7 (partial match)
        - Same side unilateral (L4 vs L4): 1.0
        - Opposite side unilateral (L4 vs R4): 0.6 (partial match)
        - Unrelated AUs (AU4 vs AU12): 0.0

    Args:
        au_a: First AU label
        au_b: Second AU label

    Returns:
        Similarity score in [0, 1]
    """
    base_a, lat_a = parse_au_label(au_a)
    base_b, lat_b = parse_au_label(au_b)

    # Invalid AU labels
    if base_a is None or base_b is None:
        return 0.0

    # Different AU numbers -> unrelated
    if base_a != base_b:
        return 0.0

    # Same AU number -> check laterality
    if lat_a == lat_b:
        # Both bilateral (None == None) or same side (l == l, r == r)
        return 1.0
    elif lat_a is None or lat_b is None:
        # One bilateral, one unilateral (AU4 vs L4) / alpha
        return 0.7
    else:
        # Different sides (L4 vs R4) / beta
        return 0.6


def compute_au_set_similarity(
        au_set_a: List[AULabel],
        au_set_b: List[AULabel],
        use_hierarchical: bool = True,
) -> float:
    """
    Compute similarity between two AU label sets.

    When use_hierarchical=True:
        - Handles lateralized AUs (L4, R4)
        - Partial matches between bilateral and unilateral
        - Uses best pairwise matching strategy

    When use_hierarchical=False:
        - Standard Jaccard similarity

    Args:
        au_set_a: List of AU labels for sample A (e.g., [4, 7] or ['l4', 7])
        au_set_b: List of AU labels for sample B
        use_hierarchical: Whether to use hierarchical AU relationships

    Returns:
        Similarity score in [0, 1]

    Examples:
        ([4, 7], [4, 7])       -> 1.0   (identical)
        ([4, 7], ['l4', 7])    -> 0.85  (AU7 exact + AU4/L4 partial)
        ([4], ['l4', 12])    -> 0.35  (bilateral vs both unilateral)
        ([6, 12], [4, 7])      -> 0.0   (no overlap)
    """
    # Handle empty sets (neutral expressions)
    if not au_set_a and not au_set_b:
        return 1.0
    if not au_set_a or not au_set_b:
        return 0.0

    # Standard Jaccard (non-hierarchical)
    if not use_hierarchical:
        set_a = set(str(au) for au in au_set_a)
        set_b = set(str(au) for au in au_set_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    # Hierarchical similarity with the best pairwise matching
    total_similarity = 0.0
    matched_indices = set()
    num_au_family_pairs = 0

    # For each AU in set A, find best match in set B
    for au_a in au_set_a:
        best_sim = 0.0
        best_idx = None

        for idx, au_b in enumerate(au_set_b):
            if idx in matched_indices:
                continue
            sim = compute_pairwise_au_similarity(au_a, au_b)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        total_similarity += best_sim
        if best_idx is not None and best_sim > 0.0:
            matched_indices.add(best_idx)
            if best_sim < 1.0:
                num_au_family_pairs += 1

    # Compute effective union size
    set_a = set(str(au) for au in au_set_a)
    set_b = set(str(au) for au in au_set_b)
    union_size = len(set_a | set_b) - num_au_family_pairs

    # Avoid division by zero
    if union_size <= 0:
        return 0.0

    similarity = total_similarity / union_size
    return min(1.0, similarity)


class SASALoss(nn.Module):
    """
    Semantic Anchoring Soft Alignment (SASA) Loss.

    A multimodal contrastive loss that:
    1. Uses text as semantic anchor (visual -> text alignment)
    2. Employs AU-based soft labels via Jaccard similarity
    3. Supports hierarchical AU relationships (bilateral/unilateral)

    Loss Formula:
        L_SASA = L^{mag->text} + L^{flow->text}

        where:
        L^{v->t} = -(1/B) * sum_i sum_j S_norm[i,j] * log(p[i,j])

    Args:
        temperature: Temperature parameter for scaling logits (default: 0.07)
        use_hierarchical: Use hierarchical AU relationships (default: True)

    Example:
        sasa = SASALoss(temperature=0.07)
        loss, loss_dict = sasa(f_mag, f_flow, f_text, au_labels)
    """

    def __init__(
            self,
            temperature: float = 0.07,
            use_hierarchical: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.use_hierarchical = use_hierarchical

    def compute_similarity_matrix(
            self,
            au_labels: List[List[AULabel]]
    ) -> torch.Tensor:
        """
        Compute AU similarity matrix for a batch.

        Uses symmetric property to optimize computation:
        S[i,j] = S[j,i], so only compute upper triangle.

        Args:
            au_labels: List of AU label lists for each sample
                      e.g., [[4], [4, 7], [6, 12], ['l4', 9]]

        Returns:
            Similarity matrix S of shape [B, B] with values in [0, 1]
        """
        B = len(au_labels)
        S = torch.zeros(B, B)

        # Compute only upper triangle (symmetric matrix)
        for i in range(B):
            S[i, i] = 1.0  # Diagonal is always 1
            for j in range(i + 1, B):
                sim = compute_au_set_similarity(
                    au_labels[i],
                    au_labels[j],
                    use_hierarchical=self.use_hierarchical
                )
                S[i, j] = sim
                S[j, i] = sim  # Symmetric

        return S

    def forward(
            self,
            f_mag: torch.Tensor,
            f_flow: torch.Tensor,
            f_text: torch.Tensor,
            au_labels: List[List[AULabel]],
            text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SASA loss.

        Args:
            f_mag: Magnified image features [B, C]
            f_flow: Optical flow features [B, C]
            f_text: Text features [B, C] or [B, N, C]
            au_labels: AU label lists for each sample
            text_mask: Optional attention mask for text sequences

        Returns:
            loss: Total SASA loss (scalar tensor)
            loss_dict: Dictionary with individual loss components
        """
        # Step 1: Feature Pooling
        if f_mag.dim() != 2:
            f_mag_pooled = f_mag.mean(dim=[-2, -1])    # [B, C]
            f_flow_pooled = f_flow.mean(dim=[-2, -1])  # [B, C]
        else:
            f_mag_pooled = f_mag
            f_flow_pooled = f_flow

        if f_text.dim() == 3:
            # Sequence features [B, N, C] -> [B, C]
            if text_mask is not None:
                mask = text_mask.unsqueeze(-1).float()
                f_text_pooled = (f_text * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                f_text_pooled = f_text.mean(dim=1)
        else:
            f_text_pooled = f_text  # Already [B, C]

        # Step 2: L2 Normalization
        f_mag_norm = F.normalize(f_mag_pooled, p=2, dim=-1)
        f_flow_norm = F.normalize(f_flow_pooled, p=2, dim=-1)
        f_text_norm = F.normalize(f_text_pooled, p=2, dim=-1)

        # Step 3: Compute AU Similarity Matrix
        similarity_matrix = self.compute_similarity_matrix(au_labels)

        # Step 4: Row Normalization (convert to soft labels)
        soft_labels = similarity_matrix / similarity_matrix.sum(dim=1, keepdim=True).clamp(min=1e-9)
        soft_labels = soft_labels.to(f_mag.device)

        # Step 5: Compute Similarity Logits
        logits_mag = torch.matmul(f_mag_norm, f_text_norm.T) / self.temperature
        logits_flow = torch.matmul(f_flow_norm, f_text_norm.T) / self.temperature

        # Step 6: Softmax Probabilities
        probs_mag = F.softmax(logits_mag, dim=1)
        probs_flow = F.softmax(logits_flow, dim=1)

        # Step 7: Soft Cross-Entropy Loss
        loss_mag = -torch.sum(soft_labels * torch.log(probs_mag.clamp(min=1e-9)), dim=1).mean()
        loss_flow = -torch.sum(soft_labels * torch.log(probs_flow.clamp(min=1e-9)), dim=1).mean()

        # Step 8: Total SASA Loss
        loss_total = loss_mag + loss_flow

        loss_dict = {
            'loss_mag_text': loss_mag.item(),
            'loss_flow_text': loss_flow.item(),
            'loss_total': loss_total.item()
        }

        return loss_total, loss_dict

    def get_soft_labels(self, au_labels: List[List[AULabel]]) -> torch.Tensor:
        """
        Get normalized soft label matrix for visualization/debugging.

        Args:
            au_labels: AU label lists for each sample

        Returns:
            Normalized soft label matrix [B, B]
        """
        S = self.compute_similarity_matrix(au_labels)
        # print(S)
        return S / S.sum(dim=1, keepdim=True).clamp(min=1e-9)

    def extra_repr(self) -> str:
        """Extra representation for print(model)."""
        return f'temperature={self.temperature}, use_hierarchical={self.use_hierarchical}'


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

def au_similarity():
    """Test AU similarity computation."""
    print("=" * 70)
    print("AU PAIRWISE SIMILARITY TEST")
    print("=" * 70)

    test_cases = [
        (4, 4, "AU4 vs AU4 (exact match)"),
        ('l4', 'l4', "L4 vs L4 (exact match, same side)"),
        (4, 'l4', "AU4 vs L4 (bilateral vs unilateral)"),
        ('l4', 4, "L4 vs AU4 (unilateral vs bilateral)"),
        ('l4', 'r4', "L4 vs R4 (opposite sides)"),
        (4, 12, "AU4 vs AU12 (unrelated)"),
        ('l4', 'l12', "L4 vs L12 (unrelated, same side)"),
    ]

    print("\nPairwise AU Similarity:")
    for au_a, au_b, desc in test_cases:
        sim = compute_pairwise_au_similarity(au_a, au_b)
        print(f"  {desc}: {sim:.2f}")


def set_similarity():
    """Test AU set similarity computation."""
    print("\n" + "=" * 70)
    print("AU SET SIMILARITY TEST")
    print("=" * 70)

    test_cases = [
        ([4, 7], [4, 7], "Identical sets"),
        ([4, 7], ['l4', 7], "AU4 vs L4 (partial match)"),
        ([4], ['l4', 'r4'], "AU4 vs L4+R4"),
        ([6, 12], [4, 7], "No overlap"),
        ([], [], "Both empty (neutral)"),
        ([4], [], "One empty"),
    ]

    print("\nSet Similarity (Hierarchical):")
    for set_a, set_b, desc in test_cases:
        sim_hier = compute_au_set_similarity(set_a, set_b, use_hierarchical=True)
        sim_std = compute_au_set_similarity(set_a, set_b, use_hierarchical=False)
        print(f"  {desc}")
        print(f"    Sets: {set_a} vs {set_b}")
        print(f"    Hierarchical: {sim_hier:.2f}, Standard: {sim_std:.2f}")


def sasa_loss():
    """Test SASA loss computation."""
    print("\n" + "=" * 70)
    print("SASA LOSS TEST")
    print("=" * 70)

    # Create dummy data
    B, C, H, W = 4, 768, 7, 7
    f_mag = torch.randn(B, C, H, W)
    f_flow = torch.randn(B, C, H, W)
    f_text = torch.randn(B, C)

    au_labels = [
        [4, 7],      # Sample 0
        ['l4', 7],   # Sample 1
        [6, 12],     # Sample 2
        [4],         # Sample 3
    ]

    print(f"\nInput shapes:")
    print(f"  f_mag:  {f_mag.shape}")
    print(f"  f_flow: {f_flow.shape}")
    print(f"  f_text: {f_text.shape}")

    print(f"\nAU Labels:")
    for i, aus in enumerate(au_labels):
        print(f"  S{i}: {aus}")

    # Initialize and compute
    sasa = SASALoss(temperature=0.07, use_hierarchical=True)
    print(f"\n{sasa}")

    # Show soft labels
    soft_labels = sasa.get_soft_labels(au_labels)
    print("\nSoft Label Matrix:")
    print("       S0    S1    S2    S3")
    for i in range(B):
        row = f"  S{i}: " + "  ".join([f"{soft_labels[i, j]:.2f}" for j in range(B)])
        print(row)

    # Compute loss
    loss, loss_dict = sasa(f_mag, f_flow, f_text, au_labels)
    print(f"\nLoss Values:")
    print(f"  L^{{mag->text}}:  {loss_dict['loss_mag_text']:.4f}")
    print(f"  L^{{flow->text}}: {loss_dict['loss_flow_text']:.4f}")
    print(f"  L_SASA (total):  {loss_dict['loss_total']:.4f}")


def cal_sim_mat():
    sasa = SASALoss(temperature=0.07)
    print(f"\n{sasa}")

    # au_labels = [
    #     [4],  # Sample 0: AU4
    #     [4, 7],  # Sample 1: AU4 + AU7
    #     [6, 12],  # Sample 2: AU6 + AU12
    #     [4],  # Sample 3: AU4
    # ]
    au_labels = [["4", "7"], ["l4", "7"], ["6", "12"], ["r4"]]

    # Compute soft labels
    print(f"\nSoft Label Matrix (S̃):")
    S_norm = sasa.get_soft_labels(au_labels)
    print("       S0    S1    S2    S3")
    for i in range(4):
        row = f"  S{i}: " + "  ".join([f"{S_norm[i, j]:.2f}" for j in range(4)])
        print(row)


if __name__ == "__main__":
    # test_au_similarity()
    # test_set_similarity()
    # test_sasa_loss()
    cal_sim_mat()
