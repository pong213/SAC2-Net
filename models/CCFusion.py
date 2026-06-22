"""
Complementary Consensus Fusion (CCF)
============================================================================
Complete cross-modal fusion architecture for micro-expression recognition.

Architecture:
    CCF
     +-- L x CCFB (Complementary-Consensus Fusion Block)
     |     +-- CEM (Complementary Exchange Module)
     |     |    +-- SpatialNormLayer
     |     |    +-- Cross-Informed Reliability Estimation -> R^mag, R^flow
     |     |    +-- NormLayer + Reliability-Biased Cross-Attention
     |     |    +-- Unreliability-Weighted Residual Update
     |     |
     |     +-- CRM (Consensus Refinement Module)
     |          +-- NormLayer
     |          +-- Reliability-Weighted Shared-Key Attention
     |          +-- Residual Update
     |          +-- NormLayer + FFN
     |
     +--

Design philosophy:
    - CEM is DIVERGENT: each modality borrows from the OTHER at unreliable locations
    - CRM is CONVERGENT: both modalities agree on WHERE to look via shared keys,
      but retrieve their OWN features from those locations
    - Reliability maps flow from CEM to CRM (computed once, consumed twice)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from models.NormFuncs import Dynamic_erf

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


# =============================================================================
# CCF Configuration
# =============================================================================
@dataclass
class CCFConfig:
    """Configuration for CCF fusion module."""
    embed_dim: int = 512    # Feature dimension D
    spatial_h: int = 7      # Spatial height H' (= H/32 for 224x224 input)
    spatial_w: int = 7      # Spatial width W'
    num_blocks: int = 3     # L: number of stacked CCF blocks
    num_heads: int = 8      # Number of attention heads
    ffn_ratio: float = 4.0  # FFN expansion ratio
    dropout: float = 0.0    # Dropout rate
    drop_path: float = 0.0  # Drop path rate
    eps: float = 1e-6       # Epsilon for numerical stability (log-reliability clamping)
    layer_scale_init_value: float = 1e-5  # Layer scale init value

    # --- Ablation toggles ---
    use_cross_informed_rem: bool = True         # True: REM sees both modalities; False: REM sees only self
    use_reliability_bias: bool = True           # Toggle log-reliability bias in cross-attention
    use_unreliability_gating: bool = True       # Toggle unreliability-weighted residual
    use_crm: bool = True                        # Toggle CRM module
    use_reliability_key_fusion: bool = True     # Toggle reliability weighting in CRM shared key
    use_layer_scale: bool = False               # Toggle layer scale in all attention blocks


# =============================================================================
# Module 1 Component: Cross-Informed Reliability Estimation
# =============================================================================
class ReliabilityEstimationModule(nn.Module):
    """
    Cross-informed spatial reliability estimation.
    Separate networks for each modality, each seeing both modalities' features.
    """

    def __init__(self, embed_dim: int, cross_informed: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.cross_informed = cross_informed
        in_channels = embed_dim * 2 if cross_informed else embed_dim

        self.rem_mag = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.rem_flow = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
            self, F_mag: torch.Tensor, F_flow: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            F_mag:  (B, D, H', W') - magnification features
            F_flow: (B, D, H', W') - optical flow features
        Returns:
            R_mag:  (B, 1, H', W') - magnification reliability, clamped to [eps, 1]
            R_flow: (B, 1, H', W') - optical flow reliability, clamped to [eps, 1]
        """
        if self.cross_informed:
            input_mag = torch.cat([F_mag, F_flow], dim=1)
            input_flow = torch.cat([F_flow, F_mag], dim=1)
        else:
            input_mag, input_flow = F_mag, F_flow

        R_mag = self.rem_mag(input_mag).clamp(min=self.eps, max=1.0)
        R_flow = self.rem_flow(input_flow).clamp(min=self.eps, max=1.0)

        return R_mag, R_flow


# =============================================================================
# Module 1 Component: Reliability-Biased Cross-Attention
# =============================================================================
class ReliabilityBiasedCrossAttention(nn.Module):
    """
    Bidirectional cross-attention with log-reliability bias.
    Each modality borrows features from reliable regions of the other.
    Separate projections per direction.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 dropout: float = 0.1, use_reliability_bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_reliability_bias = use_reliability_bias
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Separate projections for each cross-attention direction
        # mag ← flow direction
        self.W_Q_mag = nn.Linear(embed_dim, embed_dim)
        self.W_K_flow = nn.Linear(embed_dim, embed_dim)
        self.W_V_flow = nn.Linear(embed_dim, embed_dim)
        self.out_proj_mag = nn.Linear(embed_dim, embed_dim)

        # flow <- mag direction
        self.W_Q_flow = nn.Linear(embed_dim, embed_dim)
        self.W_K_mag = nn.Linear(embed_dim, embed_dim)
        self.W_V_mag = nn.Linear(embed_dim, embed_dim)
        self.out_proj_flow = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)

    def _cross_attention(self, Q, K, V, R_source, eps):
        """
        Compute multi-head cross-attention with optional reliability bias.

        Args:
            Q: (B, N, D)         - queries from target modality
            K: (B, N, D)         - keys from source modality
            V: (B, N, D)         - values from source modality
            R_source: (B, N, 1)  - reliability of source modality at each location
            eps: float           - epsilon for log clamping
        Returns:
            output: (B, N, D)    - borrowed features
        """
        B, N, _ = Q.shape

        # Reshape to multi-head: (B, num_heads, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, num_heads, N, N)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Add log-reliability bias to steer attention toward reliable source regions
        if self.use_reliability_bias:
            # R_source: (B, N, 1) → (B, 1, 1, N) for broadcasting
            log_R = torch.log(R_source.squeeze(-1).clamp(min=eps))  # (B, N)
            log_R = log_R.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn + log_R  # broadcast to (B, num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        # Weighted sum: (B, num_heads, N, head_dim)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)

        return out

    def forward(self, F_mag_seq, F_flow_seq, R_mag_seq, R_flow_seq, eps=1e-6):
        """
        Args:
            F_mag_seq:  (B, N, D) - magnification features in sequence form
            F_flow_seq: (B, N, D) - optical flow features in sequence form
            R_mag_seq:  (B, N, 1) - magnification reliability per location
            R_flow_seq: (B, N, 1) - optical flow reliability per location
            eps: float
        Returns:
            F_mag_borrowed:  (B, N, D) - features borrowed from flow for mag
            F_flow_borrowed: (B, N, D) - features borrowed from mag for flow
        """
        # mag ← flow: magnification queries, optical flow provides K/V
        Q_mag = self.W_Q_mag(F_mag_seq)
        K_flow = self.W_K_flow(F_flow_seq)
        V_flow = self.W_V_flow(F_flow_seq)
        F_mag_borrowed = self.out_proj_mag(
            self._cross_attention(Q_mag, K_flow, V_flow, R_flow_seq, eps)
        )

        # flow ← mag: optical flow queries, magnification provides K/V
        Q_flow = self.W_Q_flow(F_flow_seq)
        K_mag = self.W_K_mag(F_mag_seq)
        V_mag = self.W_V_mag(F_mag_seq)
        F_flow_borrowed = self.out_proj_flow(
            self._cross_attention(Q_flow, K_mag, V_mag, R_mag_seq, eps)
        )

        return F_mag_borrowed, F_flow_borrowed


# =============================================================================
# Module 1: Complementary Exchange Module (CEM)
# =============================================================================
class ComplementaryExchangeModule(nn.Module):
    """
    CEM: Repairs each modality by borrowing from reliable regions of the other.

    Pipeline:
        SpatialNormLayer -> REM -> R^mag, R^flow
        NormLayer -> Reliability-Biased Cross-Attention
        Unreliability-Weighted Residual Update
    """

    def __init__(self, config: CCFConfig):
        super().__init__()
        D = config.embed_dim

        # Pre-norm for REM (spatial)
        self.norm_pre_rem_mag = Dynamic_erf(D)
        self.norm_pre_rem_flow = Dynamic_erf(D)

        # Reliability Estimation
        self.rem = ReliabilityEstimationModule(
            embed_dim=D,
            cross_informed=config.use_cross_informed_rem,
            eps=config.eps,
        )

        # Pre-norm for cross-attention (sequence)
        self.norm_pre_ca_mag = Dynamic_erf(D, channels_last=True)
        self.norm_pre_ca_flow = Dynamic_erf(D, channels_last=True)

        # Cross-attention
        self.cross_attn = ReliabilityBiasedCrossAttention(
            embed_dim=D,
            num_heads=config.num_heads,
            dropout=config.dropout,
            use_reliability_bias=config.use_reliability_bias,
        )

        self.use_unreliability_gating = config.use_unreliability_gating
        self.eps = config.eps
        # Drop path
        self.drop_path = DropPath(config.drop_path) if config.drop_path > 0.0 else nn.Identity()
        # Layer Scale
        self.use_layer_scale = config.use_layer_scale
        if self.use_layer_scale:
            # Magnified feature layer scale
            self.mag_layer_scale = nn.Parameter(
                config.layer_scale_init_value * torch.ones((1, D)), requires_grad=True
            )
            # Flow feature layer scale
            self.flow_layer_scale = nn.Parameter(
                config.layer_scale_init_value * torch.ones((1, D)), requires_grad=True
            )

    def forward(
            self, F_mag: torch.Tensor, F_flow: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            F_mag, F_flow: (B, D, H', W')
        Returns:
            F_mag, F_flow: (B, D, H', W') updated features
            R_mag, R_flow: (B, 1, H', W') reliability maps (passed to CRM)
        """
        B, D, H, W = F_mag.shape
        eps = self.eps

        # --- Reliability Estimation ---
        F_mag_normed = self.norm_pre_rem_mag(F_mag)
        F_flow_normed = self.norm_pre_rem_flow(F_flow)
        R_mag, R_flow = self.rem(F_mag_normed, F_flow_normed)

        U_mag = 1.0 - R_mag
        U_flow = 1.0 - R_flow

        # --- Reshape to sequence ---
        F_mag_seq = F_mag.flatten(2).transpose(1, 2)  # (B, N, D)
        F_flow_seq = F_flow.flatten(2).transpose(1, 2)

        # Pre-norm for cross-attention
        F_mag_seq_n = self.norm_pre_ca_mag(F_mag_seq)
        F_flow_seq_n = self.norm_pre_ca_flow(F_flow_seq)

        R_mag_seq = R_mag.flatten(2).transpose(1, 2)  # (B, N, 1)
        R_flow_seq = R_flow.flatten(2).transpose(1, 2)

        # --- Cross-attention ---
        F_mag_borrowed, F_flow_borrowed = self.cross_attn(
            F_mag_seq_n, F_flow_seq_n, R_mag_seq, R_flow_seq, eps,
        )

        # --- Unreliability-weighted residual ---
        U_mag_seq = U_mag.flatten(2).transpose(1, 2)  # (B, N, 1)
        U_flow_seq = U_flow.flatten(2).transpose(1, 2)

        # Unreliability-Weighted Residual Update
        if self.use_unreliability_gating:
            if self.use_layer_scale:
                F_mag_seq = F_mag_seq + self.drop_path(self.mag_layer_scale * (U_mag_seq * F_mag_borrowed))
                F_flow_seq = F_flow_seq + self.drop_path(self.flow_layer_scale * (U_flow_seq * F_flow_borrowed))
            else:
                F_mag_seq = F_mag_seq + self.drop_path(U_mag_seq * F_mag_borrowed)
                F_flow_seq = F_flow_seq + self.drop_path(U_flow_seq * F_flow_borrowed)
        else:
            # Ablation: simple residual without gating
            F_mag_seq = F_mag_seq + self.drop_path(F_mag_borrowed)
            F_flow_seq = F_flow_seq + self.drop_path(F_flow_borrowed)

        # Reshape back to spatial: (B, N, D) → (B, D, H, W)
        F_mag = F_mag_seq.transpose(1, 2).view(B, D, H, W)
        F_flow = F_flow_seq.transpose(1, 2).view(B, D, H, W)

        return F_mag, F_flow, R_mag, R_flow


# =============================================================================
# Module 2 Component: Feed-Forward Network
# =============================================================================
class FFN(nn.Module):
    """FFN as used in Vision Transformer, standard transformer post-processing."""

    def __init__(self, embed_dim: int, ffn_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, D)
        Returns:
            x: (B, N, D)
        """
        return self.ffn(x)


# =============================================================================
# Module 2 Component: Reliability-Weighted Shared-Key Attention
# =============================================================================
class ReliabilityWeightedSharedKeyAttention(nn.Module):
    """
    Shared-key attention where both modalities attend to a consensus key map.

    Key mechanism:
        K^shared = R_hat^mag * K^mag + R_hat^flow * K^flow

    where R_hat are normalized reliability weights from CEM. Each modality
    then uses its own Q and V with this shared K:
        Attn^mag = softmax(Q^mag @ K^shared^T / sqrt(d)) @ V^mag
        Attn^flow = softmax(Q^flow @ K^shared^T / sqrt(d)) @ V^flow

    Design rationale:
        - Shared K: both modalities agree on WHERE to look (spatial consensus)
        - Separate Q: each modality asks different questions of the consensus
        - Separate V: each modality retrieves its own features (preserves identity)
        - Reliability weighting: consensus is anchored to trustworthy information
    """

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 dropout: float = 0.1, use_reliability_weighting: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_reliability_weighting = use_reliability_weighting
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Separate Q per modality
        self.W_Q_mag = nn.Linear(embed_dim, embed_dim)
        self.W_Q_flow = nn.Linear(embed_dim, embed_dim)

        # Separate K per modality (projected before reliability-weighted fusion)
        self.W_K_mag = nn.Linear(embed_dim, embed_dim)
        self.W_K_flow = nn.Linear(embed_dim, embed_dim)

        # Separate V per modality (each retrieves its own features)
        self.W_V_mag = nn.Linear(embed_dim, embed_dim)
        self.W_V_flow = nn.Linear(embed_dim, embed_dim)

        # Separate output projections
        self.out_proj_mag = nn.Linear(embed_dim, embed_dim)
        self.out_proj_flow = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)

    def _attend_shared_key(self, Q, K_shared, V):
        """Standard multi-head attention with pre-computed shared key."""
        B, N, _ = Q.shape
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K_shared = K_shared.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K_shared.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, V)
        return out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)

    def forward(
            self,
            F_mag_seq: torch.Tensor,
            F_flow_seq: torch.Tensor,
            R_mag_seq: torch.Tensor,
            R_flow_seq: torch.Tensor,
            eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            F_mag_seq:  (B, N, D) - magnification features in sequence form
            F_flow_seq: (B, N, D) - optical flow features in sequence form
            R_mag_seq:  (B, N, 1) - magnification reliability maps from CEM
            R_flow_seq: (B, N, 1) - optical flow reliability maps from CEM
            eps: float - numerical stability
        Returns:
            out_mag:  (B, N, D) - magnification features through attention
            out_flow: (B, N, D) - optical flow features through attention
        """
        # --- Project Q, K, V ---
        Q_mag = self.W_Q_mag(F_mag_seq)
        Q_flow = self.W_Q_flow(F_flow_seq)

        K_mag = self.W_K_mag(F_mag_seq)
        K_flow = self.W_K_flow(F_flow_seq)

        V_mag = self.W_V_mag(F_mag_seq)
        V_flow = self.W_V_flow(F_flow_seq)

        # --- Build shared key ---
        if self.use_reliability_weighting:
            # Normalize reliability to sum to 1 at each location
            R_total = R_mag_seq + R_flow_seq + eps  # (B, N, 1)
            R_hat_mag = R_mag_seq / R_total  # (B, N, 1)
            R_hat_flow = R_flow_seq / R_total  # (B, N, 1)
            K_shared = R_hat_mag * K_mag + R_hat_flow * K_flow  # (B, N, D)
        else:
            # Ablation: simple average
            K_shared = (K_mag + K_flow) / 2.0

        # --- Each modality attends to shared key, retrieves own values ---
        out_mag = self.out_proj_mag(
            self._attend_shared_key(Q_mag, K_shared, V_mag)
        )
        out_flow = self.out_proj_flow(
            self._attend_shared_key(Q_flow, K_shared, V_flow)
        )

        return out_mag, out_flow


# =============================================================================
# Module 2: Consensus Refinement Module (CRM)
# =============================================================================
class ConsensusRefinementModule(nn.Module):
    """
    CRM: Aligns spatial focus across modalities via shared-key attention.

    Pipeline:
        NormLayer -> Reliability-Weighted Shared-Key Attention -> Residual
        NormLayer -> FFN -> Residual

    Receives reliability maps from CEM (computed once, consumed twice).
    """

    def __init__(self, config: CCFConfig):
        super().__init__()
        D = config.embed_dim

        # Pre-norm
        self.norm_pre_attn_mag = Dynamic_erf(D, channels_last=True)
        self.norm_pre_attn_flow = Dynamic_erf(D, channels_last=True)

        # Shared-key attention
        self.shared_key_attn = ReliabilityWeightedSharedKeyAttention(
            embed_dim=D,
            num_heads=config.num_heads,
            dropout=config.dropout,
            use_reliability_weighting=config.use_reliability_key_fusion,
        )

        # Post-norm + FFN
        self.norm_post_attn_mag = Dynamic_erf(D, channels_last=True)
        self.norm_post_attn_flow = Dynamic_erf(D, channels_last=True)
        self.ffn_mag = FFN(D, config.ffn_ratio, config.dropout)
        self.ffn_flow = FFN(D, config.ffn_ratio, config.dropout)

        self.eps = config.eps
        # Drop path
        self.drop_path = DropPath(config.drop_path) if config.drop_path > 0.0 else nn.Identity()
        # Layer Scale
        self.use_layer_scale = config.use_layer_scale
        if self.use_layer_scale:
            # Magnified feature layer scale
            self.mag_layer_scale_1 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((1, D)), requires_grad=True
            )
            self.mag_layer_scale_2 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((1, D)), requires_grad=True
            )
            # Flow feature layer scale
            self.flow_layer_scale_1 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((1, D)), requires_grad=True
            )
            self.flow_layer_scale_2 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((1, D)), requires_grad=True
            )

    def forward(
            self,
            F_mag: torch.Tensor,
            F_flow: torch.Tensor,
            R_mag: torch.Tensor,
            R_flow: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            F_mag, F_flow: (B, D, H', W') features from CEM
            R_mag, R_flow: (B, 1, H', W') reliability maps from CEM
        Returns:
            F_mag, F_flow: (B, D, H', W') refined features
        """
        B, D, H, W = F_mag.shape

        # --- Reshape to sequence ---
        F_mag_seq = F_mag.flatten(2).transpose(1, 2)  # (B, N, D)
        F_flow_seq = F_flow.flatten(2).transpose(1, 2)

        R_mag_seq = R_mag.flatten(2).transpose(1, 2)  # (B, N, 1)
        R_flow_seq = R_flow.flatten(2).transpose(1, 2)

        # --- Pre-norm ---
        F_mag_seq_n = self.norm_pre_attn_mag(F_mag_seq)
        F_flow_seq_n = self.norm_pre_attn_flow(F_flow_seq)

        # --- Shared-key attention ---
        out_mag, out_flow = self.shared_key_attn(
            F_mag_seq_n, F_flow_seq_n, R_mag_seq, R_flow_seq, self.eps,
        )

        if self.use_layer_scale:
            # Residual + Attention
            F_mag_seq = F_mag_seq + self.drop_path(self.mag_layer_scale_1 * out_mag)
            F_flow_seq = F_flow_seq + self.drop_path(self.flow_layer_scale_1 * out_flow)

            # Residual + FFN
            F_mag_seq = F_mag_seq + self.drop_path(
                self.mag_layer_scale_2 * self.ffn_mag(self.norm_post_attn_mag(F_mag_seq))
            )
            F_flow_seq = F_flow_seq + self.drop_path(
                self.flow_layer_scale_2 * self.ffn_flow(self.norm_post_attn_flow(F_flow_seq))
            )
        else:
            F_mag_seq = F_mag_seq + self.drop_path(out_mag)
            F_flow_seq = F_flow_seq + self.drop_path(out_flow)

            F_mag_seq = F_mag_seq + self.drop_path(self.ffn_mag(self.norm_post_attn_mag(F_mag_seq)))
            F_flow_seq = F_flow_seq + self.drop_path(self.ffn_flow(self.norm_post_attn_flow(F_flow_seq)))

        # Reshape back to spatial: (B, N, D) → (B, D, H, W)
        F_mag = F_mag_seq.transpose(1, 2).view(B, D, H, W)
        F_flow = F_flow_seq.transpose(1, 2).view(B, D, H, W)

        return F_mag, F_flow


# =============================================================================
# CCFB: Complementary-Consensus Fusion Block
# =============================================================================
class ComplementaryConsensusFusionBlock(nn.Module):
    """
    One complete fusion block: CEM followed by CRM.

    CEM (divergent): each modality borrows from the other at weak spots.
    CRM (convergent): both modalities align spatial focus via shared keys.

    Reliability maps computed in CEM are reused in CRM (no recomputation).
    """

    def __init__(self, config: CCFConfig):
        super().__init__()
        self.cem = ComplementaryExchangeModule(config)
        self.crm = ConsensusRefinementModule(config) if config.use_crm else None

    def forward(
            self, F_mag: torch.Tensor, F_flow: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            F_mag, F_flow: (B, D, H', W')
        Returns:
            F_mag, F_flow: (B, D, H', W') updated features
            R_mag, R_flow: (B, 1, H', W') reliability maps
        """
        # CEM: complementary exchange
        F_mag, F_flow, R_mag, R_flow = self.cem(F_mag, F_flow)

        # CRM: consensus refinement (reliability maps flow from CEM)
        if self.crm is not None:
            F_mag, F_flow = self.crm(F_mag, F_flow, R_mag, R_flow)

        return F_mag, F_flow, R_mag, R_flow


# =============================================================================
# CCF: Full Architecture
# =============================================================================
class CCFusion(nn.Module):
    """
    Complementary Consensus Fusion.

    L stacked CCF blocks.
    """

    def __init__(self, config: CCFConfig):
        super().__init__()

        # L stacked CCFB blocks
        self.blocks = nn.ModuleList([
            ComplementaryConsensusFusionBlock(config)
            for _ in range(config.num_blocks)
        ])

    def forward(
            self, F_mag: torch.Tensor, F_flow: torch.Tensor
    ):
        """
        Args:
            F_mag, F_flow: (B, D, H', W') from visual encoders
        Returns:
            F_mag: (B, D, H', W') Magnification feature after CCF
            F_flow: (B, D, H', W') Optical flow feature after CCF
            aux: dict with reliability maps
        """
        all_R_mag, all_R_flow = [], []

        for block in self.blocks:
            F_mag, F_flow, R_mag, R_flow = block(F_mag, F_flow)
            all_R_mag.append(R_mag)
            all_R_flow.append(R_flow)

        # Return auxiliary information for visualization and potential losses
        reliability_maps = {
            "reliability_mag": all_R_mag,  # List of (B, 1, H', W') per block
            "reliability_flow": all_R_flow,  # List of (B, 1, H', W') per block
        }

        return F_mag, F_flow, reliability_maps


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Test configuration
    B = 4  # batch size
    D = 512  # embed dim
    H_prime = 7  # spatial height
    W_prime = 7  # spatial width

    # Create dummy inputs (simulating encoder outputs)
    F_mag = torch.randn(B, D, H_prime, W_prime, device=device)
    F_flow = torch.randn(B, D, H_prime, W_prime, device=device)

    config = CCFConfig()

    model = CCFusion(config).to(device)

    # Forward pass
    with torch.no_grad():
        F_mag_out, F_flow_out, rel_maps = model(F_mag, F_flow)
    print(F_mag_out.shape)
    print(F_flow_out.shape)
    print(len(rel_maps["reliability_mag"]))


if __name__ == "__main__":
    test()
