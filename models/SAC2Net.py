"""Multimodal Micro-expression Recognition Model

This module provides a unified architecture for multimodal MER:
- Magnified image encoder (HFE)
- Optical flow encoder (HFE)
- CLIP text encoder (frozen)
- SASA-compatible feature outputs
- CCF cross-modal fusion

Key design decisions:
- Text is used for SASA alignment (training only) and does NOT enter the classifier
- Training forward: returns SASA features + logits + reliability maps
- Testing forward: returns logits only (text encoder is skipped entirely)
- Weight initialization excludes frozen CLIP encoder

Author: Pong
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.layers import DropPath, trunc_normal_

# --- Custom modules ---
from models.Hybrid_Fast_Encoder import hybrid_fast_encoder
from models.NormFuncs import Dynamic_erf
from models.CCFusion import CCFusion, CCFConfig

# For CLIP text encoder
try:
    from transformers import CLIPTokenizer, CLIPTextModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Install with: pip install transformers")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# =============================================================================
# TEXT ENCODER (CLIP-based)
# =============================================================================

class TextEncoder(nn.Module):
    """
    Text encoder using CLIP's text transformer.
    Outputs [B, 512] CLS token features.

    Args:
        model_name: CLIP model variant
        local_path: Saved local CLIP model path
        freeze: Whether to freeze all CLIP weights
        max_length: Maximum token sequence length (CLIP default: 77)
    """

    def __init__(
            self,
            model_name: str = "openai/clip-vit-base-patch32",
            local_path: Optional[str] = r"./models/CLIP_models/clip-vit-base-patch32",
            freeze: bool = True,
            max_length: int = 77,
    ):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required. Install with: pip install transformers")

        self.max_length = max_length

        text_encoder_weights_path = os.path.join(local_path, "model.safetensors")
        if not os.path.exists(text_encoder_weights_path):
            # Load CLIP text model and tokenizer from HuggingFace
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_model = CLIPTextModel.from_pretrained(model_name)

            # Save to local
            self.tokenizer.save_pretrained(local_path)
            self.text_model.save_pretrained(local_path)
        else:
            # Load from local
            self.tokenizer = CLIPTokenizer.from_pretrained(local_path)
            self.text_model = CLIPTextModel.from_pretrained(local_path)

        # CLIP ViT-B/32 text encoder outputs 512-dim
        self.embed_dim = self.text_model.config.hidden_size  # 512

        if freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def forward(self, texts: List[str], device: Optional[torch.device] = None) -> Tensor:
        """
        Encode text strings to feature vectors.

        Note: Decorated with @torch.no_grad() since the text encoder is frozen.
        This saves memory by not building the computation graph.

        Args:
            texts: List of B text strings
            device: Target device

        Returns:
            text_features: [B, 512]
        """
        if device is None:
            device = next(self.text_model.parameters()).device

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get text features
        outputs = self.text_model(**inputs)

        # Return pooler_output (CLS token)
        text_features = outputs.pooler_output  # [B, 512]

        return text_features


# =============================================================================
# SASA PROJECTION HEAD
# =============================================================================

class SASAProjectionHead(nn.Module):
    """
    Projects visual and text features into the shared SASA alignment space.

    Visual: Global average pooling (spatial → vector) + linear projection
    Text:   Linear projection only (already a vector)

    Both outputs will be L2-normalized for cosine similarity in SASA loss.

    Args:
        embed_dim: Input feature dimension (512)
        projection_dim: Output projection dimension (512)
        spatial_size: Spatial resolution of visual features (7 for 224/32)
    """

    def __init__(self, embed_dim: int = 512, projection_dim: int = 512, spatial_size: int = 7):
        super().__init__()

        # Visual pooling: GAP
        self.mag_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, D, 1, 1)
            nn.Flatten(1),  # (B, D)
        )
        self.flow_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

        # Projection layers (visual and text → shared space)
        self.mag_proj = nn.Linear(embed_dim, projection_dim)
        self.flow_proj = nn.Linear(embed_dim, projection_dim)
        self.text_proj = nn.Linear(embed_dim, projection_dim)

    def forward(
            self,
            F_mag: Tensor,
            F_flow: Tensor,
            f_text: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Args:
            F_mag:  [B, D, H', W'] spatial magnification features
            F_flow: [B, D, H', W'] spatial optical flow features
            f_text: [B, D] text features (None during testing)

        Returns:
            f_mag_proj:  [B, projection_dim]
            f_flow_proj: [B, projection_dim]
            f_text_proj: [B, projection_dim] or None
        """
        # Pool and project visual features
        f_mag_proj = self.mag_proj(self.mag_pool(F_mag))
        f_flow_proj = self.flow_proj(self.flow_pool(F_flow))

        f_text_proj = None
        if f_text is not None:
            f_text_proj = self.text_proj(f_text)

        return f_mag_proj, f_flow_proj, f_text_proj


# =============================================================================
# Classifier HEAD
# =============================================================================
class Classifier(nn.Module):
    """
    Fuse visual and text features to get classification logits.

    Mag and Flow:
    Concat two modalities and project back to embedding dimension
    Global average pooling (spatial → vector)
    Pass through MLP to get classification logits.

    Both outputs are L2-normalized for cosine similarity in SASA loss.

    Args:
        embed_dim: Input feature dimension (512)
        spatial_size: Spatial resolution of visual features (7 for 224/32)
        num_classes: Number of output classes
    """

    def __init__(self, embed_dim: int = 512, spatial_size: int = 7, num_classes: int = 7):
        super().__init__()
        fusion_dim = embed_dim * 2

        # Project back to fusion dim
        self.projection = nn.Sequential(
            nn.Conv2d(fusion_dim, embed_dim, kernel_size=3, padding=1),  # (B, 2D, H', W') -> (B, D, H', W')
            nn.GELU(),
            # nn.Dropout(0.1),
        )

        # GAP
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification header
        self.mlp = nn.Sequential(
            Dynamic_erf(embed_dim, channels_last=True),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, F_mag: Tensor, F_flow: Tensor) -> Tensor:
        """
        Args:
            F_mag:  [B, D, H', W'] spatial magnification features
            F_flow: [B, D, H', W'] spatial optical flow features

        Returns:
            logits:  [B, num_classes]
        """
        F_fused = self.projection(torch.cat([F_mag, F_flow], dim=1))  # (B, 2D, H', W') -> (B, D, H', W')
        f_fused = torch.flatten(self.gap(F_fused), start_dim=1)  # (B, D, H', W') -> (B, D, 1, 1) -> (B, D)
        logits = self.mlp(f_fused)  # (B, D) -> (B, num_classes)

        return logits


# =============================================================================
# SAC^2-Net
# =============================================================================

class SAC2Net(nn.Module):
    """
    Multimodal Micro-expression Recognition Model.

    Architecture:
        ┌─────────────────────────────────────────────────┐
        │  Training Mode                                  │
        │                                                 │
        │  mag_images ──→ HFE ──┐                         │
        │                       ├──→ SASA Proj ──→ f_proj │ ← for SASA loss
        │                       ├──→ CCF ──→ logits       │ ← for Classification loss
        │  flow_images ─→ HFE ──┘                         │
        │                                                 │
        │  texts ──→ CLIP ──→ SASA Proj ──→ f_text        │ ← for SASA loss
        └─────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────┐
        │  Testing Mode                                   │
        │                                                 │
        │  mag_images ──→ HFE ──┐                         │
        │                       ├──→ CCF ──→ logits       │
        │  flow_images ─→ HFE ──┘                         │
        │                                                 │
        │  (no text encoder, no SASA projection)          │
        └─────────────────────────────────────────────────┘

    Args:
        num_classes: Number of emotion categories
        embed_dim: Feature dimension (must match HFE output channels)
        projection_dim: SASA projection space dimension
        spatial_size: Spatial resolution of encoder output (H/32)
        freeze_text_encoder: Whether to freeze CLIP weights
        clip_model_name: CLIP model variant identifier
        ccf_config: Configuration for CCF fusion module (None uses defaults)
    """

    def __init__(
            self,
            num_classes: int = 7,
            embed_dim: int = 512,
            projection_dim: int = 512,
            spatial_size: int = 7,
            freeze_text_encoder: bool = True,
            clip_model_name: str = "openai/clip-vit-base-patch32",
            ccf_config: Optional[dict] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # =====================================================================
        # 1. Visual Encoders (separate weights for each modality)
        # =====================================================================
        self.mag_encoder = hybrid_fast_encoder()
        self.flow_encoder = hybrid_fast_encoder()

        # =====================================================================
        # 2. Text Encoder (frozen CLIP)
        # =====================================================================
        self.text_encoder = TextEncoder(
            model_name=clip_model_name,
            freeze=freeze_text_encoder,
        )

        # =====================================================================
        # 3. SASA Projection Head (for contrastive alignment during training)
        # =====================================================================
        self.sasa_head = SASAProjectionHead(
            embed_dim=embed_dim,
            projection_dim=projection_dim,
            spatial_size=spatial_size,
        )

        # =====================================================================
        # 4. Complement-Consensus Fusion Module
        # =====================================================================
        _ccf_cfg = CCFConfig(
            **(ccf_config or {}),
        )
        self.fusion_module = CCFusion(_ccf_cfg)

        # =====================================================================
        # 5. Classifier
        # =====================================================================
        self.classifier = Classifier(embed_dim, spatial_size, num_classes)

        # =====================================================================
        # 6. Weight Initialization (EXCLUDING frozen modules)
        # =====================================================================
        self._init_weights()

        # Store config for serialization
        self.config = {
            'num_classes': num_classes,
            'embed_dim': embed_dim,
            'projection_dim': projection_dim,
            'spatial_size': spatial_size,
            'freeze_text_encoder': freeze_text_encoder,
            'clip_model_name': clip_model_name,
            'ccf_config': ccf_config,
        }

    # =========================================================================
    # WEIGHT INITIALIZATION
    # =========================================================================

    def _init_weights(self):
        """
        Initialize weights for trainable modules ONLY.

        CRITICAL: Excludes the frozen CLIP text encoder to preserve
        pretrained knowledge. Also excludes any module with
        requires_grad=False on all parameters.
        """
        # Modules to skip during initialization
        frozen_modules = {self.text_encoder}

        for name, module in self.named_modules():
            # Skip the frozen text encoder entirely
            if any(module is fm or self._is_child_of(module, fm) for fm in frozen_modules):
                continue

            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                # Fan-out initialization for conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def _is_child_of(module: nn.Module, parent: nn.Module) -> bool:
        """Check if module is a descendant of parent."""
        for child in parent.modules():
            if child is module:
                return True
        return False

    # =========================================================================
    # ENCODING METHODS
    # =========================================================================

    def encode_magnified(self, images: Tensor) -> Tensor:
        """Encode magnified images → [B, D, H', W']."""
        return self.mag_encoder(images)

    def encode_flow(self, images: Tensor) -> Tensor:
        """Encode optical flow images → [B, D, H', W']."""
        return self.flow_encoder(images)

    def encode_text(self, texts: List[str]) -> Tensor:
        """Encode text prompts → [B, D]. Only called during training."""
        device = next(self.mag_encoder.parameters()).device
        return self.text_encoder(texts, device=device)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(
            self,
            mag_images: Tensor,
            flow_images: Tensor,
            texts: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with automatic train/test mode switching.

        Training mode (model.train()):
            - Requires: mag_images, flow_images, texts
            - Returns: logits, SASA projection features, reliability maps

        Testing mode (model.eval()):
            - Requires: mag_images, flow_images (texts ignored)
            - Returns: logits only

        Args:
            mag_images:  [B, 3, H, W] magnified expression images
            flow_images: [B, 3, H, W] optical flow images
            texts:       List of B AU-description strings (training only)

        Returns:
            outputs: Dict with keys depending on mode:
                Always:
                    'logits':         [B, num_classes]
                Training only:
                    'f_mag_proj':     [B, projection_dim]
                    'f_flow_proj':    [B, projection_dim]
                    'f_text_proj':    [B, projection_dim]
                    'reliability_mag':  List of [B, 1, H', W'] per CCF block
                    'reliability_flow': List of [B, 1, H', W'] per CCF block
        """
        # ==== Step 1: Visual Encoding ====
        F_mag = self.encode_magnified(mag_images)  # [B, D, H', W']
        F_flow = self.encode_flow(flow_images)  # [B, D, H', W']

        # ==== Step 2: CCF Fusion → Logits ====
        F_mag_fused, F_flow_fused, fusion_aux = self.fusion_module(F_mag, F_flow)
        logits = self.classifier(F_mag_fused, F_flow_fused)

        outputs = {'logits': logits}

        # ==== Step 3: SASA features (training only) ====
        if self.training:
            # Validate that texts are provided during training
            if texts is None:
                raise ValueError(
                    "Text prompts are required during training for SASA loss. "
                    "Pass texts=<list of AU descriptions> to forward()."
                )

            # Encode text
            f_text = self.encode_text(texts)  # [B, D]

            # Project all three modalities into SASA alignment space
            f_mag_proj, f_flow_proj, f_text_proj = self.sasa_head(F_mag, F_flow, f_text)

            outputs.update({
                'f_mag_proj': f_mag_proj,
                'f_flow_proj': f_flow_proj,
                'f_text_proj': f_text_proj,
                # Pass through reliability maps for potential auxiliary losses
                'reliability_mag': fusion_aux.get('reliability_mag', []),
                'reliability_flow': fusion_aux.get('reliability_flow', []),
            })

        return outputs

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def get_trainable_param_groups(self, lr: float = 1e-4, lr_mult_encoder: float = 0.2):
        """
        Get parameter groups with different learning rates.

        Useful for fine-tuning where visual encoders use a lower LR
        than the fusion module and projection heads.

        Args:
            lr: Base learning rate (for fusion + projection)
            lr_mult_encoder: Multiplier for visual encoder LR

        Returns:
            List of param groups for optimizer
        """
        encoder_params = []
        other_params = []

        # Separate visual encoder params from other trainable params
        encoder_module_names = {'mag_encoder', 'flow_encoder'}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if any(name.startswith(enc) for enc in encoder_module_names):
                encoder_params.append(param)
            else:
                other_params.append(param)

        return [
            {'params': encoder_params, 'lr': lr * lr_mult_encoder, 'name': 'visual_encoders'},
            {'params': other_params, 'lr': lr, 'name': 'fusion_and_heads'},
        ]

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        components = {
            'mag_encoder': self.mag_encoder,
            'flow_encoder': self.flow_encoder,
            'text_encoder': self.text_encoder,
            'sasa_head': self.sasa_head,
            'fusion_module': self.fusion_module,
        }
        counts = {}
        for name, module in components.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {'total': total, 'trainable': trainable}

        counts['_total'] = {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        return counts

    # =========================================================================
    # SAVE AND LOAD METHODS
    # =========================================================================

    def save_pretrained(self, save_directory: str, save_name: str = "SAC2Net"):
        """
        Save model weights and config.

        Args:
            save_directory: Directory to save model
            save_name: Base name for saved files

        Saves:
            - {save_name}_config.pt: Model configuration
            - {save_name}_weights.pt: Model weights
            - {save_name}_full.pt: Full model (config + weights)
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_path = os.path.join(save_directory, f"{save_name}_config.pt")
        torch.save(self.config, config_path)

        # Save weights
        weights_path = os.path.join(save_directory, f"{save_name}_weights.pt")
        torch.save(self.state_dict(), weights_path)

        # Save full model (config + weights)
        full_path = os.path.join(save_directory, f"{save_name}_full.pt")
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, full_path)

        print(f"Model saved to {save_directory}/")
        print(f"  - Config: {save_name}_config.pt")
        print(f"  - Weights: {save_name}_weights.pt")
        print(f"  - Full: {save_name}_full.pt")

    @classmethod
    def from_pretrained(
            cls,
            save_directory: str,
            save_name: str = "SAC2Net",
            device: Optional[torch.device] = None
    ):
        """
        Load model from saved checkpoint.

        Args:
            save_directory: Directory containing saved model
            save_name: Base name of saved files
            device: Device to load weights on

        Returns:
            Loaded SAC2Net instance
        """
        device = device or torch.device("cpu")
        full_path = os.path.join(save_directory, f"{save_name}_full.pt")

        if os.path.exists(full_path):
            # Load from full checkpoint
            checkpoint = torch.load(full_path, map_location=device)
            config = checkpoint['config']
            state_dict = checkpoint['state_dict']
        else:
            # Load from separate files
            config_path = os.path.join(save_directory, f"{save_name}_config.pt")
            weights_path = os.path.join(save_directory, f"{save_name}_weights.pt")

            config = torch.load(config_path, map_location=device)
            state_dict = torch.load(weights_path, map_location=device)

        # Create model with saved config
        model = cls(**config)
        model.load_state_dict(state_dict)

        print(f"Model loaded from {save_directory}/")
        return model

    def save_checkpoint(
            self,
            save_path: str,
            optimizer: Optional[torch.optim.Optimizer] = None,
            epoch: int = 0,
            best_metric: float = 0.0,
            **kwargs
    ):
        """
        Save training checkpoint.

        Args:
            save_path: Path to save checkpoint
            optimizer: Optimizer to save state
            epoch: Current epoch
            best_metric: Best metric value
            **kwargs: Additional items to save
        """
        checkpoint = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'epoch': epoch,
            'best_metric': best_metric,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        checkpoint.update(kwargs)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    @classmethod
    def load_checkpoint(
            cls,
            checkpoint_path: str,
            optimizer: Optional[torch.optim.Optimizer] = None,
            device: Optional[torch.device] = None
    ) -> Tuple['SAC2Net', Dict]:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            optimizer: Optimizer to load state into
            device: Device to load weights on

        Returns:
            Tuple of (model, checkpoint_dict)
        """
        device = device or torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Create model
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Best metric: {checkpoint.get('best_metric', 'N/A')}")

        return model, checkpoint


# =============================================================================
# USAGE EXAMPLE AND VERIFICATION
# =============================================================================

def example_usage():
    """Demonstrates train/test mode difference."""
    print("=" * 70)
    print("SAC2Net — Train/Test Mode Verification")
    print("=" * 70)

    # --- Skip if no transformers ---
    if not HAS_TRANSFORMERS:
        print("\nSkipping: transformers library not installed.")
        print("Install with: pip install transformers")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = SAC2Net(num_classes=5, embed_dim=512).to(device)

    # Dummy inputs
    B = 4
    mag_images = torch.randn(B, 3, 224, 224, device=device)
    flow_images = torch.randn(B, 3, 224, 224, device=device)
    texts = [
        "The facial expression shows the brows pulled together with lip corners depressed.",
        "The eyelids are tightened and the nose is wrinkled.",
        "The brows are raised and the jaw drops open.",
        "The lip corners are pulled upward with cheeks raised.",
    ]

    # --- Parameter counts ---
    """
    print("\n--- Parameter Counts ---")
    counts = model.count_parameters()
    for name, c in counts.items():
        trainable_str = f"trainable: {c['trainable']:>10,}"
        total_str = f"total: {c['total']:>10,}"
        print(f"  {name:<20s}: {total_str}  |  {trainable_str}")
    """

    # ===== TRAINING MODE =====
    print("\n--- Training Mode ---")
    model.train()
    outputs_train = model(mag_images, flow_images, texts=texts)

    print(f"  Output keys: {list(outputs_train.keys())}")
    print(f"  logits:      {outputs_train['logits'].shape}")
    print(f"  f_mag_proj:  {outputs_train['f_mag_proj'].shape}")
    print(f"  f_flow_proj: {outputs_train['f_flow_proj'].shape}")
    print(f"  f_text_proj: {outputs_train['f_text_proj'].shape}")
    print(f"  reliability_mag: {outputs_train['reliability_mag']}")
    print(f"  reliability_flow: {outputs_train['reliability_flow']}")

    def reliability_entropy_loss(R):
        """Encourages decisive reliability maps (push away from 0.5)."""
        return -(R * torch.log(R) + (1 - R) * torch.log(1 - R)).mean()

    loss = torch.zeros(1).to(device)
    for R_mag, R_flow in zip(outputs_train['reliability_mag'], outputs_train['reliability_flow']):
        loss_R_mag = reliability_entropy_loss(R_mag)
        loss_R_flow = reliability_entropy_loss(R_flow)
        loss += loss_R_mag + loss_R_flow

    loss.backward()

    # Verify L2 normalization
    """
    mag = outputs_train['f_mag_proj']
    mag = F.normalize(mag, dim=-1)
    # mag_norms = outputs_train['f_mag_proj'].norm(dim=-1)
    mag_norms = mag.norm(dim=-1)
    print(f"  f_mag_proj L2 norms: {mag_norms.tolist()} (should be ~1.0)")
    """

    # Verify gradient flow (text encoder should have no gradients)
    """
    loss = outputs_train['logits'].sum()
    loss.backward()
    text_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.text_encoder.parameters()
    )
    print(f"  Text encoder has gradients: {text_has_grad} (should be False)")

    # Verify training mode raises error without texts
    try:
        model(mag_images, flow_images, texts=None)
        print("  ERROR: Should have raised ValueError without texts!")
    except ValueError as e:
        print(f"  Correctly raised ValueError when texts=None: OK")
    """

    # ===== TESTING MODE =====
    """
    print("\n--- Testing Mode ---")
    model.eval()
    with torch.no_grad():
        outputs_test = model(mag_images, flow_images)  # No texts needed!

    print(f"  Output keys: {list(outputs_test.keys())}")
    print(f"  logits:      {outputs_test['logits'].shape}")
    assert 'f_mag_proj' not in outputs_test, "SASA features should not be in test output!"
    assert 'f_text_proj' not in outputs_test, "Text features should not be in test output!"
    print(f"  No SASA features in output: OK")
    """

    # ===== TRAINING LOOP EXAMPLE =====
    """
    print("\n--- Example Training Step ---")
    model.train()
    optimizer = torch.optim.AdamW(model.get_trainable_param_groups(lr=1e-4))

    # Simulate one training step
    outputs = model(mag_images, flow_images, texts=texts)
    ce_loss = F.cross_entropy(outputs['logits'], torch.randint(0, 5, (B,), device=device))

    # SASA loss would use: outputs['f_mag_proj'], outputs['f_flow_proj'], outputs['f_text_proj']
    # sasa_loss = compute_sasa_loss(outputs['f_mag_proj'], outputs['f_flow_proj'],
    #                               outputs['f_text_proj'], soft_labels)

    total_loss = ce_loss  # + sasa_loss + reliability_entropy_loss
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"  CE loss: {ce_loss.item():.4f}")
    print(f"  Training step completed successfully!")
    """

    # ===== EVALUATION LOOP EXAMPLE =====
    """
    print("\n--- Example Evaluation Step ---")
    model.eval()
    with torch.no_grad():
        outputs = model(mag_images, flow_images)  # No texts!
        preds = outputs['logits'].argmax(dim=-1)
        print(f"  Predictions: {preds.tolist()}")
        print(f"  Evaluation step completed successfully!")

    print(f"\n{'=' * 70}")
    print("All verifications passed!")
    print(f"{'=' * 70}")
    """


if __name__ == "__main__":
    example_usage()
