from functools import partial
from typing import List, Tuple, Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, trunc_normal_
from models.NormFuncs import Dynamic_erf


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(
            self,
            patch_size: int = 16,
            stride: int = 16,
            padding: int = 0,
            in_channels: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class CPE(nn.Module):
    """Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """

    def __init__(
            self,
            in_channels: int,
            embed_dim: int = 768,
            spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
    ) -> None:
        """Build conditional positional encoding

        Args:
            in_channels: Number of input channels.
            embed_dim: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
        """
        super(CPE, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple), (
            f'"spatial_shape" must by a sequence or int, '
            f"get {type(spatial_shape)} instead."
        )
        assert len(spatial_shape) == 2, (
            f'Length of "spatial_shape" should be 2, '
            f"got {len(spatial_shape)} instead."
        )

        self.pe = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=spatial_shape,
            stride=1,
            padding=int(spatial_shape[0] // 2),
            bias=True,
            groups=embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(x) + x
        return x


class RepMixer(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=3,
        **kwargs
    ):
        super().__init__()
        self.DWConv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            groups=dim,
        )

    def forward(self, x):
        return self.DWConv(x)


class MHSA(nn.Module):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
            self,
            dim: int,
            head_dim: int = 64,
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        """Build MHSA module that can handle 3D or 4D input tensors.

        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``64``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        if len(shape) == 4:
            x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)

        # qkv: [B, N, 3 * C] -> [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        # Permute to [B, num_heads, N, head_dim] for q, k, v
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # USE FLASH ATTENTION HERE
        # This replaces the manual scale, dot product, softmax, and dropout
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )

        # [B, num_heads, N, head_dim] -> [B, N, num_heads * head_dim]
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class ConvFFN(nn.Module):
    """Convolutional FFN Module."""

    def __init__(
            self,
            in_channels: int,
            hidden_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            drop: float = 0.0,
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("ln", Dynamic_erf(out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MetaFormerBlock(nn.Module):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
            self,
            dim: int,
            token_mixer: nn.Module = nn.Identity,
            mlp_ratio: float = 4.0,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = Dynamic_erf,
            drop: float = 0.0,
            drop_path: float = 0.0,
            use_layer_scale: bool = True,
            layer_scale_init_value: float = 1e-5,
    ):
        """Build Attention Block.

        Args:
            dim: Number of embedding dimensions.
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            norm_layer: Normalization layer. Default: ``LayerNorm``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """

        super().__init__()

        self.norm1 = norm_layer(dim) if norm_layer else nn.Identity()
        self.token_mixer = token_mixer(dim=dim)
        self.norm2 = norm_layer(dim) if norm_layer else nn.Identity()

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.convffn(self.norm2(x)))
        return x


def basic_blocks(
        dim: int,
        block_index: int,
        num_blocks: List[int],
        token_mixer: nn.Module = nn.Identity,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = Dynamic_erf,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
) -> nn.Sequential:
    """Build blocks for a stage.

    Args:
        dim: Number of embedding dimensions.
        block_index: block index.
        num_blocks: List containing number of blocks per stage.
        token_mixer: Token mixer type.
        mlp_ratio: MLP expansion ratio.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        drop_rate: Dropout rate.
        drop_path_rate: Drop path rate.
        use_layer_scale: Flag to turn on layer scale regularization.
        layer_scale_init_value: Layer scale value at initialization.

    Returns:
        nn.Sequential object of all the blocks within the stage.
    """
    blocks = []
    for idx in range(num_blocks[block_index]):
        block_dpr = drop_path_rate * (idx + sum(num_blocks[:block_index])) / (sum(num_blocks) - 1)
        blocks.append(
            MetaFormerBlock(
                dim=dim,
                token_mixer=token_mixer,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop=drop_rate,
                drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
        )
    blocks = nn.Sequential(*blocks)

    return blocks


class HybridFastEncoder(nn.Module):
    """
    This class implements 'HybridFastEncoder architecture' for visual feature extraction.
    """

    def __init__(
            self,
            layers,
            embed_dims=None,
            token_mixers=None,
            mlp_ratios=None,
            norm_layer: nn.Module = Dynamic_erf, act_layer: nn.Module = nn.GELU,
            pos_embs=None,
            downsamples=None, down_patch_size=3, down_stride=2,
            drop_rate=0.0, drop_path_rate=0.0,
            use_layer_scale=True,
            layer_scale_init_value=1e-5,
            pretrained=None,
            **kwargs,
    ) -> None:

        super().__init__()

        if pos_embs is None:
            pos_embs = [None] * len(layers)

        self.patch_embed = PatchEmbed(
            patch_size=7, stride=4, padding=2,
            in_channels=3, embed_dim=embed_dims[0])

        # Build the main stages of the network architecture
        network = []
        for i in range(len(layers)):
            # Add position embeddings if requested
            if pos_embs[i] is not None:
                network.append(pos_embs[i](embed_dims[i], embed_dims[i]))
            stage = basic_blocks(
                dim=embed_dims[i],
                block_index=i,
                num_blocks=layers,
                token_mixer=token_mixers[i],
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break

            # Patch merging/downsampling between stages.
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=int(down_patch_size // 2),
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                    )
                )

        self.network = nn.ModuleList(network)

        # Initial setting
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.network:
            x = block(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W], typically [B, 3, 224, 224]

        Returns:
            Feature tensor [B, 512, 7, 7]
        """
        # Input embedding
        x = self.forward_embedding(x)
        # through backbone
        x = self.forward_tokens(x)
        return x


def hybrid_fast_encoder(pretrained=False, **kwargs):
    """"Instantiate a HybridFastEncoder."""
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 256, 512]
    # embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(CPE, spatial_shape=(7, 7))]
    token_mixers = (RepMixer, RepMixer, RepMixer, MHSA)    # MHSA: Multi Head Self Attention
    model = HybridFastEncoder(
        layers,
        embed_dims=embed_dims,
        token_mixers=token_mixers,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model
