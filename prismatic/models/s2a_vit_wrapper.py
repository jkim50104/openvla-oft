"""Implementation of additional modules for the VLA's vision transformer."""

from functools import partial
from typing import Any, Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import VisionTransformer


class PerObjectSemanticEncoder(nn.Module):
    """Encodes each object's mask along with its semantic embedding."""

    def __init__(self, head_feature_dim, lang_dim):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, head_feature_dim, kernel_size=14, stride=14),
            nn.Flatten(2),
            Rearrange('b c n -> b n c'),
            nn.LayerNorm(head_feature_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=head_feature_dim,
            nhead=8,
            dim_feedforward=head_feature_dim * 4,
            dropout=0.0,
            activation=nn.GELU(),
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cross_attn = nn.MultiheadAttention(embed_dim=head_feature_dim, num_heads=8, batch_first=True)
        self.lang_proj = nn.Linear(lang_dim, head_feature_dim)

    def forward(self, mask, semantic_embedding):
        # mask: [B, 1, H, W], semantic_embedding: [B, lang_dim]
        x = self.patch_embed(mask)  # [B, N_patches, D]
        x = self.transformer(x)     # [B, N_patches, D]

        lang_context = self.lang_proj(semantic_embedding).unsqueeze(1)  # [B, 1, D]

        enriched_embedding, _ = self.cross_attn(query=lang_context, key=x, value=x)
        enriched_embedding = enriched_embedding.squeeze(1)  # [B, D]

        return enriched_embedding  # [B, D]


class SemanticAttentionPooling(nn.Module):
    """Pools multiple object embeddings into one fixed-dimensional embedding."""

    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))
        self.key_proj = nn.Linear(embed_dim, output_dim)
        self.value_proj = nn.Linear(embed_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, object_embeddings):
        # object_embeddings: [B, N, embed_dim]
        query = self.query.expand(object_embeddings.size(0), -1, -1)  # [B, 1, output_dim]
        keys = self.key_proj(object_embeddings)  # [B, N, output_dim]
        values = self.value_proj(object_embeddings)  # [B, N, output_dim]

        attn_weights = self.softmax(query @ keys.transpose(-2, -1))  # [B, 1, N]
        pooled_embedding = attn_weights @ values  # [B, 1, output_dim]

        return pooled_embedding.squeeze(1)  # [B, output_dim]


class Seg2ActWrapper(nn.Module):
    """Complete wrapper combining RGB embeddings with semantic-aware mask embeddings."""

    def __init__(self, head_feature_dim, lang_dim, final_embed_dim):
        super().__init__()
        self.object_encoder = PerObjectSemanticEncoder(head_feature_dim, lang_dim)
        self.attention_pool = SemanticAttentionPooling(head_feature_dim, head_feature_dim)
        self.final_proj = nn.Linear(head_feature_dim * 2, final_embed_dim)

    def forward(self, rgb_embedding, masks, semantic_embeddings):
        """
        Args:
            rgb_embedding: [B, D_rgb]
            masks: [B, N, 1, H, W] - N masks per batch
            semantic_embeddings: [B, N, lang_dim] - N semantic embeddings per batch
        """
        B, N = masks.shape[:2]

        object_embeds = []
        for i in range(N):
            mask = masks[:, i]  # [B, 1, H, W]
            sem_emb = semantic_embeddings[:, i]  # [B, lang_dim]
            obj_embed = self.object_encoder(mask, sem_emb)  # [B, D]
            object_embeds.append(obj_embed.unsqueeze(1))

        object_embeds = torch.cat(object_embeds, dim=1)  # [B, N, D]

        pooled_semantic_embedding = self.attention_pool(object_embeds)  # [B, D]

        final_visual_embedding = torch.cat([rgb_embedding, pooled_semantic_embedding], dim=-1)  # [B, 2*D]
        final_embedding = self.final_proj(final_visual_embedding)  # [B, final_embed_dim]

        return final_embedding  # [B, final_embed_dim]


def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    """Utility function for monkey-patching functions."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


class FiLMedVisionTransformer(VisionTransformer):
    """
    Wrapper for timm.models.vision_transformer.VisionTransformer that overrides functions to enable infusing language
    embeddings into visual embeddings via FiLM.
    """

    def _intermediate_layers(
        self,
        x: torch.Tensor,
        language_embeddings: torch.Tensor,
        n: Union[int, Sequence] = 1,
    ):
        """
        Copy of timm.models.vision_transformer.VisionTransformer._intermediate_layers() with modifications
        to take in language embeddings as additional input.
        """
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, language_embeddings)  # Modified to receive language_embeddings
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        language_embeddings: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """
        Copy of timm.models.vision_transformer.VisionTransformer.get_intermediate_layers() with modifications
        to allow language embeddings as additional input.
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, language_embeddings, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)


class FiLMedPrismaticVisionBackbone(nn.Module):
    """
    Wrapper for OpenVLA's vision backbone that implements feature-wise linear modulation (FiLM).

    Wraps the Vision Transformers in the vision backbone to enable language conditioning through FiLM.
    Supports processing 1-3 images using dual vision backbones (SigLIP + DINOv2).
    """

    def __init__(
        self,
        vision_backbone,
        llm_dim: int = 4096,  # 4096 for Llama-2 7B
    ) -> None:
        """
        Initializes FiLM wrapper.

        Args:
            vision_backbone (PrismaticVisionBackbone): Base vision backbone.
            llm_dim (int): Dimension of language model embeddings.
        """
        super().__init__()
        self.vision_backbone = vision_backbone
        self.llm_dim = llm_dim

        # Wrap vision transformers
        self._wrap_vit(self.vision_backbone.featurizer)  # SigLIP
        if self.vision_backbone.use_fused_vision_backbone:
            self._wrap_vit(self.vision_backbone.fused_featurizer)  # DINOv2

    def _wrap_vit(self, vit) -> None:
        """
        Creates wrapper around an individual vision transformer to allow for infusion of language inputs.

        Args:
            vit (VisionTransformer): Original vision transformer.
        """
        # Wrap vision transformer blocks
        block_wrappers = []
        for block in vit.blocks:
            block_wrappers.append(
                FiLMedVisionTransformerBlock(block=block, vision_dim=vit.num_features, llm_dim=self.llm_dim)
            )
        vit.blocks = nn.Sequential(*block_wrappers)

        # Wrap vision transformer with new class that overrides functions used for forward pass
        vit.__class__ = FiLMedVisionTransformer
        vit.forward = unpack_tuple(partial(vit.get_intermediate_layers, n={len(vit.blocks) - 2}))

    def get_num_patches(self) -> int:
        """Returns the number of vision patches output by the vision backbone."""
        return self.vision_backbone.get_num_patches()

    def get_num_images_in_input(self) -> int:
        """Returns the number of input images for the vision backbone."""
        return self.vision_backbone.get_num_images_in_input()

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """Sets the number of input images for the vision backbone."""
        self.vision_backbone.set_num_images_in_input(num_images_in_input)

    def forward(self, pixel_values: torch.Tensor, language_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone with FiLM to infuse language inputs into visual features.

        Identical to PrismaticVisionBackbone.forward() except that language embeddings are also used as input.

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, C, H, W).
            language_embeddings (torch.Tensor): Language embeddings for the task description, (B, seq_len, llm_dim).
        """
        # For FiLM: Average the language embeddings of the task description
        average_language_embedding = language_embeddings.mean(dim=1)

        if self.get_num_images_in_input() == 1:
            if not self.vision_backbone.use_fused_vision_backbone:
                return self.vision_backbone.featurizer(pixel_values, average_language_embedding)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches = self.vision_backbone.featurizer(img, average_language_embedding)
            patches_fused = self.vision_backbone.fused_featurizer(img_fused, average_language_embedding)

            return torch.cat([patches, patches_fused], dim=2)

        else:
            assert self.vision_backbone.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            # Split `pixel_values` into individual images (each with 6 channels: 3 for SigLIP + 3 for DINOv2)
            images = torch.split(pixel_values, [6] * self.get_num_images_in_input(), dim=1)

            # Process each image and collect patches
            all_patches = []
            for img in images:
                # Split each image further into two stacks of channels (each with 3 channels)
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                # Get patches from both SigLIP and DINOv2 vision transformers
                patches = self.vision_backbone.featurizer(img_regular, average_language_embedding)
                patches_fused = self.vision_backbone.fused_featurizer(img_fused, average_language_embedding)

                # Concatenate SigLIP and DINOv2 patches along the hidden dimension
                combined_patches = torch.cat([patches, patches_fused], dim=2)
                all_patches.append(combined_patches)

            # Concatenate all patches along the patch dimension
            return torch.cat(all_patches, dim=1)
