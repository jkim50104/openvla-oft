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

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=head_feature_dim,
                nhead=8,
                dim_feedforward=head_feature_dim*4,
                dropout=0.0,
                activation=nn.GELU(),
                batch_first=True,
                norm_first=True
            ),
            num_layers=4
        )
        
        self.lang_proj = nn.Linear(lang_dim, head_feature_dim)#, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=head_feature_dim, num_heads=8, batch_first=True)

    def forward(self, mask, semantic_embedding, attention_mask):
        # mask: [Mask Num, 1, H, W], semantic_embedding: [Mask Num, lang_seg_len, lang_dim]
        x = self.patch_embed(mask)
        x = self.transformer(x)     # [Mask Num, vis_seq_len, D]

        semantic_embedding = self.lang_proj(semantic_embedding)  # [Mask Num, lang_seg_len, D]
        enriched_seg_masks_embedding, _ = self.cross_attn(query=x,
                                                          key=semantic_embedding, value=semantic_embedding,
                                                          key_padding_mask=~attention_mask.bool())

        return enriched_seg_masks_embedding  # [Mask Num, vis_seq_len, D]


class SemanticAttentionPooling(nn.Module):
    """Pools multiple mask embeddings patchwise into one per-patch embedding."""

    def __init__(self, embed_dim, output_dim, n_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, output_dim)
        self.key_proj = nn.Linear(embed_dim, output_dim)
        self.value_proj = nn.Linear(embed_dim, output_dim)
        self.attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=n_heads, batch_first=True)

    def forward(self, mask_embeddings):
        """
        Args:
            mask_embeddings: [Num_Masks, Vis_seq_len, embed_dim]

        Returns:
            pooled_embeddings: [Vis_seq_len, output_dim]
        """

        # Transpose to [Vis_seq_len, Num_Masks, embed_dim]
        mask_embeddings = mask_embeddings.permute(1, 0, 2)  # [S, N, D]

        # Project keys and values
        keys = self.key_proj(mask_embeddings)    # [S, N, output_dim]
        values = self.value_proj(mask_embeddings)  # [S, N, output_dim]

        # Create query: use the average of masks as initial query per patch
        query = mask_embeddings.mean(dim=1, keepdim=True)  # [S, 1, embed_dim]
        query = self.query_proj(query)                     # [S, 1, output_dim]

        # Attention: each patch's query attends over the N masks
        attn_output, _ = self.attn(
            query=query,  # [S, 1, output_dim]
            key=keys,     # [S, N, output_dim]
            value=values  # [S, N, output_dim]
        )  # attn_output: [S, 1, output_dim]

        # Remove the singleton middle dim
        pooled_embeddings = attn_output.squeeze(1)  # [S, output_dim]

        return pooled_embeddings
    

class Seg2ActVisionTransformer(nn.Module):
    """Complete wrapper for semantic-aware mask embeddings."""

    def __init__(self, head_feature_dim, lang_dim):
        super().__init__()
        self.seg_masks_encoder = PerObjectSemanticEncoder(head_feature_dim, lang_dim)        
        self.attention_pool = SemanticAttentionPooling(head_feature_dim, head_feature_dim)

    def forward(self, seg_masks, masks_lang_embd, lang_att_mask):
        """
        Args:
            seg_masks: [N, 1, H, W] - N masks
            semantic_embeddings: [N, lang_seq_len, lang_dim] - semantic embeddings per N masks
        """
        seg_masks_embed = self.seg_masks_encoder(seg_masks, masks_lang_embd, lang_att_mask)
        agg_seg_masks_embed = self.attention_pool(seg_masks_embed)
        
        return agg_seg_masks_embed


class S2AProjector(nn.Module):
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

        self.s2a_net = Seg2ActVisionTransformer(self.get_head_feature_dim(), llm_dim)
        
    def get_head_feature_dim(self) -> int:
        """Returns the number of head_feature_dim by the vision backbone."""
        return self.vision_backbone.get_head_feature_dim()
    
    def get_num_patches(self) -> int:
        """Returns the number of vision patches output by the vision backbone."""
        return self.vision_backbone.get_num_patches()

    def get_num_images_in_input(self) -> int:
        """Returns the number of input images for the vision backbone."""
        return self.vision_backbone.get_num_images_in_input()

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """Sets the number of input images for the vision backbone."""
        self.vision_backbone.set_num_images_in_input(num_images_in_input)

    def forward(self, pixel_values: torch.Tensor, seg_masks_info: dict) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone with FiLM to infuse language inputs into visual features.

        Identical to PrismaticVisionBackbone.forward() except that language embeddings are also used as input.

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, Num_masks, C, H, W).
            language_embeddings (torch.Tensor): Language embeddings for the task description, (B, Num_masks, llm_dim).
        """
        pixel_values_seg_masks = seg_masks_info["pixel_values_seg_masks"] #(B, num_masks, 1, 224,224)
        lang_ebd_seg_masks = seg_masks_info["lang_ebd_seg_masks"] #(B, num_masks, lang_seq_len, 4096)
        lang_att_mask_seg_masks = seg_masks_info["lang_att_mask_seg_masks"] #(B, num masks, lang_seq_len        
        
        patches_seg_masks = [self.s2a_net(vis_batch, lang_batch, att_batch) for (vis_batch, lang_batch, att_batch) in zip(pixel_values_seg_masks, lang_ebd_seg_masks, lang_att_mask_seg_masks)] # (B, num_masks, 256, 1024)
        patches_seg_masks = torch.stack(patches_seg_masks) # (B, 256, 1024)

        if self.get_num_images_in_input() == 1:

            if not self.vision_backbone.use_fused_vision_backbone:
                patches = self.vision_backbone.featurizer(pixel_values)
                return torch.cat([patches, patches_seg_masks], dim=2)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches = self.vision_backbone.featurizer(img) # (B, 256, 1024)
            patches_fused = self.vision_backbone.fused_featurizer(img_fused) # (B, 256, 1152)
            
            return torch.cat([patches, patches_fused, patches_seg_masks], dim=2)
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
                patches = self.vision_backbone.featurizer(img_regular)
                patches_fused = self.vision_backbone.fused_featurizer(img_fused)

                # Concatenate SigLIP and DINOv2 patches along the hidden dimension
                combined_patches = torch.cat([patches, patches_fused], dim=2)
                all_patches.append(combined_patches)

            # Concatenate all image patches along patch dimension
            all_patches = torch.cat(all_patches, dim=1)  # (B, 256*num_images, hidden_dim)

            # Now concatenate patches_seg_masks
            final_patches = torch.cat([all_patches, patches_seg_masks], dim=2)  # (B, 256*num_images, hidden_dim + seg_dim)

            return final_patches