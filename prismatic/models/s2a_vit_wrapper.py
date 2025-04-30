"""Seg‑to‑Act vision backbone with mask‑aware cross‑attention fusion.

Adds a lightweight segmentation branch that turns per‑object masks + language
embeddings into **mask tokens**, then injects that information into the RGB
patch tokens through a *single* cross‑attention block (RGB‑M fuse).  Only the
*first* image is fused when multiple images are provided; other images pass
through unchanged so the output tensor keeps the same hidden width as the
original backbone.

The design choices keep every pretrained weight in the RGB ViT intact:
* global residual add (`rgb + attn_out`) ⇒ identity at initialisation
* hidden width (`vis_dim`) stays unchanged
* sequence length increases only when multi‑image inputs are used, exactly as
  in the original PrismaticVisionBackbone.
"""
from typing import Dict, Any

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

# -----------------------------------------------------------------------------
# 1.  Mask encoder – per‑object transformer + per‑patch attention pool
# -----------------------------------------------------------------------------

class PerObjectSemanticEncoder(nn.Module):
    """Turn a single binary mask + its language embedding into patch tokens."""

    def __init__(self, hidden: int, lang_dim: int,
                 use_lang):
        super().__init__()
        self.use_lang = use_lang
            
        # convert 224×224 → 256 patch tokens of size `hidden`
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=14, stride=14),
            nn.Flatten(2),                 # (B,C,256)
            Rearrange("b c n -> b n c"),  # (B,256,C)
            nn.LayerNorm(hidden),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=8, dim_feedforward=4*hidden,
                dropout=0.0, activation=nn.GELU(), batch_first=True, norm_first=True
            ),
            num_layers=4,
        )
        if use_lang:
            self.lang_proj = nn.Linear(lang_dim, hidden, bias=False)
            self.cross_attn = nn.MultiheadAttention(hidden, 8, batch_first=True)

    def forward(self, mask: torch.Tensor, lang: torch.Tensor=None, attn_mask: torch.Tensor=None) -> torch.Tensor:
        x = self.patch_embed(mask)                # (B,256,H)
        x = self.transformer(x)                   # (B,256,H)
        if self.use_lang:
            lang = self.lang_proj(lang)               # (B,L,H)
            x, _ = self.cross_attn(x, lang, lang, key_padding_mask=~attn_mask.bool())
        return x                              # (B,256,H)


class SemanticAttentionPooling(nn.Module):
    """Pool *N masks* → one token per patch via attention across the N dimension."""

    def __init__(self, hidden: int, heads: int = 8):
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (N_mask, 256, H)
        tokens = tokens.permute(1, 0, 2)          # (256,N,H)
        q = self.q(tokens.mean(dim=1, keepdim=True))
        k = self.k(tokens)
        v = self.v(tokens)
        pooled, _ = self.attn(q, k, v)            # (256,1,H)
        return pooled.squeeze(1)                  # (256,H)


class Seg2ActVisionTransformer(nn.Module):
    """Stack Per‑object encoder + pooling => per‑patch *mask tokens*."""

    def __init__(self, hidden: int, lang_dim: int,
                 use_lang: bool, merge_masks: bool):
        super().__init__()
        self.merge_masks = merge_masks
        
        self.obj_encoder = PerObjectSemanticEncoder(hidden, lang_dim, use_lang)
        if not merge_masks:
            self.pool = SemanticAttentionPooling(hidden)
        
    def get_num_patches(self) -> int:
        """Returns the number of vision patches output by the mask encoder."""
        return self.obj_encoder.patch_embed.num_patches

    def forward(self, masks, lang=None, att=None):
        obj_tokens = self.obj_encoder(masks, lang, att)  # (N,256,H)
        
        if self.merge_masks:
            return obj_tokens
        else:
            return self.pool(obj_tokens)                     # (256,H)

# -----------------------------------------------------------------------------
# 2.  RGB–Mask fusion via one‑way cross‑attention and residual add
# -----------------------------------------------------------------------------

class RGBMFuse(nn.Module):
    def __init__(self, rgb_dim: int, mask_dim: int, heads: int = 8):
        super().__init__()
        # self.mask_proj = nn.Linear(mask_dim, rgb_dim, bias=False)
        self.attn = nn.MultiheadAttention(rgb_dim, heads, batch_first=True)

    def forward(self, rgb: torch.Tensor, mask: torch.Tensor, pad: torch.Tensor | None = None):
        # mask = self.mask_proj(mask)                        # (B,256,D_rgb)
        out, _ = self.attn(rgb, mask, mask, key_padding_mask=pad)
        return rgb + out                                   # residual


# -----------------------------------------------------------------------------
# 3.  Mask projector to LLM dimension
# -----------------------------------------------------------------------------
class S2AMaskProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, mask_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(mask_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(mask_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features

# -----------------------------------------------------------------------------
# 3.  Prismatic backbone wrapper
# -----------------------------------------------------------------------------

class S2APrismaticVisionBackbone(nn.Module):
    """PrismaticVisionBackbone + segmentation goodies.

    Parameters
    ----------
    vision_backbone : PrismaticVisionBackbone
    use_fuse        : if True, inject mask info into the RGB tokens
    use_mask_token  : if True, return mask tokens projected to `llm_dim`

    **Constraint**
    --------------
    At least *one* of the two flags must be **True**.  A `(False, False)`
    configuration is disallowed because the module would become a no‑op.
    """

    def __init__(self, vision_backbone, llm_dim: int,
                 use_fuse: bool, use_mask_token: bool,
                 use_lang: bool, merge_masks: bool):
        super().__init__()
        if not (use_fuse or use_mask_token):
            raise ValueError(
                "S2APrismaticVisionBackbone: either `use_fuse` or `use_mask_token` must be True."
            )
            
        self.vision_backbone = vision_backbone
        self.llm_dim = llm_dim
        self.vis_dim = self._compute_vis_dim()
        
        self.use_fuse = use_fuse
        self.use_mask_token = use_mask_token
        
        self.use_lang = use_lang
        self.merge_masks = merge_masks

        self.mask_encoder = Seg2ActVisionTransformer(self.vis_dim, llm_dim, use_lang, merge_masks)
        if use_fuse:
            self.fuse = RGBMFuse(self.vis_dim, self.vis_dim)
        if use_mask_token:
            self.mask_projector = S2AMaskProjector(self._use_fused_vision_backbone, self.vis_dim, self.llm_dim)
            

    # -------------------------------- private helpers -------------------------
    def _compute_vis_dim(self) -> int:
        base = self.vision_backbone.get_head_feature_dim()
        if self._use_fused_vision_backbone():
            fused = self.vision_backbone.get_fused_head_feature_dim()
            return base + fused  # 1024 + 1152 = 2176 normally
        return base
    
    def _use_fused_vision_backbone(self):
        return self.vision_backbone.use_fused_vision_backbone
    

    # --------------------------------  helpers -------------------------------    
    def get_num_patches(self) -> int:
        """Returns the number of vision patches output by the vision backbone."""
        return self.vision_backbone.get_num_patches()
    
    def get_num_images_in_input(self) -> int:
        """Returns the number of input images for the vision backbone."""
        return self.vision_backbone.get_num_images_in_input()

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """Sets the number of input images for the vision backbone."""
        self.vision_backbone.set_num_images_in_input(num_images_in_input)


    # ------------------------------ forward -----------------------------------
    def forward(self, pixel_values: torch.Tensor, seg_masks_info: Dict[str, Any]):
        # ---- 1. build mask tokens --------------------------------------------
        if self.merge_masks:
            merged_mask = torch.stack(
                [m.float().amax(dim=0,)                             # (1,H,W) per image
                for m in seg_masks_info["pixel_values_seg_masks"]], #  ↑ logical-OR over Nᵢ
                dim=0                                               # (B,1,H,W)
            )
            mask_tokens = self.mask_encoder(merged_mask) # (B, 256, vis_dim)
        else:
            mask_tokens = torch.stack([
                self.mask_encoder(m, l, a)
                for m, l, a in zip(seg_masks_info["pixel_values_seg_masks"],
                                seg_masks_info["lang_ebd_seg_masks"],
                                seg_masks_info["lang_att_mask_seg_masks"])
            ])  # (B,256,vis_dim)

        # Optional LLM‑dim projection ------------------------------------------
        mask_llm = self.mask_projector(mask_tokens) if self.use_mask_token else None

        # ---- 2. build RGB tokens ---------------------------------------------
        num_imgs = self.get_num_images_in_input()
        if num_imgs == 1:
            rgb = self._encode_single(pixel_values)
            if self.use_fuse:
                rgb = self.fuse(rgb, mask_tokens)
            return (rgb, mask_llm) if self.use_mask_token else rgb

        # ---- 2b. multi‑image --------------------------------------------------
        assert self.vision_backbone.use_fused_vision_backbone, "Multi‑image requires fused backbone"
        chunks = pixel_values.split(6, dim=1)                # list len=num_imgs, each 6‑channel
        rgb_list = [self._encode_single(c) for c in chunks]
        if self.use_fuse:
            rgb_list[0] = self.fuse(rgb_list[0], mask_tokens)  # only first image fused
        rgb_all = torch.cat(rgb_list, dim=1)                  # (B,256*num_imgs,vis_dim)
        return (rgb_all, mask_llm) if self.use_mask_token else rgb_all

    # ----------------------------- rgb utility ---------------------------------
    def _encode_single(self, pixels: torch.Tensor) -> torch.Tensor:
        if not self._use_fused_vision_backbone:
            return self.vision_backbone.featurizer(pixels)          # (B,256,1024)
        img, img_fused = pixels.split([3, 3], dim=1)
        t1 = self.vision_backbone.featurizer(img)
        t2 = self.vision_backbone.fused_featurizer(img_fused)
        return torch.cat([t1, t2], dim=2)                           # (B,256,vis_dim)
