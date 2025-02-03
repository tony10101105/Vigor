# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Encoder-decoder transformer layers for self/cross attention."""

from copy import deepcopy
import numpy as np
import einops
import torch
from torch import nn
from .utils import get_activation_fn


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def forward(
        self, tgt, memory,
        tgt_mask = None,
        memory_mask = None,
        tgt_key_padding_mask = None,
        memory_key_padding_mask = None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices

class MultiHeadAttentionSpatial(nn.Module):
    def __init__(
        self, d_model, n_head, dropout=0.1,
    ):
        super().__init__()
        assert d_model % n_head == 0, 'd_model: %d, n_head: %d' %(d_model, n_head)

        self.n_head = n_head
        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.spatial_dim = 5

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.spatial_n_head = n_head
        self.lang_cond_fc = nn.Linear(d_model, self.spatial_n_head * (self.spatial_dim + 1))

    def forward(self, q, k, v, pairwise_locs, key_padding_mask=None, txt_embeds=None):
        residual = q
        q = einops.rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = einops.rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
        v = einops.rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])

        spatial_weights = self.lang_cond_fc(residual + txt_embeds.unsqueeze(1))
        spatial_weights = einops.rearrange(spatial_weights, 'b l (h d) -> h b l d', h=self.spatial_n_head, d=self.spatial_dim+1)
        if self.spatial_n_head == 1:
            spatial_weights = einops.repeat(spatial_weights, '1 b l d -> h b l d', h=self.n_head)
        spatial_bias = spatial_weights[..., :1]
        spatial_weights = spatial_weights[..., 1:]

        loc_attn = torch.einsum('hbld,bltd->hblt', spatial_weights.float(), pairwise_locs.float()) + spatial_bias
        loc_attn = torch.sigmoid(loc_attn)

        if key_padding_mask is not None:
            mask = einops.repeat(key_padding_mask, 'b t -> h b l t', h=self.n_head, l=q.size(2))
            attn = attn.masked_fill(mask, -np.inf)
            loc_attn = loc_attn.masked_fill(mask, 0)

        fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
        fused_attn = torch.softmax(fused_attn, 3)
        
        assert torch.sum(torch.isnan(fused_attn) == 0), print(fused_attn)

        output = torch.einsum('hblt,hbtv->hblv', fused_attn, v)
        output = einops.rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, fused_attn
    
class TransformerSpatialDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
    ):
        super().__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        del self.self_attn
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout, 
        )

    def forward(
        self, tgt, memory, tgt_pairwise_locs,
        tgt_mask = None,
        memory_mask = None,
        tgt_key_padding_mask = None,
        memory_key_padding_mask = None,
    ):

        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, tgt2, tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask,
            txt_embeds=memory[:, 0],
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices

class RefEcoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()
        
        # Self attention for obj_embeds
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # cross attention for obj_embeds and cat(txt_embeds, mentioned_obj_txt_embeds)
        self.cross_a = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout_a = nn.Dropout(dropout)
        self.norm_a = nn.LayerNorm(d_model)

        # cross attention for masked_obj_embeds and masked_obj_embeds
        self.cross_b = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout_b = nn.Dropout(dropout)
        self.norm_b = nn.LayerNorm(d_model)

        # cross attention for obj_embeds and masked_obj_embeds
        self.cross_c = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout_c = nn.Dropout(dropout)
        self.norm_c = nn.LayerNorm(d_model)

        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = get_activation_fn(activation)

    def forward(self, obj_embeds, masked_obj_embeds, txt_embeds, mentioned_obj_txt_embeds):
        # obj_embeds:  torch.Size([52, B*view, 768])
        # masked_obj_embeds:  torch.Size([52, B*view, 768])
        # txt_embeds:  torch.Size([19, B*view, 768])
        # mentioned_obj_txt_embeds:  torch.Size([4, B*view, 768])
        obj_embeds_self = self.self_attn(
            obj_embeds, obj_embeds, obj_embeds,
            attn_mask=None,
            key_padding_mask=None
        )[0]
        obj_embeds = self.norm1(obj_embeds + self.dropout1(obj_embeds_self))

        ensembled_lang = torch.cat((txt_embeds, mentioned_obj_txt_embeds),dim=0)
        obj_embeds_self = self.cross_a(
            query=obj_embeds,
            key=ensembled_lang,
            value=ensembled_lang,
            attn_mask=None,
            key_padding_mask=None  # (B, L)
        )[0]
        obj_embeds = self.norm_a(obj_embeds + self.dropout_a(obj_embeds_self))

        target_feats = self.cross_b(
            query=masked_obj_embeds,
            key=mentioned_obj_txt_embeds,
            value=mentioned_obj_txt_embeds,
            attn_mask=None,
            key_padding_mask=None  # (B, L)
        )[0]
        masked_obj_embeds = self.norm_b(masked_obj_embeds + self.dropout_b(target_feats))
        final_feats = self.cross_c(
            query=obj_embeds,
            key=masked_obj_embeds,
            value=masked_obj_embeds,
            attn_mask=None,
            key_padding_mask=None  # (B, L)
        )[0]
        obj_embeds = self.norm_c(obj_embeds + self.dropout_c(final_feats))

        obj_embeds = self.norm2(obj_embeds + self.ffn(obj_embeds)).transpose(0, 1)
        return obj_embeds