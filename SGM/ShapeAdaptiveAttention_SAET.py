# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Changes: Added ShapeAdaptiveAttnProcessor with token-based fg/bg masking and mask-aware dual attention blending.
import math
from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.utils import deprecate, logging
from diffusers.models.attention_processor import Attention,AttnProcessor

logger = logging.get_logger(__name__)  
# Check if n is a perfect square
def is_perfect_square(n):
    if n < 0:
        return False
    root = math.isqrt(n)  
    return root * root == n
class ShapeAdaptiveAttnProcessor:
    r"""
    ShapeAdaptiveAttnProcessor.
    """

    def __call__(
        self,
        attn: Attention, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: Optional[torch.Tensor] = None,  
        attention_mask: Optional[torch.Tensor] = None,  
        temb: Optional[torch.Tensor] = None,  
        ma_mask: Optional[torch.Tensor] = None, 
        ref_mask: Optional[torch.Tensor] = None,
        text_input_ids=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            processor=AttnProcessor()
            return processor(
                attn,  
                hidden_states, 
                encoder_hidden_states,  
                attention_mask, 
                temb,  
                *args,
                **kwargs,
                )


        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        with_token_id=593
        start_end_token_id=[49406,49407]

        # Generate foreground/background masks based on special token positions
        def generate_fg_bg_attention_masks(batch_text_input_ids, with_token_id, start_end_token_id,sequence_length, batch_size):
            

            input_ids = batch_text_input_ids["input_ids"] 

            with_mask = input_ids == with_token_id 
            indices = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            with_indices = with_mask * indices
            last_with_idx = with_indices.max(dim=1).values.unsqueeze(1)  


            arange = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            fg_mask_bool = arange < last_with_idx  
            mask_fg = fg_mask_bool.long()
            mask_bg = (~fg_mask_bool).long()


            for special_id in start_end_token_id:
                mask_fg[input_ids == special_id] = 1
                mask_bg[input_ids == special_id] = 1 

            mask_bg_tensor = attn.prepare_attention_mask(mask_bg , sequence_length-len(mask_fg[0]), batch_size)
            mask_fg_tensor = attn.prepare_attention_mask(mask_fg , sequence_length-len(mask_bg[0]), batch_size)

            return mask_fg_tensor, mask_bg_tensor


        residual = hidden_states


        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            

            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # Prepare attention masks to split focus into foreground and background
        mask_fg,mask_bg=generate_fg_bg_attention_masks(text_input_ids, with_token_id, start_end_token_id, sequence_length, batch_size)

        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        dtype = query.dtype
        device = query.device

        mask_fg = mask_fg.repeat(2, 1)  
        mask_bg = mask_bg.repeat(2, 1)

        bsz_heads = mask_fg.shape[0] 
        seq_len = query.shape[1]      
        enc_len = mask_fg.shape[1]  
        mask_fg = mask_fg.unsqueeze(1).expand(bsz_heads, seq_len, enc_len).to(dtype=dtype, device=device)
        mask_bg = mask_bg.unsqueeze(1).expand(bsz_heads, seq_len, enc_len).to(dtype=dtype, device=device)

        half = bsz_heads // 2
        mask_fg[:half] = 1
        mask_bg[:half] = 1

        # Apply attention separately using foreground and background masks
        attn_fg = attn.get_attention_scores(query, key, mask_fg)
        hidden_fg = torch.bmm(attn_fg, value)

        attn_bg = attn.get_attention_scores(query, key, mask_bg)
        hidden_bg = torch.bmm(attn_bg, value)

        hidden_fg = attn.batch_to_head_dim(hidden_fg)
        hidden_bg = attn.batch_to_head_dim(hidden_bg)

        if ma_mask is not None:

            if ma_mask.ndim == 4:
                ma_mask = F.interpolate(ma_mask.float(), size=(height, width), mode="nearest")
                ma_mask = ma_mask.view(batch_size, 1, height * width).transpose(1, 2)  
            if ma_mask.ndim == 3:
                B, N, C = hidden_bg.shape 
                if not is_perfect_square(N):
                    ma_mask=torch.cat([ma_mask, ref_mask], dim=-1) 
                ma_mask = ma_mask.float()

                ma_mask = ma_mask.unsqueeze(0)  
                if ma_mask.shape[0] != 1:
                    ma_mask = ma_mask.unsqueeze(0)  
                else:
                    ma_mask = ma_mask.unsqueeze(1)

                if is_perfect_square(N):
                    new_hw = int(N ** 0.5)
                    ma_mask = F.interpolate(ma_mask, size=(1,new_hw, new_hw), mode="nearest")  # [1, 1, new_h, new_w]
                else:
                    # Concatenate ref_mask with ma_mask for SAET-based processing
                    new_hw = int((N/2) ** 0.5)
                    ma_mask = F.interpolate(ma_mask, size=(1,new_hw, 2*new_hw), mode="nearest") 

                ma_mask = ma_mask.view(1, -1, 1)

                ma_mask = ma_mask.expand(B, -1, -1).to(dtype=dtype, device=device)
            # Blend foreground and background attention outputs
            hidden_states = ma_mask * hidden_fg + (1 - ma_mask) * hidden_bg
        else:
            hidden_states = hidden_fg

        hidden_states = attn.to_out[0](hidden_states)

        hidden_states = attn.to_out[1](hidden_states)


        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states