# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmengine.model import ModuleList

from mmocr.models.common import PositionalEncoding, TFDecoderLayer
from mmocr.models.common.dictionary import Dictionary
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample
from .base import BaseDecoder

from transformers import LlamaForCausalLM
import loralib as lora
import copy


@MODELS.register_module()
class LlamaDecoder(BaseDecoder):
    

    def __init__(self,
                 n_layers: int = 6,
                 d_embedding: int = 512,
                 n_head: int = 8,
                 d_k: int = 64,
                 d_v: int = 64,
                 d_model: int = 512,
                 d_inner: int = 256,
                 n_position: int = 200,
                 dropout: float = 0.1,
                 module_loss: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 dictionary: Optional[Union[Dict, Dictionary]] = None,
                 max_seq_len: int = 30,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(
            module_loss=module_loss,
            postprocessor=postprocessor,
            dictionary=dictionary,
            init_cfg=init_cfg,
            max_seq_len=max_seq_len)

        self.padding_idx = self.dictionary.padding_idx
        self.start_idx = self.dictionary.start_idx
        self.max_seq_len = max_seq_len

        self.trg_word_emb = nn.Embedding(
            self.dictionary.num_classes,
            d_embedding,
            padding_idx=self.padding_idx)

        self.context_vision_TF = TFDecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.context_vision_TF_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # llama d_model: 4096
        self.llama_dim = 4096
        self.vl_linear = nn.Linear(d_model, self.llama_dim)

        print('loading llama...')
        llama_path = '/mnt/bn/zz-nas/data/llm_weights/llama_7b'
        self.decoder = LlamaForCausalLM.from_pretrained(
            llama_path, device_map='cuda'
        )
        self.decoder.lm_head = nn.Identity()

        self.lora_llm()

        # print('frozen llama weights...')
        # for name, param in self.decoder.named_parameters():
        #     param.requires_grad = False

        self.position_enc = PositionalEncoding(
            d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        pred_num_class = self.dictionary.num_classes
        self.classifier = nn.Linear(self.llama_dim, pred_num_class)
        self.softmax = nn.Softmax(dim=-1)

        # for name, param in self.decoder.named_parameters():
        #     print(name, '  ', param.requires_grad)

    def lora_llm(self):
        print('add lora to llama...')
        decoder_weight = copy.deepcopy(self.decoder.state_dict())
        lora_r = 16
        for i in range(len(self.decoder.model.layers)):
            self.decoder.model.layers[i].self_attn.q_proj = lora.Linear(4096, 4096, r=lora_r)
            self.decoder.model.layers[i].self_attn.k_proj = lora.Linear(4096, 4096, r=lora_r)
            self.decoder.model.layers[i].self_attn.v_proj = lora.Linear(4096, 4096, r=lora_r)
            self.decoder.model.layers[i].self_attn.o_proj = lora.Linear(4096, 4096, r=lora_r)
            
            
            self.decoder.model.layers[i].mlp.gate_proj = lora.Linear(4096, 11008, r=lora_r)
            self.decoder.model.layers[i].mlp.down_proj = lora.Linear(11008, 4096, r=lora_r)
            self.decoder.model.layers[i].mlp.up_proj = lora.Linear(4096, 11008, r=lora_r)
            

        print('lora_r: ', lora_r)
        print('load llama weights...')
        self.decoder.load_state_dict(decoder_weight, strict=False)
        print('freeze llama weights...')
        lora.mark_only_lora_as_trainable(self.decoder)
        del decoder_weight
        torch.cuda.empty_cache()

    def _get_source_mask(self, src_seq: torch.Tensor,
                         valid_ratios: Sequence[float]) -> torch.Tensor:
        """Generate mask for source sequence.

        Args:
            src_seq (torch.Tensor): Image sequence. Shape :math:`(N, T, C)`.
            valid_ratios (list[float]): The valid ratio of input image. For
                example, if the width of the original image is w1 and the width
                after padding is w2, then valid_ratio = w1/w2. Source mask is
                used to cover the area of the padding region.

        Returns:
            Tensor or None: Source mask. Shape :math:`(N, T)`. The region of
            padding area are False, and the rest are True.
        """

        N, T, _ = src_seq.size()
        mask = None
        if len(valid_ratios) > 0:
            mask = src_seq.new_zeros((N, T), device=src_seq.device)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(T, math.ceil(T * valid_ratio))
                mask[i, :valid_width] = 1

        return mask

    def _get_target_mask(self, trg_seq: torch.Tensor) -> torch.Tensor:
        """Generate mask for target sequence.

        Args:
            trg_seq (torch.Tensor): Input text sequence. Shape :math:`(N, T)`.

        Returns:
            Tensor: Target mask. Shape :math:`(N, T, T)`.
            E.g.:
            seq = torch.Tensor([[1, 2, 0, 0]]), pad_idx = 0, then
            target_mask =
            torch.Tensor([[[True, False, False, False],
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, False]]])
        """

        pad_mask = (trg_seq != self.padding_idx).unsqueeze(-2)

        len_s = trg_seq.size(1)
        subsequent_mask = 1 - torch.triu(
            torch.ones((len_s, len_s), device=trg_seq.device), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).bool()

        return pad_mask & subsequent_mask

    def _attention(self,
                   trg_seq: torch.Tensor,
                   src: torch.Tensor,
                   src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        trg_embedding = self.trg_word_emb(trg_seq)
        trg_pos_encoded = self.position_enc(trg_embedding)
        trg_mask = self._get_target_mask(trg_seq)
        tgt_seq = self.dropout(trg_pos_encoded)

        output = tgt_seq
        output = self.context_vision_TF(
            output,
            src,
            self_attn_mask=trg_mask,
            dec_enc_attn_mask=src_mask)
        output = self.context_vision_TF_layer_norm(output)

        return output

    def forward_train(
            self,
            feat: Optional[torch.Tensor] = None,
            out_enc: torch.Tensor = None,
            data_samples: Sequence[TextRecogDataSample] = None
    ) -> torch.Tensor:
        valid_ratios = []
        for data_sample in data_samples:
            valid_ratios.append(data_sample.get('valid_ratio'))
        src_mask = self._get_source_mask(feat, valid_ratios)
        trg_seq = []
        for data_sample in data_samples:
            trg_seq.append(data_sample.gt_text.padded_indexes.to(feat.device))
        trg_seq = torch.stack(trg_seq, dim=0)

        # trg_mask for llama
        # llama_mask = (trg_seq != self.padding_idx).long()
        # print('init_target_seq: ', trg_seq[0, :10])
        # print('llama_mask: ', llama_mask[0, :10])

        context_vision_output = self._attention(trg_seq, feat, src_mask=src_mask)
        context_vision_output = self.vl_linear(context_vision_output)
        # print('context_vision_output: ', context_vision_output.shape)
        llama_output = self.decoder(inputs_embeds=context_vision_output)
        # llama_output = self.decoder(inputs_embeds=context_vision_output, attention_mask=llama_mask)
        outputs = self.classifier(llama_output.logits)

        return outputs

    def forward_test(
            self,
            feat: Optional[torch.Tensor] = None,
            out_enc: torch.Tensor = None,
            data_samples: Sequence[TextRecogDataSample] = None
    ) -> torch.Tensor:

        valid_ratios = []
        for data_sample in data_samples:
            valid_ratios.append(data_sample.get('valid_ratio'))
        src_mask = self._get_source_mask(feat, valid_ratios)
        N = feat.size(0)
        init_target_seq = torch.full((N, self.max_seq_len + 1),
                                     self.padding_idx,
                                     device=feat.device,
                                     dtype=torch.long)
        # bsz * seq_len
        init_target_seq[:, 0] = self.start_idx

        outputs = []
        for step in range(0, self.max_seq_len):

            # trg_mask for llama
            # llama_mask = (init_target_seq != self.padding_idx).long()

            context_vision_output = self._attention(init_target_seq, feat, src_mask=src_mask)
            context_vision_output = self.vl_linear(context_vision_output)
            llama_output = self.decoder(inputs_embeds=context_vision_output) #, attention_mask=llama_mask)
            step_result = self.classifier(llama_output.logits[:, step, :])
            outputs.append(step_result)
            _, step_max_index = torch.max(step_result, dim=-1)
            init_target_seq[:, step + 1] = step_max_index

        outputs = torch.stack(outputs, dim=1)

        return self.softmax(outputs)
