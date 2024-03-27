# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Sequence, Union
from mmocr.models.textrecog.decoders import llama_decoder

import torch
import torch.nn as nn
from mmengine.model import ModuleList

from mmocr.models.common import PositionalEncoding, TFDecoderLayer
from mmocr.models.common.dictionary import Dictionary
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample
from .base import BaseDecoder

from transformers import LlamaForCausalLM
from torch.nn import MSELoss

from .llama_decoder import LlamaDecoder


@MODELS.register_module()
class LlamaDistillDecoder(BaseDecoder):
    

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

        # self.trg_word_emb = nn.Embedding(
        #     self.dictionary.num_classes,
        #     d_embedding,
        #     padding_idx=self.padding_idx)

        # self.context_vision_TF = TFDecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        # self.context_vision_TF_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # llama d_model: 4096
        self.llama_dim = 4096
        # self.vl_linear = nn.Linear(d_model, self.llama_dim)

        # print('loading llama...')
        # llama_path = '/mnt/bn/zz-nas/data/llm_weights/llama_7b'
        # self.decoder = LlamaForCausalLM.from_pretrained(
        #     llama_path, device_map='cpu'
        # )
        # self.decoder.lm_head = nn.Identity()
        # print('frozen llama weights...')
        # for name, param in self.decoder.named_parameters():
        #     param.requires_grad = False

        self.llama_model = LlamaDecoder(n_layers=6,
        d_embedding=768,
        n_head=8,
        d_model=768,
        d_inner=3072,
        d_k=96,
        d_v=96)
        print('load lora llama...')
        self.llama_model.load_state_dict(
            torch.load('/mnt/bn/zz-nas/Union14M/mmocr-dev-1.x/work_dirs/maerec_b_union14m/epoch_10_llama.pth'))
        print('frozen lora llama...')
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        
        self.small_decoder = ModuleList([
            TFDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.small_decoder_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # self.position_enc = PositionalEncoding(
        #     d_embedding, n_position=n_position)
        # self.dropout = nn.Dropout(p=dropout)

        pred_num_class = self.dictionary.num_classes
        # self.classifier = nn.Linear(self.llama_dim, pred_num_class)
        self.small_classifier = nn.Linear(d_model, pred_num_class)
        # self.softmax = nn.Softmax(dim=-1)

        self.distill_linear = nn.Linear(d_model, self.llama_dim)

        for name, param in self.decoder.named_parameters():
            print(name, '  ', param.requires_grad)

    def _get_source_mask(self, src_seq: torch.Tensor,
                         valid_ratios: Sequence[float]) -> torch.Tensor:

        N, T, _ = src_seq.size()
        mask = None
        if len(valid_ratios) > 0:
            mask = src_seq.new_zeros((N, T), device=src_seq.device)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(T, math.ceil(T * valid_ratio))
                mask[i, :valid_width] = 1

        return mask

    def _get_target_mask(self, trg_seq: torch.Tensor) -> torch.Tensor:

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
        # q-former

        trg_embedding = self.llama_model.trg_word_emb(trg_seq)
        trg_pos_encoded = self.llama_model.position_enc(trg_embedding)
        trg_mask = self._get_target_mask(trg_seq)
        tgt_seq = self.llama_model.dropout(trg_pos_encoded)

        output = tgt_seq
        output = self.llama_model.context_vision_TF(
            output,
            src,
            self_attn_mask=trg_mask,
            dec_enc_attn_mask=src_mask)
        output = self.llama_model.context_vision_TF_layer_norm(output)

        return output

    def small_attention(self,
                   trg_seq: torch.Tensor,
                   src: torch.Tensor,
                   src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        trg_embedding = self.llama_model.trg_word_emb(trg_seq)
        trg_pos_encoded = self.llama_model.position_enc(trg_embedding)
        trg_mask = self._get_target_mask(trg_seq)
        tgt_seq = self.llama_model.dropout(trg_pos_encoded)

        output = tgt_seq
        for dec_layer in self.small_decoder:
            output = dec_layer(
                output,
                src,
                self_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
        output = self.small_decoder_layer_norm(output)

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
        attn_output = self.small_attention(trg_seq, feat, src_mask=src_mask)
        outputs = self.small_classifier(attn_output)

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
        init_target_seq = torch.full((N, self.max_seq_len + 1), self.padding_idx, device=feat.device, dtype=torch.long)
        # bsz * seq_len
        init_target_seq[:, 0] = self.start_idx

        outputs = []
        for step in range(0, self.max_seq_len):
            decoder_output = self.small_attention(init_target_seq, feat, src_mask=src_mask)
            step_result = self.small_classifier(decoder_output[:, step, :])
            outputs.append(step_result)
            _, step_max_index = torch.max(step_result, dim=-1)
            init_target_seq[:, step + 1] = step_max_index

        outputs = torch.stack(outputs, dim=1)

        return self.softmax(outputs)

    def loss(self,
             feat: Optional[torch.Tensor] = None,
             out_enc: Optional[torch.Tensor] = None,
             data_samples: Optional[Sequence[TextRecogDataSample]] = None
             ) -> Dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feat (Tensor, optional): Features from the backbone. Defaults
                to None.
            out_enc (Tensor, optional): Features from the encoder.
                Defaults to None.
            data_samples (list[TextRecogDataSample], optional): A list of
                N datasamples, containing meta information and gold
                annotations for each of the images. Defaults to None.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """

        if self.training and getattr(self, 'module_loss') is not None:
            data_samples = self.module_loss.get_targets(data_samples)

        valid_ratios = []
        for data_sample in data_samples:
            valid_ratios.append(data_sample.get('valid_ratio'))
        src_mask = self._get_source_mask(feat, valid_ratios)
        trg_seq = []
        for data_sample in data_samples:
            trg_seq.append(data_sample.gt_text.padded_indexes.to(feat.device))
        trg_seq = torch.stack(trg_seq, dim=0)

        # llama output
        context_vision_output = self._attention(trg_seq, feat, src_mask=src_mask)
        context_vision_output = self.llama_model.vl_linear(context_vision_output)
        llama_output = self.llama_model.decoder(inputs_embeds=context_vision_output)
        llama_fea = llama_output.logits
        llama_logits = self.llama_model.classifier(llama_fea)

        # small_decoder output
        small_decoder_fea = self.small_attention(trg_seq, feat, src_mask=src_mask)
        small_decoder_logits = self.small_classifier(small_decoder_fea)

        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

        loss_ce = self.module_loss(small_decoder_logits, data_samples)['loss_ce']
        loss_llm_ce = self.module_loss(llama_logits, data_samples)['loss_ce']
        loss_distill = 0.

        loss_distill += soft_cross_entropy(llama_logits, small_decoder_logits)
        loss_distill += MSELoss()(self.distill_linear(small_decoder_fea), llama_fea)
        
        tot_loss = {
            'loss_ce': loss_ce,
            'loss_llm_ce': loss_llm_ce,
            'loss_distill': loss_distill
        }

        return tot_loss
