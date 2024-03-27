# Copyright (c) OpenMMLab. All rights reserved.

from audioop import add
from cmd import PROMPT
from typing import Dict, List

import torch
import torch.nn as nn

from mmocr.registry import MODELS
from mmocr.utils.typing_utils import (ConfigType, InitConfigType,
                                      OptConfigType, OptRecSampleList,
                                      RecForwardResults, RecSampleList)
from .base import BaseRecognizer

from mmengine.structures import LabelData

from open_flamingo import create_model_and_transforms

import torch.nn.functional as Functional

import copy

from PIL import Image
from torchvision import transforms
import numpy as np

import torch.distributed as dist

import json

from einops import rearrange

import random

import re

@MODELS.register_module()
class Flamingo(BaseRecognizer):

    def __init__(self,
                 preprocessor: OptConfigType = None,
                 data_len = 5,
                 prompt_type = 0,  
                 model_type='S',
                 vit_type='small',
                 pos_process = False,
                 layer_num=12,
                 load_weight = None,
                 load_mae = None,
                 unfreeze_vit = False,
                 unfreeze_decoder = False,
                 use_icl = False,
                 lm_path = None,
                 pos_path = None,
                 pool_path = None,
                 data_preprocessor: ConfigType = None,
                 init_cfg: InitConfigType = None) -> None:

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        # Preprocessor module, e.g., TPS
        if preprocessor is not None:
            self.preprocessor = MODELS.build(preprocessor)     
            
        self.prompt_type = prompt_type
        self.model_type = model_type
        self.vit_type = vit_type

        if self.vit_type == 'small':
            self.img_size = (32, 128)
        else:
            self.img_size = (224, 224)

        self.load_weight = load_weight
        self.unfreeze_vit = unfreeze_vit
        self.unfreeze_decoder = unfreeze_decoder
        self.load_mae = load_mae

        self.use_icl = use_icl
        self.lm_path = lm_path
        self.pos_path = pos_path
        self.pool_path = pool_path
        
        assert data_len > 0
        self.data_len = data_len
        self.pos_process = pos_process
        self.layer_num=layer_num

        assert not (self.load_weight is not None and self.load_mae is not None)
        self.init_model()
        self.print_first_prompt = True

        if self.pos_process:
            self.ini_pos()

        self.all_prompt = None
        self.all_vision_fea = None

        # for pos_process
        self.english_re = re.compile(r'[A-Za-z]',re.S)
        self.all_fail_pos = 0


    def ini_pos(self):
        from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ColorJitter, RandomRotation

        self.pos_processor = Compose([
            ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5),
            RandomRotation(30, expand=True),
            Resize(self.img_size),
            ToTensor(),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.origin_processor = Compose([
            Resize(self.img_size),
            ToTensor(),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        # a dict containing character_position of all images
        # key: img_path     value: List[float]
        print('load character position dict...')
        with open(self.pos_path, 'r') as json_file:
            self.character_position = json.load(json_file)


    def init_model(self):

        if self.model_type == 'S':
            lm_path = self.lm_path
        else:
            raise Exception("Invalid model_type {}".format(self.model_type))

        if self.vit_type == 'small':
            vit_path = 'ViT-B-4'
        else:
            raise Exception("Invalid vit_type {}".format(self.vit_type))

        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=vit_path,
            clip_vision_encoder_pretrained=None,
            lang_encoder_path=lm_path,
            tokenizer_path=lm_path,
            cross_attn_every_n_layers=1,
            #decoder_layers_attr_name='decoder.layers'#'transformer.h'
        )
        
        self.model = model
        self.tokenizer = tokenizer

        if self.vit_type == 'small':
            from functools import partial
            import torch.nn as nn
            from timm.models.vision_transformer import VisionTransformer
            ori_vit = VisionTransformer(
                img_size=(32, 128),
                patch_size=4,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            self.model.vision_encoder.conv1 = copy.deepcopy(ori_vit.patch_embed.proj)
            self.model.vision_encoder.ln_pre = nn.Identity()
            del ori_vit
        else:
            raise Exception("Invalid vit_type {}".format(self.vit_type))
            
        del model
        del tokenizer
        del image_processor

        if self.load_weight is not None:
        
            print('load weight...')
            ori_dict = torch.load(
                self.load_weight, map_location='cpu'
            )
            new_dict = {}
            for k in ori_dict.keys():
                tmp_k = k
                if k[:6] == 'model.':
                    tmp_k = tmp_k[6:]
                new_dict[tmp_k] = copy.deepcopy(ori_dict[k])

            del ori_dict

            self.model.load_state_dict(new_dict, strict=False)

        if self.load_mae is not None:
            print('load mae vit...')
            if self.vit_type == 'small':
                model_dict = torch.load(self.load_mae, map_location='cpu')
                ori_dict = model_dict['model']
                clip_dict = {}
                clip_dict['positional_embedding'] = copy.deepcopy(ori_dict['pos_embed'].squeeze(0))
                block_keys_ori = ['norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
                              'attn.proj.weight', 'attn.proj.bias', 'norm2.weight', 'norm2.bias',
                                'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias']
                block_keys_clip = ['ln_1.weight', 'ln_1.bias', 'attn.in_proj_weight', 'attn.in_proj_bias',
                                'attn.out_proj.weight', 'attn.out_proj.bias', 'ln_2.weight', 'ln_2.bias',
                                'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias']
                for i in range(12):
                    for k_ori, k_clip in zip(block_keys_ori, block_keys_clip):
                        clip_dict['transformer.resblocks.{}.'.format(i)+k_clip] =\
                            copy.deepcopy(ori_dict['blocks.{}.'.format(i)+k_ori])
                clip_dict['ln_post.weight'] = copy.deepcopy(ori_dict['norm.weight'])
                clip_dict['ln_post.bias'] = copy.deepcopy(ori_dict['norm.bias'])
                clip_dict['conv1.weight'] = copy.deepcopy(ori_dict['patch_embed.proj.weight'])
                clip_dict['conv1.bias'] = copy.deepcopy(ori_dict['patch_embed.proj.bias'])
                del ori_dict
                del model_dict
                self.model.vision_encoder.load_state_dict(clip_dict, strict=False)

            else:
                raise Exception("Invalid vit_type {}".format(self.vit_type))

        
        # unfreeze vit
        if self.unfreeze_vit:
            print('unfreeze vit...')
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = True
            # discard unused params
            if hasattr(self.model.vision_encoder, 'proj'):
                self.model.vision_encoder.proj.requires_grad = False
            self.model.vision_encoder.ln_post.weight.requires_grad = False
            self.model.vision_encoder.ln_post.bias.requires_grad = False

        if self.unfreeze_decoder:
            print('unfreeze decoder...')
            for param in self.model.lang_encoder.parameters():
                param.requires_grad = True

        if self.layer_num < 12:
            for i in range(11, self.layer_num-1, -1):
                del self.model.vision_encoder.transformer.resblocks[i]
                del self.model.lang_encoder.model.decoder.layers[i]
                del self.model.lang_encoder.old_decoder_blocks[i]
                del self.model.lang_encoder.gated_cross_attn_layers[i]

        # print(self.model.vision_encoder)

        # print(self.model.lang_encoder)

    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        """Directly extract features from the backbone."""
        if self.with_preprocessor:
            inputs = self.preprocessor(inputs)
        if self.with_backbone:
            inputs = self.backbone(inputs)
        return inputs

    def pos_preprocess(self, data_samples: RecSampleList, device, test) -> list:

        # cal position-based transform for all available data points
        imgs_pos, gts_pos = [], []
        imgs_ori, gts_ori = [], []
        # imgs_trs constains only data without position informations
        imgs_trs, gts_trs = [], []
        
        for data in data_samples:

            gt_text = data.gt_text.item

            if data.img_path not in self.character_position:
                t_cur_img = Image.open(data.img_path).convert('RGB')
                imgs_ori.append(self.origin_processor(t_cur_img).unsqueeze(0))

                gts_ori.append(gt_text)

                imgs_trs.append(self.pos_processor(t_cur_img).unsqueeze(0))

                gts_trs.append(gt_text)

                continue

            character_pos = self.character_position[data.img_path]

            if len(gt_text) != len(character_pos):
                print('character position error, data point ignored !')
                continue

            if len(gt_text) <= 1:
                continue

            gt_len = len(gt_text)

            # preprocess character_pos
            t_cur_img = Image.open(data.img_path).convert('RGB')
            width, height = t_cur_img.size
            character_pos = np.array(character_pos)

            imgs_ori.append(self.origin_processor(t_cur_img).unsqueeze(0))


            gts_ori.append(gt_text)

            # start to divide

            if len(re.findall(self.english_re,gt_text)):
                num_divide = random.randint(2, max(int(len(gt_text)*1.0), 2))
            else:
                num_divide = random.randint(1, max(int(len(gt_text)*0.7), 1))
            for i in range(num_divide):
                pos_l = random.randint(0, gt_len-1)
                pos_r = random.randint(pos_l, gt_len-1)
                w_l = character_pos[pos_l-1][2] if pos_l >= 1 else 0
                w_r = character_pos[pos_r][2]

                try:
                    cur_pos_img = t_cur_img.crop((w_l, 0, w_r, height))
                    cur_pos_img_trs = self.pos_processor(cur_pos_img)
                    imgs_pos.append(cur_pos_img_trs.unsqueeze(0))
                    
                    gts_pos.append(gt_text[pos_l:pos_r+1])
                except:
                    self.all_fail_pos += 1
                    print('cut image failed {}'.format(self.all_fail_pos))
                    pass


        # decide all imgs
        true_imgs, true_gts = imgs_ori, gts_ori

        # decide pos imgs
        pos_imgs_sample_num = min(len(imgs_ori), len(imgs_pos))
        if pos_imgs_sample_num > 0:
            perm_ids = torch.randperm(len(imgs_pos))
            for i in range(pos_imgs_sample_num):
                true_imgs.append(imgs_pos[perm_ids[i]])
                true_gts.append(gts_pos[perm_ids[i]])
        
        # decide trs imgs
        if len(imgs_trs) > 0:
            trs_imgs_sample_num = random.randint(2, max(2, int(0.3*len(imgs_trs))))
            perm_ids = torch.randperm(len(imgs_trs))
            for i in range(trs_imgs_sample_num):
                true_imgs.append(imgs_trs[perm_ids[i]])
                true_gts.append(gts_trs[perm_ids[i]])

        true_imgs = torch.cat(true_imgs, dim=0) # icl_len*2.2, 3, 224, 224
        img_shape = true_imgs.shape[-3:]
        true_imgs = true_imgs.view(-1, 1, *img_shape)  # icl_len*2.2, 1, 3, 224, 224
        true_gts = np.array(true_gts)

        # random perm all imgs
        perm_ids = torch.randperm(len(true_gts))
        true_imgs = true_imgs[perm_ids]
        true_gts = true_gts[perm_ids]

        # avoid cuda-oom
        true_imgs = true_imgs[:80]
        true_gts = true_gts[:80]

        true_imgs = true_imgs.unsqueeze(0) # 1, 50, 1, 3, 224, 224


        prompts = []
        prompt_type = self.prompt_type
        cur_p = ""
        if not test:
            for i in range(len(true_gts)):
                if prompt_type == 1:
                    cur_p += "<image>The text in the image is \'{}\'<|endofchunk|>".format(true_gts[i])
                else:
                    cur_p += "<image>{}<|endofchunk|>".format(true_gts[i])
        else:
            for i in range(len(true_gts)-1):
                if prompt_type == 1:
                    cur_p += "<image>The text in the image is \'{}\'<|endofchunk|>".format(true_gts[i])
                else:
                    cur_p += "<image>{}<|endofchunk|>".format(true_gts[i])
            if prompt_type == 1:
                cur_p += "<image>The text in the image is"
            else:
                cur_p += "<image>"
        prompts.append(cur_p)

        if self.print_first_prompt:
            print('prompt: ', prompts[0])
            self.print_first_prompt = False
        
        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        lang_x = self.tokenizer(
            prompts,
            return_tensors="pt", padding=True,
        )

        labels = lang_x["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[labels == self.tokenizer.eos_token] = -100
        media_token_id = self.tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        labels[labels == media_token_id] = -100

        true_imgs = true_imgs.to(device)
        if self.with_preprocessor:
            true_imgs = self.preprocessor(true_imgs)

        res = [true_imgs, lang_x["input_ids"].to(device), lang_x["attention_mask"].to(device), labels.to(device)]
        
        return res

    def preprocess(self, inputs: torch.Tensor, data_samples: RecSampleList, data_len=None, test=False) -> List:
        """Directly extract features from the backbone."""

        if self.pos_process and not test:
            return self.pos_preprocess(data_samples, device=inputs.device, test=test)
        
        if self.with_preprocessor:
            inputs = self.preprocessor(inputs)

        if data_len is None:
            data_len = self.data_len
        img_size = inputs.shape[-3:]
        if inputs.shape[0] % data_len != 0:
            print('data_len does not mod inputs.shape, tail data dropped !!')
            inputs = inputs[:inputs.shape[0]//data_len * data_len]
            data_samples = data_samples[:inputs.shape[0]//data_len * data_len]
        inputs = inputs.view(-1, data_len, *img_size).unsqueeze(2)

        # print('inputs: ', inputs.shape)
        
        gts = []
        for gt in data_samples:
            gts.append(gt.gt_text.item)
        all_gt = []
        for i in range(0, len(gts), data_len):
            all_gt.append(gts[i:i+data_len])

        prompts = []
        prompt_type = self.prompt_type
        for gt in all_gt:
            cur_p = ""
            if not test:
                for i in range(len(gt)):
                    if prompt_type == 1:
                        cur_p += "<image>The text in the image is \'{}\'<|endofchunk|>".format(gt[i])
                    else:
                        cur_p += "<image>{}<|endofchunk|>".format(gt[i])
            else:
                for i in range(len(gt)-1):
                    if prompt_type == 1:
                        cur_p += "<image>The text in the image is \'{}\'<|endofchunk|>".format(gt[i])
                    else:
                        cur_p += "<image>{}<|endofchunk|>".format(gt[i])
                if prompt_type == 1:
                    cur_p += "<image>The text in the image is"
                else:
                    cur_p += "<image>"
            prompts.append(cur_p)

        if self.print_first_prompt:
            print('prompt: ', prompts[0])
            self.print_first_prompt = False
        
        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        lang_x = self.tokenizer(
            prompts,
            return_tensors="pt", padding=True,
        )

        labels = lang_x["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[labels == self.tokenizer.eos_token] = -100
        media_token_id = self.tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        labels[labels == media_token_id] = -100

        res = [inputs, lang_x["input_ids"].to(inputs.device), lang_x["attention_mask"].to(inputs.device), labels.to(inputs.device)]

        return res

    def loss(self, inputs: torch.Tensor, data_samples: RecSampleList,
             **kwargs) -> Dict:
        
        in_context_inputs = self.preprocess(inputs, data_samples)

        del inputs
        
        return {'loss': self.model(*in_context_inputs)[0]}

    def get_test_context(self, device, self_img_path=None, inputs=None) -> List:

        context_size = 2
        if self.all_prompt is None:
            self.sum_good = 0
            self.sum_bad = 0
            with open(self.pool_path, 'r') as json_file:
                self.all_prompt = json.load(json_file)

        if self.all_vision_fea is None:
            self.get_all_vision_fea(device)

        all_vision_distance = None

        # cal vision distance
        with torch.no_grad():
            imgs = []
            if inputs is not None:
                imgs.append(inputs.unsqueeze(0))
            else:
                imgs.append(self.origin_processor(Image.open(self_img_path).convert('RGB')).unsqueeze(0))
            imgs = torch.cat(imgs, dim=0).to(device)
            vision_x = imgs.unsqueeze(1).unsqueeze(0)
            b, T, F = vision_x.shape[:3]
            vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
            vision_x = self.model.vision_encoder(imgs)[1]
            vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
            vision_x = self.model.perceiver(vision_x)
            vision_x = vision_x.squeeze(0).squeeze(0).mean(0, keepdim=True)
            vision_x = Functional.normalize(vision_x, dim=-1) # (1, 1024)
            all_vision_distance = torch.mm(self.all_vision_fea, vision_x.transpose(0, 1)).squeeze(-1) # (N,)
            all_vision_distance = Functional.normalize(all_vision_distance, dim=0).cpu()

        q = torch.topk(all_vision_distance, context_size * 10, largest=True)[1]
        context = []
        r_context = []
        for i in q:
            context.append(self.all_prompt[i])
        if len(context) < context_size:
            remains = context_size - len(context)
            for i in range(remains):
                context.append(r_context[i])
        else:
            context = context[:context_size]
        return context

    def get_all_vision_fea(self, device):
        assert self.all_vision_fea is None
        assert self.all_prompt is not None

        self.all_vision_fea = []

        print('calculate all vision features...')

        from tqdm import tqdm
        for prompt in tqdm(self.all_prompt):
            img_path = prompt['img_path']
            with torch.no_grad():
                imgs = []
                imgs.append(self.origin_processor(Image.open(img_path).convert('RGB')).unsqueeze(0))
                
                imgs = torch.cat(imgs, dim=0).to(device)
                vision_x = imgs.unsqueeze(1).unsqueeze(0)
                b, T, F = vision_x.shape[:3]
                vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
                vision_x = self.model.vision_encoder(imgs)[1]
                vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
                vision_x = self.model.perceiver(vision_x)
                vision_x = vision_x.squeeze(0).squeeze(0).mean(0, keepdim=True)
                self.all_vision_fea.append(Functional.normalize(vision_x, dim=-1))

        self.all_vision_fea = torch.cat(self.all_vision_fea, dim=0)

        print('all_vision_fea: ', self.all_vision_fea.shape)
        

    def handle_test_context(self, context, data_sample, device):
        context.append(
            {
                'img_path': data_sample.img_path,
                'gt_text': data_sample.gt_text.item
            }
        )
        ctx_images = []
        for i, bad_case in enumerate(context):
            if i < len(ctx_images) - 1:
                ctx_images.append(self.origin_processor(Image.open(bad_case['img_path']).convert('RGB')).unsqueeze(0))
            else:
                ctx_images.append(self.origin_processor(Image.open(bad_case['img_path']).convert('RGB')).unsqueeze(0))
                
        ctx_gts = [bad_case['gt_text'] for bad_case in context]
        imgs = torch.cat(ctx_images, dim=0)
        ctx_images = imgs.unsqueeze(1).unsqueeze(0)

        prompt = ""
        for i in range(len(ctx_gts)-1):
            prompt += "<image>{}<|endofchunk|>".format(ctx_gts[i])
        prompt += "<image>"

        ctx_gts = np.array(ctx_gts)
        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        lang_x = self.tokenizer(
            [prompt],
            return_tensors="pt", padding=True,
        )

        out = self.model.generate(
            vision_x=ctx_images.to(device),
            lang_x=lang_x["input_ids"].to(device),
            attention_mask=lang_x["attention_mask"].to(device),
            max_new_tokens=48,
            num_beams=3,
        )

        true_out = self.tokenizer.decode(out[0])

        true_out = true_out.replace('</s>', '')[len(prompt):].replace('<|endofchunk|>', '').replace('<image>', '')

        return context, true_out


    def predict(self, inputs: torch.Tensor, data_samples: RecSampleList,
                **kwargs) -> RecSampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Image input tensor.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.

        Returns:
            list[TextRecogDataSample]:  A list of N datasamples of prediction
            results. Results are stored in ``pred_text``.
        """

        use_icl = self.use_icl
        
        in_context_inputs = self.preprocess(inputs, data_samples, data_len=1, test=True)

        input_feats, input_ids, attention_masks, labels = in_context_inputs

        for i in range(len(data_samples)):
            tmp_input_feats=input_feats[i].unsqueeze(0)
            tmp_input_ids=input_ids[i].unsqueeze(0)
            tmp_attention_masks = attention_masks[i].unsqueeze(0)
            out = self.model.generate(tmp_input_feats, tmp_input_ids, tmp_attention_masks, 
                                    max_new_tokens=48)
            input_ids_len = len(input_ids[i])
            out = out[:, input_ids_len:]
            
            tmp_pred_text = self.tokenizer.decode(out[0]).replace('<|endofchunk|>', '').replace('<image>', '').replace('</s>', '')
            
            
            if use_icl:

                context = self.get_test_context(device=inputs.device, self_img_path=data_samples[i].img_path,)
                
                context, true_pred = self.handle_test_context(context, data_samples[i], device=inputs.device)


            else:
                true_pred = tmp_pred_text


            pred_text = LabelData()
            pred_text.item = true_pred

            data_samples[i].pred_text = pred_text

        return data_samples

    def _forward(self,
                 inputs: torch.Tensor,
                 data_samples: OptRecSampleList = None,
                 **kwargs) -> RecForwardResults:
        """Network forward process. Usually includes backbone, encoder and
        decoder forward without any post-processing.

         Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (list[TextRecogDataSample]): A list of N
                datasamples, containing meta information and gold
                annotations for each of the images.

        Returns:
            Tensor: A tuple of features from ``decoder`` forward.
        """
        in_context_inputs = self.preprocess(inputs, data_samples)
        return self.model(*in_context_inputs)[1]