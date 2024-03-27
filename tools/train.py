# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

import torch

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

import copy



def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='The dir to save logs and models')
    parser.add_argument(
        '--resume', action='store_true', help='Whether to resume checkpoint.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='Enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='Whether to scale the learning rate automatically. It requires '
        '`auto_scale_lr` in config, and `base_batch_size` in `auto_scale_lr`')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():

    """
    from open_flamingo import create_model_and_transforms


    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="bigscience/bloomz-1b7",
        tokenizer_path="bigscience/bloomz-1b7",
        cross_attn_every_n_layers=1,
        decoder_layers_attr_name='transformer.h'
    )  

    ori_dict = torch.load(
        '/mnt/bn/zz-nas/Union14M/mmocr-dev-1.x/work_dirs/icl/epoch_3.pth', map_location='cuda:0'
    )
    new_dict = {}
    for k in ori_dict.keys():
        tmp_k = k
        if k[:6] == 'model.':
            tmp_k = tmp_k[6:]
        new_dict[tmp_k] = copy.deepcopy(ori_dict[k])

    del ori_dict

    model.load_state_dict(new_dict)
    """

    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # cfg.load_from = '/mnt/bn/zz-nas/Union14M/checkpoints/maerec_b_union14m.pth'
    # cfg.load_from = '/mnt/bn/zz-nas/Union14M/checkpoints/epoch_10.pth'

    # enable automatic-mixed-precision training
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                '`OptimWrapper` but got {}.'.format(optim_wrapper))
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.resume:
        cfg.resume = True

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # print('len: ', len(runner.train_dataloader))
    # loader = runner.build_dataloader(runner.train_dataloader)
    # for x in loader:
    #     print(x)
    #     break


    """
    print('len: ', len(runner.test_dataloader))
    icl_len = 5
    loader = runner.build_dataloader(runner.test_dataloader)
    for x in loader:
        images = x['inputs']
        all_im = []
        for im in images:
            all_im.append(im.unsqueeze(0))
            # print('images: ', im.shape)
        all_im = torch.cat(all_im, dim=0)
        img_size = all_im.shape[-3:]
        all_im = all_im.view(-1, icl_len, *img_size).unsqueeze(2)
        print('all_im: ', all_im.shape)
        gts = []
        for gt in x['data_samples']:
            gts.append(gt.gt_text.item)
        print('gts: ', len(gts))

        all_gt = []
        for i in range(0, len(gts), icl_len):
            all_gt.append(gts[i:i+icl_len])
        # print('all_gt: ', all_gt)


        all_im = all_im[:2]
        all_gt = all_gt[:2]



        prompts = []
        for gt in all_gt:
            cur_p = ""
            for i in range(len(gt)-1):
                cur_p += "<image>The text in the image is \'{}\'<|endofchunk|>".format(gt[i])
            cur_p += "<image>The text in the image is"
            prompts.append(cur_p)
        print('prompts: ', prompts[0])
        
        tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        lang_x = tokenizer(
            prompts,
            return_tensors="pt", padding=True,
        )

        
        # labels = lang_x["input_ids"].clone()


        # print('lang_x: ', lang_x["input_ids"])

        # print('decode: ', tokenizer.decode(lang_x["input_ids"][0]))

        generated_text = model.generate(
            vision_x=all_im.float(),
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=32,
            num_beams=3,
        )
        print("Generated text: ", tokenizer.decode(generated_text[0]))

        # output = model(
        #     vision_x=all_im.float(),
        #     lang_x=lang_x["input_ids"],
        #     attention_mask=lang_x["attention_mask"],
        #     labels=labels
        # )
        

        # print(output)

        # print(type(x))
        # for k in x.keys():
        #     print(k)
        #     print(x[k][0])
        # break
    """
    

    # start training
    #torch.save(runner.model.module.state_dict(), '/mnt/bn/zz-nas/Union14M/mmocr-dev-1.x/work_dirs/icl/test_model.pth')

    # print('load runner model')
    # runner.model.module.load_state_dict(
    #     torch.load('/mnt/bn/zz-nas/Union14M/checkpoints/maerec_b_union14m.pth')['state_dict']
    # )
    # runner.load_checkpoint('/mnt/bn/zz-nas/Union14M/checkpoints/maerec_b_union14m.pth')
    runner.train()


if __name__ == '__main__':
    main()
