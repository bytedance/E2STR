# E2STR

The official implementation of E2STR: Multi-modal In-Context Learning Makes an Ego-evolving Scene Text Recognizer (CVPR-2024) [PDF](https://arxiv.org/pdf/2311.13120.pdf)

## environment

1. install mmocr 1.0.0

2. install requirements.txt

## data & model

1. Download Union14M-L from [Union14M-L](https://github.com/Mountchicken/Union14M)

2. Download the MAE pretrained ViT weight from [MAERec](https://github.com/Mountchicken/Union14M)

3. Download [OPT-125M](https://huggingface.co/facebook/opt-125m)

4. Download all the test dataset (listed in Table 1 and Table 2) and **modify all the data_root in configs/textrecog/_base_/datasets**. We may upload them later.

5. The 600k training data with character-wise annotations will be available later. But currently, the repository can also run well without this training data (i.e., you can perform in-context training with only Transform Strategy by modifying 'JSON FILE FOR CHARACTER-WISE POSITION INFORMATION' as None). Also refer to Table 4.

## train

1. stage1: vanilla STR training

modify 'MAE PRETRAIN WEIGHT PATH' / 'LM WEIGHT PATH' / 'CHECKPOINT SAVE PATH' / 'SAVE_NAME' in configs/textrecog/icl_ocr/stage1.py

```
sh run_stage1.sh
```

2. stage2: in-context training

modify 'STAGE-1 WEIGHT PATH' / 'LM WEIGHT PATH' / 'JSON FILE FOR CHARACTER-WISE POSITION INFORMATION' / 'CHECKPOINT SAVE PATH' / 'SAVE_NAME' in configs/textrecog/icl_ocr/stage2.py

```
sh run_stage2.sh
```

## evaluate

1. Construct the in-context pool (i.e., a json file) by randomly sample data from any target training set. The json file should be structured as follows:

```
[
{
    'img_path': ,
    'gt_text': 
}
]
```

2. Modify 'JSON FILE FOR IN-CONTEXT POOL' in configs/textrecog/icl_ocr/stage2.py


3. Run the following command to evaluate the model.

```
bash tools/dist_test.sh ./configs/textrecog/icl_ocr/S-stage2.py 'STAGE2-CHECKPOINT-PATH' 8
```

## Citation

If you find our models / code / papers useful in your research, please consider giving stars ⭐ and citations 📝

```
@article{zhao2023multi,
  title={Multi-modal In-Context Learning Makes an Ego-evolving Scene Text Recognizer},
  author={Zhao, Zhen and Huang, Can and Wu, Binghong and Lin, Chunhui and Liu, Hao and Zhang, Zhizhong and Tan, Xin and Tang, Jingqun and Xie, Yuan},
  journal={CVPR},
  year={2024}
}
```