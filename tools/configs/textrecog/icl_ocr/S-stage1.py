# modify :     (1) model    (2) schedual    (3) default hooks   (4) batch_size   (5) dataset   (6) predict

_base_ = [
    '_base_icl.py',
    '../_base_/datasets/pos_data.py',
    '../_base_/datasets/MPSC.py',
    '../_base_/datasets/IAM_word.py',
    '../_base_/datasets/union14m_train.py',
    '../_base_/datasets/clean_IC13.py',
    '../_base_/datasets/clean_IC15.py',
    '../_base_/datasets/clean_cute80.py',
    '../_base_/datasets/clean_IIIT5k.py',
    '../_base_/datasets/clean_SVT.py',
    '../_base_/datasets/clean_SVTP.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/icl_ocr/S-stage1.py',
]

model = dict(
    type='Flamingo',
    data_len = 1,
    prompt_type = 0, 
    model_type='S',
    vit_type='small',
    pos_process = False,
    load_weight = None,
    load_mae = 'MAE PRETRAIN WEIGHT PATH',
    unfreeze_vit = True,
    unfreeze_decoder = True,
    lm_path = 'LM WEIGHT PATH',
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))

# dataset settings
train_list = [
    _base_.union14m_challenging, _base_.union14m_hard, _base_.union14m_medium,
    _base_.union14m_normal, _base_.union14m_easy, _base_.pos_train
]

val_list = [
    _base_.MPSC_benchmark,
    _base_.IAM_word_benchmark,
]

test_list = [
    _base_.clean_cute80, _base_.clean_iiit5k,
    _base_.clean_svt, _base_.clean_svtp,
    _base_.clean_ic13, _base_.clean_ic15,
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=100),
                    checkpoint=dict(type='ICLCheckpointHook', interval=1, 
                    save_path='CHECKPOINT SAVE PATH', save_name='SAVE_NAME'))

auto_scale_lr = dict(base_batch_size=512)

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)
val_dataset = dict(
    type='ConcatDataset', datasets=val_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)


val_evaluator = dict(
    dataset_prefixes=['MPSC', 'IAM_word'])


test_evaluator = dict(
    dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])