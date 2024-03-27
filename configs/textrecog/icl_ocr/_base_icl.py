dictionary = dict(
    type='Dictionary',
    dict_file=  # noqa
    '{{ fileDirname }}/../../../dicts/english_digits_symbols_space.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

img_resize = (128, 32)

model = dict(
    type='Flamingo',
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))                                     # $$

train_pipeline = [
    dict(type='LoadImageFromFile', ignore_empty=True, min_size=0),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=img_resize),                            # $$
    dict(
        type='RandomApply',
        prob=0.5,
        transforms=[
            dict(
                type='RandomChoice',
                transforms=[
                    dict(
                        type='RandomRotate',
                        max_angle=15,
                    ),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomAffine',
                        degrees=15,
                        translate=(0.3, 0.3),
                        scale=(0.5, 2.),
                        shear=(-45, 45),
                    ),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomPerspective',
                        distortion_scale=0.5,
                        p=1,
                    ),
                ])
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(type='PyramidRescale'),
            dict(
                type='mmdet.Albu',
                transforms=[
                    dict(type='GaussNoise', var_limit=(20, 20), p=0.5),
                    dict(type='MotionBlur', blur_limit=7, p=0.5),
                ]),
        ]),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1),
        ]),
    dict(
        type='PackTextRecogInputs',                               # $$
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_resize),                                # $$
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',                               # $$
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=0, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=1, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=3, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
            ],
            [dict(type='Resize', scale=img_resize)],
            # add loading annotation after ``Resize`` because ground truth
            # does not need to do resize data transform
            [dict(type='LoadOCRAnnotations', with_text=True)],
            [
                dict(
                    type='PackTextRecogInputs',
                    meta_keys=('img_path', 'ori_shape', 'img_shape',
                               'valid_ratio'))
            ]
        ])
]
