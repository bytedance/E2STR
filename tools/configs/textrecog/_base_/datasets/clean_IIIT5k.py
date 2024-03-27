clean_IIIT5k_textrecog_data_root = 'data/benchmark_cleansed/images/IIIT5k'

clean_iiit5k = dict(
    type='OCRDataset',
    data_root=clean_IIIT5k_textrecog_data_root,
    ann_file='anno.json',
    test_mode=True,
    pipeline=None)
