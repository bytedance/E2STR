clean_SVT_textrecog_data_root = 'data/benchmark_cleansed/images/SVT'

clean_svt = dict(
    type='OCRDataset',
    data_root=clean_SVT_textrecog_data_root,
    ann_file='anno.json',
    test_mode=True,
    pipeline=None)
