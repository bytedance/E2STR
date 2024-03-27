clean_IC15_textrecog_data_root = 'data/benchmark_cleansed/images/IC15'

clean_ic15 = dict(
    type='OCRDataset',
    data_root=clean_IC15_textrecog_data_root,
    ann_file='anno.json',
    test_mode=True,
    pipeline=None)
