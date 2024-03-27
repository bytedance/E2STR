clean_IC13_textrecog_data_root = 'data/benchmark_cleansed/images/IC13'

clean_ic13 = dict(
    type='OCRDataset',
    data_root=clean_IC13_textrecog_data_root,
    ann_file='anno.json',
    test_mode=True,
    pipeline=None)
