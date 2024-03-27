clean_cute80_textrecog_data_root = 'data/benchmark_cleansed/images/CUTE80'

clean_cute80 = dict(
    type='OCRDataset',
    data_root=clean_cute80_textrecog_data_root,
    ann_file='anno.json',
    test_mode=True,
    pipeline=None)
