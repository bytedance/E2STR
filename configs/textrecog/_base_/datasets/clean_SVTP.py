clean_SVTP_textrecog_data_root = 'data/benchmark_cleansed/images/SVTP'

clean_svtp = dict(
    type='OCRDataset',
    data_root=clean_SVTP_textrecog_data_root,
    ann_file='anno.json',
    test_mode=True,
    pipeline=None)
