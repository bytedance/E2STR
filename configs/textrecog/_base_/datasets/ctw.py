ctw_data_root = 'data/test_data/ctw'

ctw_benchmark = dict(
    type='OCRDataset',
    data_root=ctw_data_root,
    ann_file='annotation.json',
    test_mode=True,
    pipeline=None)
