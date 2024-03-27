WOST_data_root = 'data/test_data/WOST'

WOST_benchmark = dict(
    type='OCRDataset',
    data_root=WOST_data_root,
    ann_file='annotation.json',
    test_mode=True,
    pipeline=None)
