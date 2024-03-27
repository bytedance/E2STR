cocotext_data_root = 'data/test_data/cocotext'

cocotext_benchmark = dict(
    type='OCRDataset',
    data_root=cocotext_data_root,
    ann_file='annotation.json',
    test_mode=True,
    pipeline=None)

