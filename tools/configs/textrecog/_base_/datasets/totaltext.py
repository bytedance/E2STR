totaltext_data_root = 'data/test_data/totaltext'

totaltext_benchmark = dict(
    type='OCRDataset',
    data_root=totaltext_data_root,
    ann_file='annotation.json',
    test_mode=True,
    pipeline=None)

