HOST_data_root = 'data/test_data/HOST'

HOST_benchmark = dict(
    type='OCRDataset',
    data_root=HOST_data_root,
    ann_file='annotation.json',
    test_mode=True,
    pipeline=None)

