MPSC_data_root = 'data/test_data/MPSC'

MPSC_benchmark = dict(
    type='OCRDataset',
    data_prefix=dict(img_path=MPSC_data_root),
    ann_file=f'{MPSC_data_root}/annotation_test.json',
    test_mode=True,
    pipeline=None)

