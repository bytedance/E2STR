WordArt_data_root = 'data/test_data/WordArt'

WordArt_benchmark = dict(
    type='OCRDataset',
    data_root=WordArt_data_root,
    ann_file='annotation.json',
    test_mode=True,
    pipeline=None)
