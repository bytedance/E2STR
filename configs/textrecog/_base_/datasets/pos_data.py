pos_data_root = 'data/train_data/pos_data'

pos_train = dict(
    type='OCRDataset',
    data_root=pos_data_root,
    ann_file='annotation.json',
    pipeline=None)

