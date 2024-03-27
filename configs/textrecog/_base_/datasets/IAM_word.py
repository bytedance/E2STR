IAM_word_data_root = 'data/test_data/IAM_word'

IAM_word_benchmark = dict(
    type='OCRDataset',
    data_prefix=dict(img_path='data/test_data/IAM_word/iam_words/words'),
    ann_file=f'{IAM_word_data_root}/test.json',
    test_mode=True,
    pipeline=None)
