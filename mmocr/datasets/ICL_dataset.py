# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data.sampler import Sampler
from mmengine.dataset import BaseDataset

from mmocr.registry import DATASETS
import torch


@DATASETS.register_module()
class ICLDataset(BaseDataset):

    def __getitem__(self, idx: int):
            """Get the idx-th image and data information of dataset after
            ``self.pipeline``, and ``full_init`` will be called if the dataset has
            not been fully initialized.

            During training phase, if ``self.pipeline`` get ``None``,
            ``self._rand_another`` will be called until a valid image is fetched or
            the maximum limit of refetech is reached.

            Args:
                idx (int): The index of self.data_list.

            Returns:
                dict: The idx-th image and data information of dataset after
                ``self.pipeline``.
            """
            # Performing full initialization by calling `__getitem__` will consume
            # extra memory. If a dataset is not fully initialized by setting
            # `lazy_init=True` and then fed into the dataloader. Different workers
            # will simultaneously read and parse the annotation. It will cost more
            # time and memory, although this may work. Therefore, it is recommended
            # to manually call `full_init` before dataset fed into dataloader to
            # ensure all workers use shared RAM from master process.
            if not self._fully_initialized:
                print_log(
                    'Please call `full_init()` method manually to accelerate '
                    'the speed.',
                    logger='current',
                    level=logging.WARNING)
                self.full_init()

            if self.test_mode:
                data = self.prepare_data(idx)
                if data is None:
                    raise Exception('Test time pipline should not get `None` '
                                    'data_sample')
                return [data]

            self.ICL_examples = 1
            all_data_num = self.ICL_examples + 1
            N = self.__len__()
            sample_indices = torch.randint(low=0, high=N, size=(all_data_num,))

            all_data = []
            
            for indice in sample_indices:

                success_fetch = False
            
                for _ in range(self.max_refetch + 1):
                    data = self.prepare_data(indice)
                    # Broken images or random augmentations may cause the returned data
                    # to be None
                    if data is None:
                        indice = self._rand_another()
                        continue
                    all_data.append(data)
                    success_fetch = True
                    break

                if not success_fetch:
                    raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                                    'Please check your image path and pipeline')
            
            return all_data

