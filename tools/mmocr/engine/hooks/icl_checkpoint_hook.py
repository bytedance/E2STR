# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path
from typing import List, Optional, Sequence, Union

from mmengine.dist import is_main_process
from mmengine.registry import HOOKS

from mmengine.hooks import CheckpointHook


import torch
import os

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class ICLCheckpointHook(CheckpointHook):

    def __init__(self,
                save_path: str = '/mnt/bn/zz-nas/Union14M/mmocr-dev-1.x/work_dirs/icl',
                save_name: str = 'Test',
                 interval: int = -1,
                 by_epoch: bool = True,
                 save_optimizer: bool = True,
                 save_param_scheduler: bool = True,
                 out_dir: Optional[Union[str, Path]] = None,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 save_best: Union[str, List[str], None] = None,
                 rule: Union[str, List[str], None] = None,
                 greater_keys: Optional[Sequence[str]] = None,
                 less_keys: Optional[Sequence[str]] = None,
                 file_client_args: Optional[dict] = None,
                 filename_tmpl: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 published_keys: Union[str, List[str], None] = None,
                 save_begin: int = 0,
                 **kwargs) -> None:
        super().__init__(interval,
                 by_epoch,
                 save_optimizer,
                 save_param_scheduler,
                 out_dir,
                 max_keep_ckpts,
                 save_last,
                 save_best,
                 rule,
                 greater_keys,
                 less_keys,
                 file_client_args,
                 filename_tmpl,
                 backend_args,
                 published_keys,
                 save_begin,
                 **kwargs)
        self.sava_path = save_path
        self.save_name = save_name

    def _save_checkpoint_with_step(self, runner, step, meta):
        # remove other checkpoints before save checkpoint to make the
        # self.keep_ckpt_ids are saved as expected
        if self.max_keep_ckpts > 0:
            # _save_checkpoint and _save_best_checkpoint may call this
            # _save_checkpoint_with_step in one epoch
            if len(self.keep_ckpt_ids) > 0 and self.keep_ckpt_ids[-1] == step:
                pass
            else:
                if len(self.keep_ckpt_ids) == self.max_keep_ckpts:
                    _step = self.keep_ckpt_ids.popleft()
                    if is_main_process():
                        ckpt_path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(_step))

                        if self.file_backend.isfile(ckpt_path):
                            self.file_backend.remove(ckpt_path)
                        elif self.file_backend.isdir(ckpt_path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(ckpt_path)

                self.keep_ckpt_ids.append(step)
                runner.message_hub.update_info('keep_ckpt_ids',
                                               list(self.keep_ckpt_ids))

        
        #ckpt_filename = self.filename_tmpl.format(step)

        save_location = os.path.join(self.sava_path, '{}_epoch_{}.pth'.format(self.save_name, step))

        torch.save(runner.model.module.state_dict(), save_location)

        # self.last_ckpt = self.file_backend.join_path(self.out_dir,
        #                                              ckpt_filename)
        # runner.message_hub.update_info('last_ckpt', self.last_ckpt)

        # runner.save_checkpoint(
        #     self.out_dir,
        #     ckpt_filename,
        #     self.file_client_args,
        #     save_optimizer=self.save_optimizer,
        #     save_param_scheduler=self.save_param_scheduler,
        #     meta=meta,
        #     by_epoch=self.by_epoch,
        #     backend_args=self.backend_args,
        #     **self.args)

        # Model parallel-like training should involve pulling sharded states
        # from all ranks, but skip the following procedure.
        if not is_main_process():
            return
