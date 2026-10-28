# -*- coding: utf-8 -*-
"""TokenMatcher-style LLCM IR data statistics and train/query/gallery lists.

Each sample is (absolute_image_path, pid, zero_based_camid), matching the
existing sysu_rgb/sysu_ir/regdb_rgb/regdb_ir interfaces.
Official LLCM camera IDs c1..c9 are mapped to 0..8 for training.
The training PID is reindexed only to maintain existing dataset conventions;
unsupervised training MUST generate pseudo labels independently of these IDs.
"""
from __future__ import absolute_import, print_function

import os.path as osp
import re
from pathlib import Path

from ..utils.data import BaseImageDataset


class llcm_ir(BaseImageDataset):
    dataset_dir = 'ir_modify'
    _pattern = re.compile(r'^(-?\d+)_c(\d+)_', re.IGNORECASE)
    _exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, root, trial=0, verbose=True, **kwargs):
        super(llcm_ir, self).__init__()
        dataset_root = self._find_dataset_root(root)
        self.dataset_dir = osp.join(dataset_root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        if not train:
            raise RuntimeError('Empty LLCM IR training set: ' + self.train_dir)

        self.train, self.query, self.gallery = train, query, gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(gallery)

        if verbose:
            print('=> llcm_ir loaded')
            self.print_dataset_statistics(train, query, gallery)

    @classmethod
    def _find_dataset_root(cls, root):
        """Accept either LLCM root or TokenMatcher get_data(name, data_dir) root."""
        given = Path(root).expanduser().resolve()
        candidates = [given, given.parent, given / 'LLCM', given / 'llcm',
                      given.parent / 'LLCM', given.parent / 'llcm']
        for candidate in candidates:
            if (candidate / cls.dataset_dir / 'bounding_box_train').is_dir():
                return str(candidate)
        checked = ', '.join(str(x / cls.dataset_dir) for x in candidates)
        raise RuntimeError('LLCM IR Market-style directory not found. '
                           'Run prepare_llcm.py first. Checked: ' + checked)

    def _check_before_run(self):
        for p in [self.train_dir, self.query_dir, self.gallery_dir]:
            if not osp.isdir(p):
                raise RuntimeError('LLCM directory does not exist: ' + p)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = sorted(
            p for p in Path(dir_path).iterdir()
            if p.is_file() and p.suffix.lower() in self._exts
        )
        parsed = []
        for p in img_paths:
            m = self._pattern.match(p.name)
            if m is None:
                raise ValueError('Invalid Market1501-style LLCM filename: ' + str(p))
            pid, camid = int(m.group(1)), int(m.group(2))
            if pid == -1:
                continue
            if pid < 0:
                raise ValueError('Negative non-junk PID: ' + str(p))
            if not (1 <= camid <= 9):
                raise ValueError('LLCM camera ID must be in 1..9: ' + str(p))
            parsed.append((str(p.resolve()), pid, camid - 1))

        if relabel:
            # Stable reindexing to 0 ... (number of training identities - 1).
            id_mapping = {pid: i for i, pid in enumerate(sorted({x[1] for x in parsed}))}
            parsed = [(path, id_mapping[pid], camid) for path, pid, camid in parsed]
        return parsed
