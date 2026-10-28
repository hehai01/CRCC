# -*- coding: utf-8 -*-
"""CRCC / TokenMatcher style STAGE-1 training on LLCM (VIS + NIR).

Requires existing CRCC project modules and the previously installed LLCM
Market1501-style dataset classes (`llcm_rgb`, `llcm_ir`). Training code was
adapted directly from the user-provided SYSU stage-1 source: DBSCAN, reranking,
CSM splitting/merging, ClusterMemory and ClusterContrastTrainer_VIT_s1 are kept.

IMPORTANT:
- Only the official LLCM training split is used for optimization.
- `pid` is read by dataset management for stats/optional diagnostic only; all
  training targets come from DBSCAN/CSM pseudo labels.
- Evaluation uses LLCM test identities only for reporting. Selecting a "best"
  epoch using test data is test-set model selection; OFF by default.
- The unmodified CRCC splitting/merging code must support 9 camera slots.
- Needs `test_llcm.py` (provided in earlier conversation) beside this file for
  its verified LLCM test query/gallery/metric + preprocessing functions.

Example:
python Pre_train_llcm.py --data-dir data/LLCM --pretrained-path D:/HH/vit_base_ics_cfs_lup.pth --workers 0 --eval-step 5

Defaults for batch size, sampling, iters, seeds and learning rates follow the
user-provided SYSU argument parser (64 / 8 / 100 / seed=1 / lr0=0.00035).
"""
from __future__ import print_function, absolute_import
import argparse
import collections
import os
import os.path as osp
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from CRCC.CCM import optimal_transport4,conversion
from clustercontrast import datasets, models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer_VIT_s1,ClusterContrastTrainer_VIT_s2
from clustercontrast.evaluators import extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor, Preprocessor_aug
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import save_checkpoint,load_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from ChannelAug import ChannelAdapGray, ChannelRandomErasing, ChannelExchange, ChannelT

# These are the actual CSM implementations of your CRCC project.
from CSM.splitting import is_splitting
try:
    from CSM.Merging import is_Merging
except ImportError:
    from CRCC.CSM.Merging import is_Merging

# Cluster statistics are a diagnostic only. Never use ground-truth training IDs
# to generate loss labels, memory banks or sampling identities.
try:
    from evel_cluster1 import print_intra_acc,print_cm_acc
except ImportError:
    print_intra_acc = None

from test_llcm import (
    process_query_llcm, process_gallery_llcm, eval_llcm,
    make_dataloader as make_llcm_eval_loader,
    extract_features as extract_llcm_eval_features,
)


def get_data(name, data_dir):
    return datasets.create(name, root=data_dir)


def get_cluster_indices(labels):
    by_cluster = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        label = int(label)
        if label >= 0:
            by_cluster[label].append(idx)
    return by_cluster

def GetclusterSamples(pseudo_labels, dataset):

    assert len(dataset) == len(pseudo_labels), \
        f"Dataset size ({len(dataset)}) 和标签数量 ({len(pseudo_labels)}) 不匹配！"
    # 确保数据集已按文件名排序（与特征顺序一致）
    sorted_dataset = sorted(dataset)
    cluster_to_indices = collections.defaultdict(list)
    for idx, label in enumerate(pseudo_labels):
        if label != -1:
            cluster_to_indices[label].append(idx)  # 每个簇对应图片路径的索引

    # 初始化结果字典
    cluster_label_ture = collections.defaultdict(list)

    for label, indices in cluster_to_indices.items():
        for idx in indices:
            # 获取样本信息（sorted_dataset已排序，与features顺序一致）
            _, pid, _ = sorted_dataset[idx]
            cluster_label_ture[label].append(pid)

    return cluster_label_ture, cluster_to_indices

def get_train_loader_ir(args, dataset, height, width, batch_size, workers,num_instances, iters, trainset=None, no_cam=False, train_transformer=None,train_transformer1=None):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=getattr(dataset, "images_dir", None), transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_aug(train_set, root=getattr(dataset, "images_dir", None), transform=train_transformer,
                                        transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_train_loader_color(args, dataset, height, width, batch_size, workers,num_instances, iters, trainset=None, no_cam=False, train_transformer=None,train_transformer1=None):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=getattr(dataset, "images_dir", None), transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_aug(train_set, root=getattr(dataset, "images_dir", None), transform=train_transformer,
                                        transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None, test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=getattr(dataset, "images_dir", None), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader




def evaluate_llcm(model, args, epoch):
    """LLCM official query/gallery sampling; one or more trial averages.

    The reported per-epoch evaluation is DIAGNOSTIC. For an official test, use
    `test_llcm.py --direction both`, 10 trials and a preselected checkpoint.
    """
    eval_args = argparse.Namespace(**vars(args))
    eval_args.workers = 0
    model.eval()
    directions = [args.eval_direction] if args.eval_direction != 'both' else ['nir2vis', 'vis2nir']
    results = {}
    for direction in directions:
        mode_q, mode_g = (2, 1) if direction == 'nir2vis' else (1, 2)
        q_paths, q_labels, q_cams = process_query_llcm(args.data_dir, mode=mode_q)
        print(f"[LLCM {direction}] query images={len(q_paths)} epoch={epoch}")
        q_feat = extract_llcm_eval_features(model, make_llcm_eval_loader(q_paths, eval_args), mode_q, torch.device("cuda"))
        trial_metrics = []
        for trial in range(args.eval_trials):
            g_paths, g_labels, g_cams = process_gallery_llcm(args.data_dir, mode=mode_g, trial=trial)
            g_feat = extract_llcm_eval_features(model, make_llcm_eval_loader(g_paths, eval_args), mode_g, torch.device("cuda"))
            distmat = -(q_feat @ g_feat.T)
            cmc, mAP, mINP = eval_llcm(distmat, q_labels, g_labels, q_cams, g_cams)
            trial_metrics.append([cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP])
        avg = np.mean(np.asarray(trial_metrics), axis=0)
        results[direction] = dict(rank1=float(avg[0]), map=float(avg[4]))
        print("[LLCM {}] {} trials: R1={:.2%}, R5={:.2%}, R10={:.2%}, R20={:.2%}, mAP={:.2%}, mINP={:.2%}".format(direction, args.eval_trials, *avg))
    return results


def check_dataset(args):
    if not osp.isdir(args.data_dir):
        raise FileNotFoundError(f"LLCM dataset root not found: {args.data_dir}")
    rgb = get_data('llcm_rgb', args.data_dir)
    ir = get_data('llcm_ir', args.data_dir)
    print(f"Dataset ready: RGB train={len(rgb.train)}, IR train={len(ir.train)}")
    for name, ds in [('RGB', rgb), ('IR', ir)]:
        if not ds.train:
            raise RuntimeError(f"Empty {name} training set")
        cameras = sorted(set(x[2] for x in ds.train))
        print(f"{name} cam IDs (zero-based): {cameras}")
        if min(cameras) < 0 or max(cameras) > 8:
            raise RuntimeError(f"Unexpected LLCM {name} camid: {cameras}; expected 0..8")
        for path, _, _ in ds.train[:3]:
            if not osp.isfile(path):
                raise FileNotFoundError(f"{name} train image missing: {path}")
    return rgb, ir


def validate_args(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required by your CRCC stage-1 trainer')
    if not args.pretrained_path or not osp.isfile(args.pretrained_path):
        raise FileNotFoundError('Please pass --pretrained-path to existing pretrained ViT weights; received: ' + str(args.pretrained_path))
    if args.epochs <= 0 or args.iters <= 0:
        raise ValueError('--epochs and --iters must be positive')
    if args.num_instances <= 0:
        raise ValueError("--num-instances must be > 0")
    if args.batch_size <= 0 or args.batch_size % 2 or args.batch_size % args.num_instances or (args.batch_size // 2) % args.num_instances:
        raise ValueError('Batch size must be even, and both batch-size and batch-size/2 divisible by --num-instances')
    if args.workers < 0 or args.eval_step < 0:
        raise ValueError("--workers/--eval-step must be >= 0")
    if not (0 < args.eps <= 1) or not (0 < args.eps_ir <= 1) or not (0 < args.eps_rgb <= 1):
        raise ValueError('DBSCAN eps, eps-ir, eps-rgb should be in (0, 1] for this distance representation')
    if args.eval_trials < 1 or args.eval_trials > 10:
        raise ValueError('--eval-trials must be in 1..10')


def make_parser():
    """Keep the original CRCC/SYSU first-stage CLI and defaults.

    Dataset defaults and stage-1 log folder are changed for LLCM.  Options
    ending in 1 / lr1 / weight-decay1 are retained for parity with the SYSU
    script but apply only to stage 2, not to this stage-1 program.
    """
    parser = argparse.ArgumentParser(
        description='Adaptive Cluster-Level Optimization for USL-VI-ReID (LLCM Stage 1)'
    )
    working_dir = osp.dirname(osp.abspath(__file__))

    # data: match the user-provided SYSU configuration
    parser.add_argument('-d', '--dataset', type=str, default='LLCM')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--num-instances', type=int, default=8)

    # DBSCAN & Jaccard; --eps controls both modalities unless overridden
    parser.add_argument('--eps', type=float, default=0.6)
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help='reserved for stage 2 / multi-scale reliability')
    parser.add_argument('--eps-rgb', type=float, default=None,
                        help='optional override of --eps for RGB DBSCAN')
    parser.add_argument('--eps-ir', type=float, default=None,
                        help='optional override of --eps for IR DBSCAN')
    parser.add_argument('--k1', type=int, default=30)
    parser.add_argument('--k2', type=int, default=6)

    # model
    parser.add_argument('-a', '--arch', type=str, default='vit_base',
                        choices=models.names())
    parser.add_argument('-pp', '--pretrained-path', type=str,
                        default='D:/pythonProject/ACLO/examples/pretrained/vit_base_ics_cfs_lup.pth')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2)
    parser.add_argument('-mb', '--memorybank', type=str, default='CM',
                        choices=['CM', 'CMhcl'])
    parser.add_argument('--smooth', type=float, default=0)

    # ViT
    parser.add_argument('--drop-path-rate0', type=float, default=0.3)
    parser.add_argument('--drop-path-rate1', type=float, default=0.3,
                        help='retained for stage 2; unused by stage 1')
    parser.add_argument('--hw-ratio', type=int, default=2)
    parser.add_argument('--self-norm', action='store_true',
                        help='retained for compatibility; unused by current model factory call')
    parser.add_argument('--conv-stem', action='store_true')

    # optimizers; stage 1 uses lr0 and weight_decay0
    parser.add_argument('--lr0', type=float, default=0.00035)
    parser.add_argument('--lr1', type=float, default=0.000035,
                        help='reserved for stage 2')
    parser.add_argument('--weight-decay0', type=float, default=5e-4)
    parser.add_argument('--weight-decay1', type=float, default=5e-4,
                        help='reserved for stage 2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20,
                        help='retained for compatibility; stage 1 uses CosineAnnealingLR')

    # training
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05)

    # paths: do NOT join a Windows absolute path to the script directory.
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data', 'LLCM'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'),
                        help='base log directory; stage 1 uses its llcm_s1 subfolder')
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action='store_true')
    parser.add_argument('--no-cam', action='store_true')
    parser.add_argument('--cls-token-num', type=int, default=4)

    # LLCM-specific evaluation / diagnostics
    parser.add_argument('--test-batch', type=int, default=64)
    parser.add_argument('--eval-trials', type=int, default=1,
                        help='1 for periodic diagnostics; 10 for final LLCM evaluation')
    parser.add_argument('--eval-direction', choices=['nir2vis', 'vis2nir', 'both'],
                        default='nir2vis')
    parser.add_argument('--select-best-on-test', action='store_true',
                        help='not recommended: select model_best based on LLCM test labels')
    parser.add_argument('--report-train-gt-metrics', action='store_true',
                        help='training GT for ARI/purity reporting only, never loss')
    parser.add_argument('--check-data', action='store_true',
                        help='check converted LLCM training paths without launching training')
    return parser


def main():
    args = make_parser().parse_args()
    args.data_dir = osp.abspath(osp.expanduser(args.data_dir))
    args.logs_dir = osp.abspath(osp.expanduser(args.logs_dir))
    # Original SYSU code accepts logs root and creates logs/sysu_s1; do likewise.
    if osp.basename(osp.normpath(args.logs_dir)).lower() != 'llcm_s1':
        args.logs_dir = osp.join(args.logs_dir, 'llcm_s1')
    # Honor the single original --eps for both modalities, unless overriden.
    if args.eps_ir is None:
        args.eps_ir = args.eps
    if args.eps_rgb is None:
        args.eps_rgb = args.eps
    args.device = 'cuda'  # test_llcm.make_dataloader expects this attribute
    if args.dataset.lower() != 'llcm':
        raise ValueError('This script is dedicated to LLCM. Use --dataset LLCM.')
    if args.check_data:
        check_dataset(args)
        return
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    if args.report_train_gt_metrics and print_intra_acc is None:
        raise ImportError('Cannot import print_intra_acc from evel_cluster1')
    #main_worker1(args)
    main_worker2(args,'llcm_s1','llcm_s2')
def main_worker1(args):
    # Stage-1 ONLY. The uploaded SYSU extract called main_worker2() but did not
    # include its definition. Never call a missing stage-2 function here.
    global cls_token_num
    cls_token_num = args.cls_token_num
    data_dir = args.data_dir
    start_time = time.monotonic()
    best_R1 = -1.0
    best_mAP = 0.0
    best_epoch = -1
    os.makedirs(args.logs_dir, exist_ok=True)
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print('==========\nArgs:{}\n=========='.format(args))
    iters = args.iters
    print('==> Load unlabeled LLCM dataset')

    # Create datasets
    dataset_ir = get_data('llcm_ir', args.data_dir)
    dataset_rgb = get_data('llcm_rgb', args.data_dir)

    # Create model
    model = models.create(args.arch, img_size=(args.height, args.width), drop_path_rate=args.drop_path_rate0, pretrained_path=args.pretrained_path, hw_ratio=args.hw_ratio, conv_stem=args.conv_stem,cls_token_num=cls_token_num)
    model.cuda()
    model = nn.DataParallel(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr0, momentum=0.9, weight_decay=args.weight_decay0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Trainer
    trainer = ClusterContrastTrainer_VIT_s1(model)
    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = args.eps_ir
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = args.eps_rgb
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, 64, args.workers,testset=sorted(dataset_rgb.train))
            features_rgb, ture_ids_rgb = extract_features(model, cluster_loader_rgb, print_freq=50, mode=1)
            del cluster_loader_rgb

            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            ids_rgb = torch.cat([ture_ids_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            del ture_ids_rgb

            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width, 64, args.workers,testset=sorted(dataset_ir.train))
            features_ir, ture_ids_ir = extract_features(model, cluster_loader_ir, print_freq=50, mode=2)
            del cluster_loader_ir

            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            ids_ir = torch.cat([ture_ids_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            del ture_ids_ir

            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2, search_option=2)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2, search_option=2)
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
            del rerank_dist_rgb
            del rerank_dist_ir

            pseudo_labels_rgb1 = torch.from_numpy(pseudo_labels_rgb)
            pseudo_labels_ir1 = torch.from_numpy(pseudo_labels_ir)
            print_intra_acc(ids_rgb, ids_ir, pseudo_labels_rgb1, pseudo_labels_ir1)
            del pseudo_labels_ir1, pseudo_labels_rgb1

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        #cluster_index_rgb = get_cluster_indices(pseudo_labels_rgb)
        #cluster_index_ir = get_cluster_indices(pseudo_labels_ir)

        cluster_ids_ir, cluster_index_ir = GetclusterSamples(pseudo_labels_ir, dataset_ir.train)
        cluster_ids_rgb, cluster_index_rgb = GetclusterSamples(pseudo_labels_rgb, dataset_rgb.train)

        print("RGB聚类详情：")
        for label in cluster_ids_rgb.keys():
            print("伪标签{}：{}".format(label, cluster_ids_rgb[label]))
        print("=" * 40)
        print("IR聚类详情：")
        for label in cluster_ids_ir.keys():
            print("伪标签{}：{}".format(label, cluster_ids_ir[label]))

        print("=" * 40)


        pseudo_labels_rgb_crm = is_splitting(pseudo_labels_rgb, features_rgb, cluster_index_rgb)
        pseudo_labels_ir_crm = is_splitting(pseudo_labels_ir, features_ir, cluster_index_ir)

        pseudo_labels_rgb_hat, pseudo_labels_ir_hat = is_Merging(pseudo_labels_rgb_crm, pseudo_labels_ir_crm,features_rgb, features_ir)





        num_cluster_ir = len(set(pseudo_labels_ir_hat)) - (1 if -1 in pseudo_labels_ir_hat else 0)
        num_cluster_rgb = len(set(pseudo_labels_rgb_hat)) - (1 if -1 in pseudo_labels_rgb_hat else 0)

        # ClusterMemory indexes by label directly; non-dense labels would break
        # the memory bank even if the number of clusters appears correct.
        for modal_name, labels in (("IR", pseudo_labels_ir_hat), ("RGB", pseudo_labels_rgb_hat)):
            ids = sorted({int(v) for v in labels if int(v) >= 0})
            if ids != list(range(len(ids))):
                raise RuntimeError(f"{modal_name} CSM returned sparse cluster IDs; expected 0..{len(ids)-1}. Remap CSM labels before ClusterMemory.")

        if num_cluster_ir == 0 or num_cluster_rgb == 0:
            raise RuntimeError(f"No viable clusters after CSM at epoch {epoch}: IR={num_cluster_ir}, RGB={num_cluster_rgb}. Tune --eps-ir/--eps-rgb.")






        cluster_features_ir = generate_cluster_features(pseudo_labels_ir_hat, features_ir)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb_hat, features_rgb)
        memory_ir = ClusterMemory(768 * cls_token_num, num_cluster_ir, temp=args.temp, momentum=args.momentum, mode=args.memorybank, smooth=args.smooth, num_instances=args.num_instances).cuda()
        memory_rgb = ClusterMemory(768 * cls_token_num, num_cluster_rgb, temp=args.temp, momentum=args.momentum, mode=args.memorybank, smooth=args.smooth, num_instances=args.num_instances).cuda()

        if args.memorybank == 'CM':
            memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        elif args.memorybank == 'CMhcl':
            memory_ir.features = F.normalize(cluster_features_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb


        pseudo_labeled_dataset_ir = []
        ir_label = []
        pid_ir = []
        for i, ((fname, pid, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir_hat)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, int(label), cid))
                ir_label.append(int(label))
                pid_ir.append(pid)
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label = []
        pid_rgb = []
        for i, ((fname, pid, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb_hat)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, int(label), cid))
                rgb_label.append(int(label))
                pid_rgb.append(pid)
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))
        ir_label = torch.tensor(ir_label)
        rgb_label = torch.tensor(rgb_label)
        pid_rgb = torch.tensor(pid_rgb)
        pid_ir = torch.tensor(pid_ir)

        print_intra_acc(pid_rgb, pid_ir, rgb_label, ir_label)





        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        height = args.height
        width = args.width
        train_transformer_rgb = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5)
        ])

        train_transformer_rgb1 = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)
        ])

        transform_thermal = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])

        transform_thermal1 = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.ColorJitter(brightness=0.5),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5),
            ChannelT(probability=0.5)
        ])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                              args.batch_size, args.workers, args.num_instances, iters,
                                              trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,
                                              train_transformer=transform_thermal,
                                              train_transformer1=transform_thermal1)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                                  args.batch_size // 2, args.workers, args.num_instances, iters,
                                                  trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,
                                                  train_transformer=train_transformer_rgb,
                                                  train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir, train_loader_rgb, optimizer,print_freq=args.print_freq, train_iters=len(train_loader_ir))


        # LLCM has VIS/NIR directions rather than SYSU's all/indoor split.
        current = {}
        if args.eval_step > 0 and ((epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.epochs):
            current = evaluate_llcm(model, args, epoch)
            chosen_direction = 'nir2vis' if 'nir2vis' in current else 'vis2nir'
            score = current[chosen_direction]['rank1']
            if score > best_R1:
                best_R1 = score
                best_mAP = current[chosen_direction]['map']
                best_epoch = epoch
                if args.select_best_on_test:
                    save_checkpoint({
                        'state_dict': model.state_dict(), 'epoch': epoch + 1,
                        'best_mAP': best_mAP, 'best_R1': best_R1,
                    }, is_best=True, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            print('Epoch {}: eval={}, selected best-R1={:.2%} at epoch {} (test diagnostic)'.format(epoch, current, best_R1, best_epoch))

        lr_scheduler.step()
        # Always write a real, usable checkpoint for stage-2 initialization.
        state = {
            'state_dict': model.state_dict(), 'epoch': epoch + 1,
            'best_mAP': best_mAP, 'best_R1': best_R1,
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
        }
        # is_best False: TokenMatcher serialization writes checkpoint.pth.tar
        # without overwriting model_best.pth.tar unless user opts into test selection.
        save_checkpoint(state, is_best=False, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        if (epoch + 1) == args.epochs:
            # Last-epoch checkpoint is default for a model-selection-neutral report.
            import shutil
            shutil.copy2(osp.join(args.logs_dir, 'checkpoint.pth.tar'),
                         osp.join(args.logs_dir, 'model_last.pth.tar'))
        print('the learning rate is ', optimizer.state_dict()['param_groups'][0]['lr'])
        print('---------------------------------------------------------------------')

    print('Total running time:', timedelta(seconds=time.monotonic() - start_time))


def main_worker2(args, log_base_name, log_s2_name):
    start_epoch = 0
    best_mAP = 0
    best_R1 = 0
    best_epoch = 0
    data_dir = args.data_dir
    args.logs_dir = osp.join('logs' + '/' + log_s2_name)
    start_time = time.monotonic()
    global cls_token_num
    cls_token_num = args.cls_token_num

    print('==========\nArgs:{}\n=========='.format(args))
    iters = args.iters
    print('==> Load unlabeled LLCM dataset')

    # Create datasets
    dataset_ir = get_data('llcm_ir', args.data_dir)
    dataset_rgb = get_data('llcm_rgb', args.data_dir)

    # Create model
    model = models.create(args.arch, img_size=(args.height, args.width), drop_path_rate=args.drop_path_rate1, pretrained_path=args.pretrained_path, hw_ratio=args.hw_ratio, conv_stem=args.conv_stem,cls_token_num=cls_token_num)
    model.cuda()
    model = nn.DataParallel(model)
    checkpoint = load_checkpoint(osp.join("logs/"+log_base_name, 'model_last.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr1, momentum=0.9, weight_decay=args.weight_decay1)  ##########args.lr1
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    # Trainer
    trainer = ClusterContrastTrainer_VIT_s2(model)
    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = args.eps_ir
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = args.eps_rgb
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, 64, args.workers,testset=sorted(dataset_rgb.train))
            features_rgb, ture_ids_rgb = extract_features(model, cluster_loader_rgb, print_freq=50, mode=1)
            del cluster_loader_rgb

            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            ids_rgb = torch.cat([ture_ids_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            del ture_ids_rgb

            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width, 64, args.workers,testset=sorted(dataset_ir.train))
            features_ir, ture_ids_ir = extract_features(model, cluster_loader_ir, print_freq=50, mode=2)
            del cluster_loader_ir

            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            ids_ir = torch.cat([ture_ids_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            del ture_ids_ir

            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2, search_option=2)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2, search_option=2)
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
            del rerank_dist_rgb
            del rerank_dist_ir
            pseudo_labels_rgb1 = torch.from_numpy(pseudo_labels_rgb)
            pseudo_labels_ir1 = torch.from_numpy(pseudo_labels_ir)

        num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
        num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
        print('==> Statistics for   IR epoch {}: {} '.format(epoch, num_cluster_ir))
        print('==> Statistics for  RGB epoch {}: {} '.format(epoch, num_cluster_rgb))
        print_intra_acc(ids_rgb, ids_ir, pseudo_labels_rgb1, pseudo_labels_ir1)
        del pseudo_labels_ir1, pseudo_labels_rgb1


        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_index_rgb = get_cluster_indices(pseudo_labels_rgb)
        cluster_index_ir = get_cluster_indices(pseudo_labels_ir)

        #cluster_ids_ir, cluster_index_ir = GetclusterSamples(pseudo_labels_ir, dataset_ir.train)
        #cluster_ids_rgb, cluster_index_rgb = GetclusterSamples(pseudo_labels_rgb, dataset_rgb.train)
        print('====>CSM')
        pseudo_labels_rgb_crm = is_splitting(pseudo_labels_rgb, features_rgb, cluster_index_rgb)
        pseudo_labels_ir_crm = is_splitting(pseudo_labels_ir, features_ir, cluster_index_ir)
        pseudo_labels_rgb_hat, pseudo_labels_ir_hat = is_Merging(pseudo_labels_rgb_crm, pseudo_labels_ir_crm,features_rgb, features_ir)
        print('====>CSM done!!!')


        pseudo_labeled_dataset_ir = []
        ir_label = []
        pid_ir = []
        outlier_ir = 0
        filtered_features_ir = []  # 新增：存储过滤后的特征
        for i, ((fname, pid, cid), label) in enumerate(
                zip(sorted(dataset_ir.train), pseudo_labels_ir_hat)):  # zip将排序后的数据集与伪标签一一配对。
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))  # [('文件名'，label，camid)]跟之前的写的效果一样
                ir_label.append(label.item())  # 存取有效的伪标签除去-1噪声标签
                pid_ir.append(pid)
                filtered_features_ir.append(features_ir[i])  # 新增：保留对应索引的特征
            else:
                outlier_ir = outlier_ir + 1
        filtered_features_ir = torch.stack(filtered_features_ir, dim=0)

        pseudo_labeled_dataset_rgb = []
        rgb_label = []
        pid_rgb = []
        filtered_features_rgb = []  # 新增：存储过滤后的特征
        outlier_rgb = 0
        for i, ((fname, pid, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb_hat)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
                pid_rgb.append(pid)
                filtered_features_rgb.append(features_rgb[i])  # 新增：保留对应索引的特征
            else:
                outlier_rgb = outlier_rgb + 1
        filtered_features_rgb = torch.stack(filtered_features_rgb, dim=0)

        num_cluster_ir_new = len(set(pseudo_labels_ir_hat)) - (1 if -1 in pseudo_labels_ir_hat else 0)  # 统计聚类的个数，也是就是伪标签的个数
        num_cluster_rgb_new = len(set(pseudo_labels_rgb_hat)) - (1 if -1 in pseudo_labels_rgb_hat else 0)
        print('==> Statistics for   IR epoch {}: {} clusters outlier {}'.format(epoch, num_cluster_ir_new, outlier_ir))
        print('==> Statistics for  RGB epoch {}: {} clusters outlier {}'.format(epoch, num_cluster_rgb_new, outlier_rgb))
        ####转类型去除计算离群点
        ir_label = torch.tensor(ir_label)
        rgb_label = torch.tensor(rgb_label)
        pid_rgb = torch.tensor(pid_rgb)
        pid_ir = torch.tensor(pid_ir)
        print_intra_acc(pid_rgb, pid_ir, rgb_label, ir_label)

        centers_rgb = generate_cluster_features(pseudo_labels_rgb_hat, features_rgb)
        centers_ir = generate_cluster_features(pseudo_labels_ir_crm, features_ir)
        del features_ir
        del features_rgb

        # Create specific memory
        memory_ir = ClusterMemory(768 * cls_token_num, num_cluster_ir_new, temp=args.temp, momentum=args.momentum,
                                  mode=args.memorybank, smooth=args.smooth, num_instances=args.num_instances).cuda()
        memory_rgb = ClusterMemory(768 * cls_token_num, num_cluster_rgb_new, temp=args.temp, momentum=args.momentum,
                                   mode=args.memorybank, smooth=args.smooth, num_instances=args.num_instances).cuda()
        if args.memorybank == 'CM':
            memory_ir.features = F.normalize(centers_ir, dim=1).cuda()
            memory_rgb.features = F.normalize(centers_rgb, dim=1).cuda()
        elif args.memorybank == 'CMhcl':
            memory_ir.features = F.normalize(centers_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(centers_rgb.repeat(2, 1), dim=1).cuda()
        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb
        print('===>CCM')
        RGB_TO_IR, IR_TO_RGB = optimal_transport4(filtered_features_rgb, filtered_features_ir, rgb_label, ir_label,
                                                  centers_rgb, centers_ir)
        assign1 = np.argmax(RGB_TO_IR, axis=1)
        assign2 = np.argmax(IR_TO_RGB, axis=1)
        i2r = {}
        r2i = {}
        for i in range(num_cluster_rgb_new):
            r2i[i] = assign1[i]
        for i in range(num_cluster_ir_new):
            i2r[i] = assign2[i]
        print("CCM done")


        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        height = args.height
        width = args.width
        train_transformer_rgb = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5)
        ])

        train_transformer_rgb1 = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)
        ])

        transform_thermal = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])

        transform_thermal1 = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.ColorJitter(brightness=0.5),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5),
            ChannelT(probability=0.5)
        ])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                              args.batch_size, args.workers, args.num_instances, iters,
                                              trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,
                                              train_transformer=transform_thermal,
                                              train_transformer1=transform_thermal1)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                                  (args.batch_size), args.workers, args.num_instances, iters,
                                                  trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,
                                                  train_transformer=train_transformer_rgb,
                                                  train_transformer1=train_transformer_rgb1)
        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()
        trainer.train(epoch, train_loader_ir, train_loader_rgb, optimizer,print_freq=args.print_freq, train_iters=len(train_loader_ir), i2r=i2r, r2i=r2i)

        # LLCM has VIS/NIR directions rather than SYSU's all/indoor split.
        current = {}
        if args.eval_step > 0 and ((epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.epochs):
            current = evaluate_llcm(model, args, epoch)
            chosen_direction = 'nir2vis' if 'nir2vis' in current else 'vis2nir'
            score = current[chosen_direction]['rank1']
            if score > best_R1:
                best_R1 = score
                best_mAP = current[chosen_direction]['map']
                best_epoch = epoch
                if args.select_best_on_test:
                    save_checkpoint({
                        'state_dict': model.state_dict(), 'epoch': epoch + 1,
                        'best_mAP': best_mAP, 'best_R1': best_R1,
                    }, is_best=True, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            print('Epoch {}: eval={}, selected best-R1={:.2%} at epoch {} (test diagnostic)'.format(epoch, current,
                                                                                                    best_R1,
                                                                                                    best_epoch))

        lr_scheduler.step()
        # Always write a real, usable checkpoint for stage-2 initialization.
        state = {
            'state_dict': model.state_dict(), 'epoch': epoch + 1,
            'best_mAP': best_mAP, 'best_R1': best_R1,
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
        }
        # is_best False: TokenMatcher serialization writes checkpoint.pth.tar
        # without overwriting model_best.pth.tar unless user opts into test selection.
        save_checkpoint(state, is_best=False, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        if (epoch + 1) == args.epochs:
            # Last-epoch checkpoint is default for a model-selection-neutral report.
            import shutil
            shutil.copy2(osp.join(args.logs_dir, 'checkpoint.pth.tar'),
                         osp.join(args.logs_dir, 'model_last.pth.tar'))
        print('the learning rate is ', optimizer.state_dict()['param_groups'][0]['lr'])
        print('---------------------------------------------------------------------')

    print('Total running time:', timedelta(seconds=time.monotonic() - start_time))

if __name__ == '__main__':
    # Required for Windows torch.utils.data.DataLoader with num_workers > 0.
    main()
