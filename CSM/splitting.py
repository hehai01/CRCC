import torch
import collections
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

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


def is_splitting(pseudo_labels, features,cluster_index):

    # =========================================================
    # cluster centers
    # =========================================================
    centers = F.normalize(generate_cluster_features(pseudo_labels, features), dim=1)
    #print(pseudo_labels.shape)

    #K = centers.shape[0]
    #print(K)
    # 有效标签列表（过滤-1噪声）
    valid_label_ids = sorted([int(l) for l in np.unique(pseudo_labels) if l != -1])
    #print(len(valid_label_ids))

    # # 2. 计算每个簇的Si：簇内样本到中心平均L2距离
    Si_list = []
    for idx, lbl in enumerate(valid_label_ids):
        c = centers[lbl]
        # 取出当前簇所有样本
        mask = (pseudo_labels == lbl)
        cluster_samples = features[mask]
        # 每个样本到中心欧式距离
        dists = torch.norm(cluster_samples - c, p=2, dim=1)
        Si = torch.mean(dists)
        Si_list.append(Si)

    k = 2
    Si_tensor = torch.stack(Si_list)
    # 筛选大于0.6的索引和对应数值
    high_mask = Si_tensor > 0.60
    high_idx = high_mask.nonzero(as_tuple=True)[0]
    #high_values = Si_tensor[high_mask]

    clusters_to_split = high_idx.tolist()
    #print("clusters_to_split:", clusters_to_split)
    #print("对应Si值：", high_values.tolist())
        # 中心距离矩阵
    if clusters_to_split:
        new_pseudo_labels = pseudo_labels.copy()
        next_label = max(pseudo_labels) + 1 if -1 not in set(pseudo_labels) else max([l for l in pseudo_labels if l != -1]) + 1
        for label in clusters_to_split:
            # 找出需要分割的簇对应的样本索引
            #print('==>split_clusters label:{}'.format(label))
            indices = cluster_index[label]
            cluster_features = features[indices]
            # 使用 K-Means 算法进行分割
            kmeans = KMeans(n_clusters=k).fit(cluster_features)
            cluster_labels = kmeans.labels_
            # 保留原始簇的伪标签给第一个子簇
            for i, cluster_label_index in enumerate(cluster_labels):
                if cluster_label_index == 0:
                    new_pseudo_labels[indices[i]] = label
                else:
                    new_pseudo_labels[indices[i]] = next_label + cluster_label_index - 1
            next_label += k - 1

        # 更新 cluster_index
        cluster_index = {}
        for i, label in enumerate(new_pseudo_labels):
            if label == -1:  # 跳过-1标签，不纳入簇索引
                continue
            if label not in cluster_index:
                cluster_index[label] = []
            cluster_index[label].append(i)
        pseudo_labels = new_pseudo_labels
        cluster_index, pseudo_labels = old_to_new(cluster_index, pseudo_labels)
    return pseudo_labels


def old_to_new(cluster_index, pseudo_labels):
    # 收集有效标签（仅包含非-1，按首次出现顺序）
    valid_keys = []
    for key in sorted(cluster_index.keys()):
        if key not in valid_keys:
            valid_keys.append(key)

    # 构建旧标签到新标签的映射（从0开始连续编号）
    old_to_new_map = {old_label: new_label for new_label, old_label in enumerate(valid_keys)}

    # 生成新的簇索引（使用新标签）
    new_cluster_index = {
        old_to_new_map[old_label]: indices
        for old_label, indices in cluster_index.items()
    }
    # 更新伪标签，保留-1标签不变
    new_pseudo_labels = pseudo_labels.copy()
    for idx, label in enumerate(pseudo_labels):
        if label != -1:
            new_pseudo_labels[idx] = old_to_new_map[label]

    return new_cluster_index, new_pseudo_labels