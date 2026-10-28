import collections
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

# ----------------------------
# cluster center
# ----------------------------
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)

    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[label].append(features[i])

    centers = [
        torch.stack(centers[k], dim=0).mean(0)
        for k in sorted(centers.keys())
    ]

    return torch.stack(centers, dim=0)


# ----------------------------
# relabel
# ----------------------------
def old_to_new(cluster_index, pseudo_labels):

    valid_keys = []
    for key in cluster_index.keys():
        if key != -1 and key not in valid_keys:
            valid_keys.append(key)

    old_to_new_map = {}
    for i, k in enumerate(valid_keys):
        old_to_new_map[k] = i

    new_cluster_index = {
        old_to_new_map[k]: cluster_index[k]
        for k in valid_keys
    }

    new_pseudo_labels = pseudo_labels.copy()
    for i, l in enumerate(pseudo_labels):
        if l != -1:
            new_pseudo_labels[i] = old_to_new_map[l]

    return new_cluster_index, new_pseudo_labels


# ----------------------------
# main function
# ----------------------------
def is_Merging(pseudo_labels_rgb, pseudo_labels_ir, features_rgb, features_ir):

    pseudo_labels_rgb = np.array(pseudo_labels_rgb)
    pseudo_labels_ir = np.array(pseudo_labels_ir)

    # =========================================================
    # cluster centers
    # =========================================================
    centers_rgb = F.normalize(
        generate_cluster_features(pseudo_labels_rgb, features_rgb), dim=1
    )
    centers_ir = F.normalize(
        generate_cluster_features(pseudo_labels_ir, features_ir), dim=1
    )

    # cross similarity
    sim = torch.mm(centers_rgb, centers_ir.T)

    rgb2ir = torch.argmax(sim, dim=1).cpu().numpy()
    ir2rgb = torch.argmax(sim, dim=0).cpu().numpy()

    # =========================================================
    # cluster index
    # =========================================================
    cluster_index_rgb = collections.defaultdict(list)
    cluster_index_ir = collections.defaultdict(list)

    for i, l in enumerate(pseudo_labels_rgb):
        if l != -1:
            cluster_index_rgb[l].append(i)

    for i, l in enumerate(pseudo_labels_ir):
        if l != -1:
            cluster_index_ir[l].append(i)

    # =========================================================
    # RGB mutual nearest graph
    # =========================================================
    sim_rgb = torch.mm(centers_rgb, centers_rgb.T)
    sim_rgb.fill_diagonal_(-1e9)
    rgb_nn = torch.argmax(sim_rgb, dim=1).cpu().numpy()

    rgb_groups = defaultdict(list)
    for rgb_id, ir_id in enumerate(rgb2ir):
        rgb_groups[ir_id].append(rgb_id)

    merge_edges_rgb = set()

    for ir_id, rgb_list in rgb_groups.items():

        if len(rgb_list) < 2:
            continue

        for i in range(len(rgb_list)):
            for j in range(i + 1, len(rgb_list)):

                a, b = rgb_list[i], rgb_list[j]

                if rgb_nn[a] == b and rgb_nn[b] == a:
                    merge_edges_rgb.add((a, b) if a < b else (b, a))

    # =========================================================
    # apply RGB merge
    # =========================================================
    #print(merge_edges_rgb)
    for i, j in merge_edges_rgb:
        if i in cluster_index_rgb and j in cluster_index_rgb:
            cluster_index_rgb[i].extend(cluster_index_rgb[j])
            del cluster_index_rgb[j]
            pseudo_labels_rgb[pseudo_labels_rgb == j] = i

    # IMPORTANT: relabel AFTER merge
    cluster_index_rgb, pseudo_labels_rgb = old_to_new(
        cluster_index_rgb,
        pseudo_labels_rgb
    )

    # # # =========================================================
    # # # IR mutual nearest graph
    # # =========================================================
    sim_ir = torch.mm(centers_ir, centers_ir.T)
    sim_ir.fill_diagonal_(-1e9)
    ir_nn = torch.argmax(sim_ir, dim=1).cpu().numpy()

    ir_groups = defaultdict(list)
    for ir_id, rgb_id in enumerate(ir2rgb):
        ir_groups[rgb_id].append(ir_id)

    merge_edges_ir = set()

    for rgb_id, ir_list in ir_groups.items():

        if len(ir_list) < 2:
            continue

        for i in range(len(ir_list)):
            for j in range(i + 1, len(ir_list)):

                a, b = ir_list[i], ir_list[j]

                if ir_nn[a] == b and ir_nn[b] == a:
                    merge_edges_ir.add((a, b) if a < b else (b, a))
    #
    # # =========================================================
    # # apply IR merge
    # # =========================================================
    # #print(merge_edges_ir)
    for i, j in merge_edges_ir:
        if i in cluster_index_ir and j in cluster_index_ir:
            cluster_index_ir[i].extend(cluster_index_ir[j])
            del cluster_index_ir[j]
            pseudo_labels_ir[pseudo_labels_ir == j] = i

    # IMPORTANT: relabel AFTER merge
    cluster_index_ir, pseudo_labels_ir = old_to_new(
        cluster_index_ir,
        pseudo_labels_ir
    )

    return pseudo_labels_rgb, pseudo_labels_ir