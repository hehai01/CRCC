import time
import numpy as np
import torch
import torch.nn.functional as F
import collections
import copy

def create_instance_OT_matrix(labels_RGB, RGB_instance_IR_label, num_cls_rgb, num_cls_ir):
    # 初始化 RGB_TO_IR 矩阵为全零矩阵

    RGB_TO_IR = torch.zeros(num_cls_rgb, num_cls_ir, dtype=torch.int64)
    # 遍历每个样本
    for i in range(len(labels_RGB)):
        rgb_label = labels_RGB[i]
        ir_label = RGB_instance_IR_label[i]

        # 根据当前样本的 RGB 标签和匹配到的 IR 标签，在矩阵对应位置计数加 1
        RGB_TO_IR[rgb_label, ir_label] += 1
    return RGB_TO_IR


def get_cn(features_RGB, features_IR, labels_RGB, labels_IR):
    """
       Args:
           features_RGB: [N_rgb, D]
           features_IR:  [N_ir, D]
           labels_RGB:   [N_rgb]
           labels_IR:    [N_ir]

       Returns:
           cross_labels_RGB: [N_rgb]
           cross_labels_IR:  [N_ir]
           neighbor_indices_RGB: list
           neighbor_indices_IR:  list
       """

    features_RGB = F.normalize(features_RGB, p=2, dim=1)
    features_IR = F.normalize(features_IR, p=2, dim=1)

    # 只计算一次跨模态相似度
    # [N_rgb, N_ir]
    sim = torch.mm(features_RGB, features_IR.T)

    def neighbor_vote(sim_matrix, target_labels):
        """
        sim_matrix: [N_query, N_target]
        target_labels: [N_target]
        """
        # 每个 query 对应的最大跨模态相似度
        max_sim = sim_matrix.max(dim=1, keepdim=True).values

        # 相似度 >= 0.8 * max_sim 的样本作为邻居
        neighbor_mask = sim_matrix >= 0.8 * max_sim

        cross_labels = []
        neighbor_indices = []

        for i in range(sim_matrix.size(0)):
            indices = torch.where(neighbor_mask[i])[0]
            neighbor_indices.append(indices)

            # 邻居标签多数投票
            neighbor_labels = target_labels[indices]
            pseudo_label = torch.mode(neighbor_labels).values

            cross_labels.append(pseudo_label)

        cross_labels = torch.stack(cross_labels)

        return cross_labels, neighbor_indices

    # RGB -> IR
    cross_labels_RGB, neighbor_indices_RGB = neighbor_vote(sim, labels_IR)

    # IR -> RGB，直接使用 sim 的转置
    cross_labels_IR, neighbor_indices_IR = neighbor_vote(sim.T, labels_RGB)

    return cross_labels_RGB,cross_labels_IR

def conversion(pseudo_labels, result):
    """
    将单模态的伪标签转换为另一个模态的对应标签，同时保留-1标签不变
    Args:
        pseudo_labels (list or torch.Tensor): 源模态的伪标签列表
        result (dict): 从源模态标签到目标模态标签的映射
    Returns:
        torch.Tensor: 转换后的目标模态伪标签
    """
    # 如果输入是张量，先转为numpy数组
    if isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = pseudo_labels.cpu().numpy()
    # 过滤掉-1标签后检查缺失映射
    valid_labels = [label for label in pseudo_labels if label != -1]
    missing_labels = set(valid_labels) - set(result.keys())
    if missing_labels:
        raise ValueError(f"映射中缺少以下标签: {missing_labels}")

    # 使用向量化操作加速转换，保留-1不变
    def map_label(label):
        return result[label] if label != -1 else -1

    converted = np.vectorize(map_label)(pseudo_labels)

    # 转换回张量
    return torch.tensor(converted, dtype=torch.long)

def filter_valid_features(features, labels):
    """
    过滤掉标签为-1的特征
    :param features: 特征张量，形状为[N, 2048]
    :param labels: 标签张量，形状为[N]
    :return: 有效特征张量，形状为[M, 2048]，M为标签非-1的样本数
    """
    # 确保标签和特征维度匹配
    assert features.shape[0] == labels.shape[0], "特征和标签的样本数量必须一致"

    # 创建掩码：标签不等于-1的位置为True
    valid_mask = labels != -1

    # 应用掩码过滤特征
    valid_features = features[valid_mask]

    return valid_features


def pairwise_distance(x, y):
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)

    return dist_mat


def generate_one_hot_label(labels, num_cls):
    num_instance = labels.shape[0]
    one_hot_labels = torch.zeros(num_instance, num_cls)
    one_hot_order = torch.arange(num_instance)
    one_hot_labels[one_hot_order, labels] = 1

    return one_hot_labels


def sinkhorn_solver(P, lambda_sk=25, max_iter=500):
    tt = time.time()

    num_instance = P.shape[0]
    num_clusters = P.shape[1]

    alpha = torch.ones((num_instance, 1)) / num_instance  # initial value for alpha
    beta = torch.ones((num_clusters, 1)) / num_clusters  # initial value for beta

    inv_K = 1. / num_instance
    inv_N = 1. / num_clusters
    PS = torch.exp(-lambda_sk * P)

    err = 1e6
    step = 0

    while err > 1e-1 and step < max_iter:
        alpha = inv_K / (torch.mm(PS, beta))  # (KxN) @ (N,1) = K x 1
        beta_new = inv_N / (torch.mm(alpha.t(), PS)).T  # ((1,K) @ (KxN)).t() = N x 1
        if step % 10 == 0:
            err = np.nansum(np.abs(beta / beta_new - 1))
        beta = beta_new
        step += 1

    print("Sinkhorn-Knopp   Error: {:.3f}   Total step: {}   Total time: {:.3f}".format(err, step, time.time() - tt))
    P_out = torch.diag(alpha.squeeze()).mm(PS).mm(torch.diag(beta.squeeze()))

    return P_out

def optimal_transport( features_RGB, features_IR, labels_RGB, labels_IR, centers_RGB, centers_IR):

    num_cls_rgb = centers_RGB.shape[0]  # RGB类别数
    num_cls_ir = centers_IR.shape[0]  # IR类别数

    # 生成跨模态标签：将RGB实例分配到IR类别，IR实例分配到RGB类别
    # 1. RGB实例到IR类别中心的传输矩阵

    features_RGB = F.normalize(features_RGB,2,1)
    features_IR = F.normalize(features_IR,2,1)
    centers_RGB = F.normalize(centers_RGB,2,1)
    centers_IR = F.normalize(centers_IR,2,1)

    sim1 = (torch.mm(features_RGB,centers_IR.T)).exp()
    cost_R2I = 1/sim1
    #cost_matrix_rgb2ir = pairwise_distance(features_RGB, centers_IR)  # 形状 [N_rgb, num_cls_ir]
    transport_matrix_rgb2ir = sinkhorn_solver(cost_R2I,lambda_sk=25)  # 传输矩阵，形状 [N_rgb, num_cls_ir]
    y_rgb_cm = generate_one_hot_label(transport_matrix_rgb2ir.argmax(1), num_cls_ir)  # 硬分配的one-hot标签

    # 2. IR实例到RGB类别中心的传输矩阵
    #cost_matrix_ir2rgb = pairwise_distance(features_IR, centers_RGB)  # 形状 [N_ir, num_cls_rgb]
    sim2 = (torch.mm(features_IR,centers_RGB.T)).exp()
    cost_I2R = 1/sim2
    transport_matrix_ir2rgb = sinkhorn_solver(cost_I2R,lambda_sk=25)  # 传输矩阵，形状 [N_ir, num_cls_rgb]
    #print(transport_matrix_ir2rgb)

    ##
    #print(transport_matrix_ir2rgb*num_cls_rgb)

    y_ir_cm = generate_one_hot_label(transport_matrix_ir2rgb.argmax(1), num_cls_rgb)  # 硬分配的one-hot标签，这里也是一个矩阵

    # 提取硬标签（非one-hot格式，用于实例级对齐）
    RGB_instance_IR_label = y_rgb_cm.argmax(1)  # 每个RGB实例对应的IR类别索引（形状 [N_rgb]）
    IR_instance_RGB_label = y_ir_cm.argmax(1)  # 每个IR实例对应的RGB类别索引（形状 [N_ir]）

    RGB_TO_IR = create_instance_OT_matrix(labels_RGB, RGB_instance_IR_label, num_cls_rgb, num_cls_ir)
    IR_TO_RGB = create_instance_OT_matrix(labels_IR, IR_instance_RGB_label, num_cls_ir, num_cls_rgb)


    # 返回所有需要的变量（原始标签、跨模态标签、类别中心等）
    return RGB_TO_IR, IR_TO_RGB
    # RGB_instance_IR_label, IR_instance_RGB_label---实例级别的匹配的结果
    # y_rgb_cm, y_ir_cm--跨模态one-hot标签
    # RGB_TO_IR, IR_TO_RGB---匹配投票的矩阵



def optimal_transport2(centers_RGB, centers_IR):

    centers_RGB = F.normalize(centers_RGB,2,1)
    centers_IR = F.normalize(centers_IR,2,1)

    #聚类级别的OT
    #center_cost_matrix = pairwise_distance(centers_RGB, centers_IR)  # [num_cls_rgb, num_cls_ir]
    similarity = ((torch.mm(centers_RGB, centers_IR.T)) / 1).exp().cpu()
    cost = 1 / similarity
    #center_transport_matrix = sinkhorn_solver(center_cost_matrix)  # [num_cls_rgb, num_cls_ir]
    center_transport_matrix = sinkhorn_solver(cost)  # [num_cls_rgb, num_cls_ir]
    # RGB->IR中心映射（取每行最大值）
    rgb2ir_center_mapping = center_transport_matrix.argmax(1)  # [num_cls_rgb]
    # IR->RGB中心映射（取每列最大值）
    ir2rgb_center_mapping = center_transport_matrix.argmax(0)  # [num_cls_ir]

    return rgb2ir_center_mapping,ir2rgb_center_mapping



def get_anchor(features, labels, centers):

    valid_label_ids = sorted([int(l) for l in np.unique(labels) if l != -1])
    anchor = []
    for idx, lbl in enumerate(valid_label_ids):
        c = centers[idx]
        # 取出当前簇所有样本
        mask = (labels == lbl)
        cluster_samples = features[mask]
        # 每个样本到中心欧式距离
        dists = torch.norm(cluster_samples - c, p=2, dim=1)
        k = min(20, cluster_samples.shape[0])
        _, topk_indices = torch.topk(dists, k=k, largest=False)
        selected_samples = cluster_samples[topk_indices]
        anchor_feature = torch.mean(selected_samples, dim=0)
        anchor.append(anchor_feature)
    anchor = torch.stack(anchor, dim=0)
    return anchor



def optimal_transport3(features_RGB, features_IR, labels_RGB, labels_IR, centers_RGB, centers_IR):
    anchor_rgb = get_anchor(features_RGB, labels_RGB, centers_RGB)
    anchor_ir = get_anchor(features_IR, labels_IR, centers_IR)

    rgb2ir_anchor_mapping,ir2rgb_anchor_mapping = optimal_transport2(anchor_rgb,anchor_ir)

    rgb2ir_anchor_mapping = rgb2ir_anchor_mapping.cpu().numpy()
    ir2rgb_anchor_mapping = ir2rgb_anchor_mapping.cpu().numpy()

    mutual_pairs = []
    # RGB anchor i -> IR anchor j
    for i, j in enumerate(rgb2ir_anchor_mapping):
        # check reverse consistency: IR j -> RGB i
        if ir2rgb_anchor_mapping[j] == i:
            mutual_pairs.append((i, j))

    centers_RGB = F.normalize(centers_RGB, 2, 1)
    centers_IR = F.normalize(centers_IR, 2, 1)

    # 聚类级别的OT
    similarity = ((torch.mm(centers_RGB, centers_IR.T)) / 1).exp().cpu()
    cost = 1 / similarity

    print(mutual_pairs)
    #如果mutual_pairs不空
    # 将mutual_pairs相应位置上在cost矩阵中设为0
    if mutual_pairs:
        mutual_pairs = torch.tensor(mutual_pairs, dtype=torch.long)
        for pair in mutual_pairs:
            i = pair[0].item()
            j = pair[1].item()
            cost[i, j] = 0.0
    center_transport_matrix = sinkhorn_solver(cost)  # [num_cls_rgb, num_cls_ir]
    rgb2ir_center_mapping = center_transport_matrix.argmax(1)  # [num_cls_rgb]
    ir2rgb_center_mapping = center_transport_matrix.argmax(0)  # [num_cls_ir]
    return rgb2ir_center_mapping, ir2rgb_center_mapping


def optimal_transport4(features_RGB,features_IR,labels_RGB,labels_IR,centers_RGB,centers_IR,alpha=0.4):

    num_cls_rgb = centers_RGB.shape[0]  # RGB类别数
    num_cls_ir = centers_IR.shape[0]  # IR类别数
    features_RGB = F.normalize(features_RGB, p=2, dim=1)
    features_IR = F.normalize(features_IR, p=2, dim=1)
    centers_RGB = F.normalize(centers_RGB, p=2, dim=1)
    centers_IR = F.normalize(centers_IR, p=2, dim=1)

    # =========================================================
    # RGB -> IR
    # =========================================================
    sim1 = torch.mm(features_RGB, centers_IR.T).exp()
    cost_R2I = 1 / sim1

    transport_matrix_rgb2ir = sinkhorn_solver(cost_R2I,lambda_sk=25)

    RGB_instance_IR_label = transport_matrix_rgb2ir.argmax(dim=1)

    # =========================================================
    # IR -> RGB
    # =========================================================
    sim2 = torch.mm(features_IR, centers_RGB.T).exp()
    cost_I2R = 1 / sim2

    transport_matrix_ir2rgb = sinkhorn_solver(cost_I2R,lambda_sk=25)

    IR_instance_RGB_label = transport_matrix_ir2rgb.argmax(dim=1)

    # =========================================================
    # Instance-level cross-modal neighbor voting
    # =========================================================
    cross_labels_RGB, cross_labels_IR = get_cn(features_RGB,features_IR,labels_RGB,labels_IR)

    # =========================================================
    # RGB -> IR transport refinement
    # =========================================================

    # OT分配结果和邻居投票结果是否一致
    # [N_rgb]
    consistent_RGB = (RGB_instance_IR_label == cross_labels_RGB)

    refined_transport_rgb2ir = transport_matrix_rgb2ir.clone()

    rgb_no_reliable_cluster = 0
    # 遍历 RGB 的伪标签类别
    for cls in torch.unique(labels_RGB):
        # 当前RGB伪标签类别
        class_mask = (labels_RGB == cls)
        # 当前类别中：
        # OT跨模态标签 == 邻居跨模态标签
        reliable_mask = class_mask & consistent_RGB

        # 当前类别中不一致的样本
        unreliable_mask = class_mask & (~consistent_RGB)

        # 没有不一致样本则跳过
        if not unreliable_mask.any():
            continue

        # 如果该类别存在可靠样本
        if reliable_mask.any():

            # 同一RGB伪标签类别下，
            # 所有可靠样本的跨模态传输概率均值
            mean_transport = transport_matrix_rgb2ir[reliable_mask].mean(dim=0)

            # 对不可靠样本进行加权修正
            refined_transport_rgb2ir[unreliable_mask] = (alpha * transport_matrix_rgb2ir[unreliable_mask]+ (1 - alpha) * mean_transport.unsqueeze(0))
        else:
            # 当前簇不存在高置信度样本
            rgb_no_reliable_cluster += 1
    print("rgb:",rgb_no_reliable_cluster)

    # =========================================================
    # IR -> RGB 同样处理
    # =========================================================
    consistent_IR = (
        IR_instance_RGB_label == cross_labels_IR
    )

    refined_transport_ir2rgb = transport_matrix_ir2rgb.clone()

    ir_no_reliable_cluster = 0
    for cls in torch.unique(labels_IR):

        class_mask = (labels_IR == cls)

        reliable_mask = class_mask & consistent_IR
        unreliable_mask = class_mask & (~consistent_IR)

        if not unreliable_mask.any():
            continue

        if reliable_mask.any():

            mean_transport = transport_matrix_ir2rgb[
                reliable_mask
            ].mean(dim=0)

            refined_transport_ir2rgb[unreliable_mask] = (
                alpha * transport_matrix_ir2rgb[unreliable_mask]
                + (1 - alpha) * mean_transport.unsqueeze(0)
            )
        else:
            ir_no_reliable_cluster += 1
    print("ir:",ir_no_reliable_cluster)



    RGB_TO_IR = create_instance_OT_matrix(labels_RGB, refined_transport_rgb2ir.argmax(1), num_cls_rgb, num_cls_ir)
    IR_TO_RGB = create_instance_OT_matrix(labels_IR, refined_transport_ir2rgb.argmax(1), num_cls_ir, num_cls_rgb)

    return refined_transport_rgb2ir.argmax(1),refined_transport_ir2rgb.argmax(1),RGB_instance_IR_label,IR_instance_RGB_label,cross_labels_RGB, cross_labels_IR












