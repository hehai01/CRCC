import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from IPython import embed
import math









































class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=0, epsilon=0.1, topk_smoothing=False):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.k = 1 if not topk_smoothing else self.num_classes // 50

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        if self.k > 1:
            topk = torch.argsort(-log_probs)[:, :self.k]
            targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)
            targets += torch.zeros_like(log_probs).scatter_(1, topk, self.epsilon / self.k)
        else:
            targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class SoftEntropy(nn.Module):
    def __init__(self, input_prob=False):
        super(SoftEntropy, self).__init__()
        self.input_prob = input_prob
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        if self.input_prob:
            loss = (- targets.detach() * log_probs).mean(0).sum()
        else:
            loss = (- F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
        return loss


class SoftEntropySmooth(nn.Module):
    def __init__(self, epsilon=0.1):
        super(SoftEntropySmooth, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, soft_targets, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        soft_targets = F.softmax(soft_targets, dim=1)
        smooth_targets = (1 - self.epsilon) * targets + self.epsilon * soft_targets
        loss = (- smooth_targets.detach() * log_probs).mean(0).sum()
        return loss


class Softmax(nn.Module):

    def __init__(self, feat_dim, num_class, temp=0.05):
        super(Softmax, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.temp = temp

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        outputs = feats.mm(kernel_norm)
        outputs /= self.temp
        loss = F.cross_entropy(outputs, labels)
        return loss


class CircleLoss(nn.Module):
    """Implementation for "Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    Note: this is the classification based implementation of circle loss.
    """

    def __init__(self, feat_dim, num_class, margin=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.gamma = gamma

        self.O_p = 1 + margin
        self.O_n = -margin
        self.delta_p = 1 - margin
        self.delta_n = margin

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        index_pos = torch.zeros_like(cos_theta)
        index_pos.scatter_(1, labels.data.view(-1, 1), 1)
        index_pos = index_pos.bool()
        index_neg = torch.ones_like(cos_theta)
        index_neg.scatter_(1, labels.data.view(-1, 1), 0)
        index_neg = index_neg.bool()

        alpha_p = torch.clamp_min(self.O_p - cos_theta.detach(), min=0.)
        alpha_n = torch.clamp_min(cos_theta.detach() - self.O_n, min=0.)

        logit_p = alpha_p * (cos_theta - self.delta_p)
        logit_n = alpha_n * (cos_theta - self.delta_n)

        output = cos_theta * 1.0
        output[index_pos] = logit_p[index_pos]
        output[index_neg] = logit_n[index_neg]
        output *= self.gamma

        return F.cross_entropy(output, labels)


class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, feat_dim, num_class, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = feat_dim
        self.out_features = num_class
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(feat_dim, num_class))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight, dim=1))
        cosine = torch.mm(F.normalize(input), F.normalize(self.weight, dim=0))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return F.cross_entropy(output, label)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features = ' + str(self.in_features) \
            + ', out_features = ' + str(self.out_features) \
            + ', s = ' + str(self.s) \
            + ', m = ' + str(self.m) + ')'




class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        print('Initializing FocalLoss for training: alpha={}, gamma={}'.format(self.alpha, self.gamma))

    def forward(self, input, target):
        assert input.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1

        logpt = F.log_softmax(input, dim=1)
        logpt_gt = logpt.gather(1, target.unsqueeze(1))
        logpt_gt = logpt_gt.view(-1)
        pt_gt = logpt_gt.exp()
        assert logpt_gt.size() == pt_gt.size()

        loss = -self.alpha * (torch.pow((1 - pt_gt), self.gamma)) * logpt_gt

        return loss.mean()


class LabelRefineLoss(nn.Module):
    def __init__(self, lambda1=0.0):
        super(LabelRefineLoss, self).__init__()
        self.lambda1 = lambda1
        print('Initializing LabelRefineLoss for training: lambda1={}'.format(self.lambda1))

    def forward(self, input, target):
        assert input.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1

        logpt = F.log_softmax(input, dim=1)
        logpt_gt = logpt.gather(1, target.unsqueeze(1))
        logpt_gt = logpt_gt.view(-1)
        logpt_pred, _ = torch.max(logpt, 1)
        logpt_pred = logpt_pred.view(-1)
        assert logpt_gt.size() == logpt_pred.size()
        loss = - (1 - self.lambda1) * logpt_gt - self.lambda1 * logpt_pred

        return loss.mean()

class FocalTopLoss(nn.Module):
    def __init__(self, top_percent=0.7):
        super(FocalTopLoss, self).__init__()
        self.top_percent = top_percent

    def masked_softmax_multi_focal(self, vec, targets=None, dim=1):
        exps = torch.exp(vec)
        one_hot_pos = F.one_hot(targets, num_classes=exps.shape[1])

        one_hot_neg = one_hot_pos.new_ones(size=one_hot_pos.shape)
        one_hot_neg = one_hot_neg - one_hot_pos

        neg_exps = exps.new_zeros(size=exps.shape)
        neg_exps[one_hot_neg > 0] = exps[one_hot_neg > 0]
        ori_neg_exps = neg_exps
        neg_exps = neg_exps / neg_exps.sum(dim=1, keepdim=True)

        new_exps = exps.new_zeros(size=exps.shape)
        new_exps[one_hot_pos > 0] = exps[one_hot_pos > 0]

        sorted, indices = torch.sort(neg_exps, dim=1, descending=True)
        sorted_cum_sum = torch.cumsum(sorted, dim=1)
        sorted_cum_diff = (sorted_cum_sum - self.top_percent).abs()
        sorted_cum_min_indices = sorted_cum_diff.argmin(dim=1)

        min_values = sorted[torch.range(0, sorted.shape[0] - 1).long(), sorted_cum_min_indices]
        min_values = min_values.unsqueeze(dim=-1) * ori_neg_exps.sum(dim=1, keepdim=True)
        ori_neg_exps[ori_neg_exps < min_values] = 0

        new_exps[one_hot_neg > 0] = ori_neg_exps[one_hot_neg > 0]

        masked_sums = exps.sum(dim, keepdim=True)
        return new_exps / masked_sums

    def forward(self, input, target):
        masked_sim = self.masked_softmax_multi_focal(input, target)
        return F.nll_loss(torch.log(masked_sim + 1e-6), target)  


class RCLoss(nn.Module):
    def __init__(self, sigma=1, delta=1):
        super(RCLoss, self).__init__()
        self.sigma = sigma
        self.delta = delta

    def forward(self, s_emb, t_emb):
        N = len(s_emb)
        s_emb = F.normalize(s_emb,2,1)
        t_emb = F.normalize(t_emb,2,1)
        S_dist = torch.cdist(s_emb, s_emb)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)
        #print("S_dist:",S_dist)
        with torch.no_grad():
            T_dist = torch.cdist(t_emb, t_emb)
            #print("T_dist:",T_dist)
            W = torch.exp(-T_dist / self.sigma)#高斯核函数计算权值
            #print("高斯核函数计算权值:",W)

            identity_matrix = torch.eye(N).cuda(non_blocking=True)#构建单位矩阵
            pos_weight = W * (1 - identity_matrix) #除去对角线上的权值
            pos_weight = torch.where(pos_weight > 0.25,pos_weight,torch.zeros_like(pos_weight))
            #print("拉近的权重：",pos_weight)
            neg_weight = torch.relu((0.4 - W)) * (1 - identity_matrix)*2 #权值取1-
            #print("推远的权重：", neg_weight)
        pull_losses = torch.relu(S_dist).pow(2) * pos_weight#将输入张量中小于 0 的元素置为 0，大于等于 0 的元素保持不变。
        push_losses = torch.relu(self.delta - (0.4*S_dist)).pow(2) * neg_weight
        #print("拉近的loss",pull_losses.sum())
        #print("推远的loss",push_losses.sum())
        loss = (pull_losses.sum() + push_losses.sum()) / (N * (N - 1))#对于 N 个样本，样本对的总数为 N * (N - 1)
        return loss,W

def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [B, m, d]
      y: pytorch Variable, with shape [B, n, d]
    Returns:
      dist: pytorch Variable, with shape [B, m, n]
    """
    B = x.size(0)
    m, n = x.size(1), y.size(1)
    x_norm = torch.pow(x, 2).sum(2, keepdim=True).sqrt().expand(B, m, n)
    y_norm = torch.pow(y, 2).sum(2, keepdim=True).sqrt().expand(B, n, m).transpose(-2, -1)
    xy_intersection = x @ y.transpose(-2, -1)
    dist = xy_intersection/(x_norm * y_norm)
    return torch.abs(dist)
class Dissimilar(object):
    def __init__(self, dynamic_balancer=True):
        self.dynamic_balancer = dynamic_balancer

    def __call__(self, features):
        B, N, C = features.shape
        dist_mat = cosine_dist(features, features)  # B*N*N
        # dist_mat = euclidean_dist(features, features)
        # 上三角index
        top_triu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        _dist = dist_mat[:, top_triu]

        # 1.用softmax替换平均，使得相似度更高的权重更大
        if self.dynamic_balancer:
            weight = F.softmax(_dist, dim=-1)
            dist = torch.mean(torch.sum(weight * _dist, dim=1))
        # 2.直接平均
        else:
            dist = torch.mean(_dist, dim=(0, 1))
        return dist
    @staticmethod
    def cosine_similarity(x, y):
        # 特征归一化到单位球面上（增强数值稳定性）
        x_norm = F.normalize(x, dim=-1)  # [B, D]
        y_norm = F.normalize(y, dim=-1)  # [B, D]
        cos_sim = torch.sum(x_norm * y_norm, dim=-1)  # [B]

        return torch.abs(cos_sim)

def Bg_kl(logits1, logits2):####输入:(60,206),(60,206)
      KL = nn.KLDivLoss(reduction='batchmean')
      kl_loss_12 = KL(F.log_softmax(logits1, 1), F.softmax(logits2, 1))
      kl_loss_21 = KL(F.log_softmax(logits2, 1), F.softmax(logits1, 1))
      bg_loss_kl = kl_loss_12 + kl_loss_21
      return  bg_loss_kl
def get_centers(mapping,centers):
      matched_centers = []
      #print(centers.size())
      for label in mapping.keys():
            get_label = mapping[label]
            matched_centers.append(centers[get_label])
      centers1 =torch.stack(matched_centers, dim=0)
      #print(centers1.size())
      return centers1
class CrossLearning(torch.nn.Module):
    def __init__(self):
        super(CrossLearning, self).__init__()
        self.sigma = 1.0
        self.ratio = 1.0
    def forward(self, features, centers_intra, centers_inter, inter_mapping):

        features = F.normalize(features,2,1)
        dist_1_2 = torch.mm(features, centers_inter.T) / self.sigma
        get_center = get_centers(inter_mapping,centers_intra)
        dist_1_1 = torch.mm(features, get_center.T) / self.ratio
        loss = Bg_kl(dist_1_1,dist_1_2)
        return loss
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
class TripletLoss_ADP(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha=1, gamma=1, square=0):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.square = square

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap * self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an * self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        # ranking_loss = nn.SoftMarginLoss(reduction = 'none')
        # loss1 = ranking_loss(closest_negative - furthest_positive, y)

        # squared difference
        if self.square == 0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma * (closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            diff_pow = torch.clamp_max(diff_pow, max=88)

            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)

            loss = self.ranking_loss(diff_pow, y)

        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)

        # compute accuracy
        #correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss

class OriTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        inputs = F.normalize(inputs, p=2, dim=1)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct



def pdist(A, B, squared = False, eps = 1e-12):
    D = A.pow(2).sum(1) + (-2) * B.mm(A.t())
    D = (B.pow(2).sum(1) + D.t()).clamp(min=eps)
    
    if not squared:
        D = D.sqrt()
        
    if torch.equal(A,B):
        D = D.clone()
        D[range(len(A)), range(len(A))] = 0
        
    return D
    
class Relaxed_Contra(nn.Module):
    """Relaxed Contrative loss function. """
    def __init__(self, sigma=1, delta=1):
        super(Relaxed_Contra, self).__init__()
        
        self.sigma = sigma
        self.delta = delta
        
    def forward(self, s_emb, t_emb):
        s_emb = F.normalize(s_emb, p=2, dim=1)
        
        T_dist = pdist(t_emb, t_emb, False)
        dist_mean = T_dist.mean(1, keepdim=True)
        T_dist = T_dist / dist_mean
            
        with torch.no_grad():
            S_dist = pdist(s_emb, s_emb, False)
            P = torch.exp(-S_dist.pow(2) / self.sigma)
        
        pos_weight = P
        neg_weight = 1-P
        
        pull_losses = torch.relu(T_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - T_dist).pow(2) * neg_weight

        pull_losses = pull_losses[T_dist>0]
        push_losses = push_losses[T_dist>0]
        loss = (pull_losses.sum() + push_losses.sum()) / (len(s_emb) * (len(s_emb) - 1))

        return loss
    

class CenterLoss(nn.Module):


    def __init__(self, num_classes, feat_dim, reduction='mean'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.reduction = reduction

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).to(device=x.device, dtype=torch.long)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        loss = distmat * mask.float()

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class MCLoss2(nn.Module):
    def __init__(self, tau=1, theta=0.8):
        super(MCLoss2, self).__init__()
        self.tau = tau
        self.theta = theta

    def forward(self, features, centers):
        # 归一化特征和原型
        features = F.normalize(features, p=2, dim=1)
        # centers = F.normalize(centers, p=2, dim=1)  #中心归一化

        # 计算相似度矩阵
        sim = torch.mm(features, centers.T)  # N x C

        # 筛选正样本掩码
        with torch.no_grad():
            assign = sim.argmax(dim=1)  # (B,)
            pos_mask = F.one_hot(assign, centers.size(0)).float()  # (B,C)

        # 计算指数相似度
        exp_sim = torch.exp(sim / self.tau)
        exp_sim_pos = pos_mask * exp_sim
        a = exp_sim.sum(dim=1) #[1,64]
        b = exp_sim_pos.sum(dim=1)
        #print(a)
        #print(b)
        loss = -torch.log(b/a).mean()

        return loss


class RCcentersLoss_Aligned(nn.Module):
    def __init__(self,  delta=0.5, temperature=0.1):
        super().__init__()
        self.delta = delta
        self.temperature = temperature

    def forward(self, features, centers):
        """
        Args:
            features: (B, D) 输入特征
            centers: (C, D) 类别中心
            labels: (B,) 可选，提供特征对应的中心索引
        """
        # 归一化处理
        features = F.normalize(features, p=2, dim=1)  # (B,D)
        centers = F.normalize(centers, p=2, dim=1)  # (C,D)
        # 计算相似度矩阵
        sim_matrix = torch.mm(features, centers.T)  # (B,C)

        # 无监督模式：动态分配最近中心
        with torch.no_grad():
            assign = sim_matrix.argmax(dim=1)  # (B,)
            pos_mask = F.one_hot(assign, centers.size(0)).float()  # (B,C)
            neg_mask = 1 - pos_mask

        # 权重计算（基于相似度）
        with torch.no_grad():
            weights = torch.exp(sim_matrix / self.temperature)  # (B,C)
            pos_weights = weights * pos_mask
            neg_weights = (1 - weights) * neg_mask

        # 损失计算
        pos_loss = torch.relu(1 - sim_matrix).pow(2) * pos_weights  # Pull损失
        neg_loss = torch.relu(sim_matrix - self.delta).pow(2) * neg_weights  # Push损失
        # 加权求和
        loss = (pos_loss.sum() + neg_loss.sum()) / (pos_mask.sum() + neg_mask.sum() + 1e-8)
        return loss


class CrossModalMMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(CrossModalMMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, source_labels, target_labels,
                        source_to_target_map, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        # 创建跨模态匹配矩阵
        match_matrix = torch.zeros((n_samples, n_samples), device=source.device)

        # 构建匹配矩阵：
        # 1. 相同模态相同ID
        # 2. 跨模态匹配ID
        for i in range(source.size(0)):
            for j in range(source.size(0)):
                if source_labels[i] == source_labels[j]:
                    match_matrix[i, j] = 1.0  # 同模态同ID

        for i in range(target.size(0)):
            for j in range(target.size(0)):
                if target_labels[i] == target_labels[j]:
                    match_matrix[i + source.size(0), j + source.size(0)] = 1.0  # 同模态同ID

        # 跨模态匹配
        for i in range(source.size(0)):
            mapped_id = source_to_target_map[source_labels[i].item()]
            for j in range(target.size(0)):
                if target_labels[j] == mapped_id:
                    match_matrix[i, j + source.size(0)] = 1.0
                    match_matrix[j + source.size(0), i] = 1.0

        total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            # 只考虑匹配对的距离
            valid_distances = L2_distance[match_matrix.bool()]
            bandwidth = torch.mean(valid_distances) if len(valid_distances) > 0 else 1.0

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 只对匹配对应用核函数
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) * match_matrix
                      for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    def forward(self, source, target, source_labels, target_labels, source_to_target_map):
        kernels = self.guassian_kernel(
            source, target, source_labels, target_labels, source_to_target_map,
            kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma
        )

        batch_size = source.size(0)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss = torch.mean(XX + YY - XY - YX)
        return loss



class ICLoss(nn.Module):
    def __init__(self, sigma=1.0, delta=1.0):
        super(ICLoss, self).__init__()
        self.sigma = sigma
        self.delta = delta

    def forward(self, s_emb, t_emb, pseudo_labels):
        N = len(s_emb)

        # L2 normalization
        s_emb = F.normalize(s_emb, p=2, dim=1)
        t_emb = F.normalize(t_emb, p=2, dim=1)

        # Student distance matrix
        S_dist = torch.cdist(s_emb, s_emb)

        # Row-wise mean normalization
        S_dist = S_dist / (S_dist.mean(1, keepdim=True).clamp_min(1e-12))

        with torch.no_grad():

            # Teacher distance matrix
            T_dist = torch.cdist(t_emb, t_emb)

            # Original similarity matrix
            W = torch.exp(-T_dist / self.sigma)

            # Pseudo labels
            pseudo_labels = pseudo_labels.view(-1).to(
                s_emb.device
            )

            # Exclude diagonal elements
            identity_matrix = torch.eye(N, device=s_emb.device, dtype=torch.bool)

            # Positive mask: same pseudo-label
            pos_mask = (pseudo_labels.unsqueeze(1)== pseudo_labels.unsqueeze(0)) & (~identity_matrix)

            # Negative mask: different pseudo-label
            neg_mask = (pseudo_labels.unsqueeze(1)!= pseudo_labels.unsqueeze(0)) & (~identity_matrix)

            # Exclude noise samples (-1), if present
            valid = pseudo_labels != -1
            valid_pair = valid[:, None] & valid[None, :]

            pos_mask = pos_mask & valid_pair
            neg_mask = neg_mask & valid_pair

            # Positive sample weights
            pos_weight = W * pos_mask.float()
            print(pos_weight)
            # Negative sample weights
            neg_weight = (torch.relu(0.5 - W)* neg_mask.float())
            print("-",neg_weight)
        # Pull loss: same pseudo-identity
        pull_losses = (torch.relu(S_dist) * pos_weight)

        # Push loss: different pseudo-identities
        push_losses = (torch.relu(self.delta - S_dist)* neg_weight)

        # Valid ordered sample pairs
        num_pairs = (pos_mask.sum() + neg_mask.sum()).clamp_min(1)

        loss = (pull_losses.sum() + push_losses.sum()) / num_pairs

        return loss