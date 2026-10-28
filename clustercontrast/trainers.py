from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
from .models.losses import ICLoss,Dissimilar
import torch

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
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
class ClusterContrastTrainer_VIT_s1(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_VIT_s1, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.Dissimilar_LOSS = Dissimilar(dynamic_balancer=True)

    def train(self, epoch, data_loader_ir, data_loader_rgb, optimizer, print_freq=10, train_iters=400):

        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            #indexes_rgb = torch.cat((indexes_rgb, indexes_rgb), -1)
            _, f_out_rgb, f_out_ir, SDC_rgb, SDC_ir = self._forward(inputs_rgb, inputs_ir, modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            # 原始特性的多样性loss
            Dissimilar_LOSS_ir = self.Dissimilar_LOSS(torch.stack(SDC_ir, dim=1))
            Dissimilar_LOSS_rgb = self.Dissimilar_LOSS(torch.stack(SDC_rgb, dim=1))
            Dissimilar_LOSS1 = Dissimilar_LOSS_ir + Dissimilar_LOSS_rgb

            loss = loss_ir + loss_rgb +Dissimilar_LOSS1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t' 
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss Dissimilar {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir,loss_rgb,Dissimilar_LOSS1))

    def _parse_data_rgb(self, inputs):
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, modal=0):
        return self.encoder(x1, x2, modal)

class ClusterContrastTrainer_VIT_s2(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_VIT_s2, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.ICLoss = ICLoss(1,1)
        self.Dissimilar_LOSS = Dissimilar(dynamic_balancer=True)

    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400, i2r=None, r2i=None):

        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            #indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            _, f_out_rgb, f_out_ir, SDC_rgb, SDC_ir = self._forward(inputs_rgb, inputs_ir, modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            # 原始特性的多样性loss
            Dissimilar_LOSS_ir = self.Dissimilar_LOSS(torch.stack(SDC_ir, dim=1))
            Dissimilar_LOSS_rgb = self.Dissimilar_LOSS(torch.stack(SDC_rgb, dim=1))
            Dissimilar_LOSS1 = Dissimilar_LOSS_ir + Dissimilar_LOSS_rgb

            # # # # # NPC Loss
            f_out_rgb_de = f_out_rgb.detach()  ## 目标特征（RGB梯度分离）
            f_out_ir_de = f_out_ir.detach()
            loss_icl_rgb = self.ICLoss(f_out_rgb, f_out_rgb_de,labels_rgb)  ## 约束RGB特征内部关系一致性
            loss_icl_ir = self.ICLoss(f_out_ir, f_out_ir_de,labels_ir)
            loss_ICL = loss_icl_ir+loss_icl_rgb

            # cross contrastive learning
            if r2i:
                rgb2ir_labels = torch.tensor([r2i[key.item()] for key in labels_rgb]).cuda()
                ir2rgb_labels = torch.tensor([i2r[key.item()] for key in labels_ir]).cuda()
                alternate = False
                if alternate:
                    # accl
                    if epoch % 2 == 1:
                        cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long())

                    else:
                        cross_loss = self.memory_ir(f_out_rgb, rgb2ir_labels.long())

                else:
                    cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long()) + self.memory_ir(f_out_rgb, rgb2ir_labels.long())

            else:
                cross_loss = torch.tensor(0.0)

            # total loss
            loss = loss_ir + loss_rgb + 0.5 *cross_loss + Dissimilar_LOSS1+loss_ICL

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t' 
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'loss_cross {:.3f}\t'
                      'Loss Diss {:.3f}\t'
                      'Loss ICL {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir,loss_rgb,
                              cross_loss,Dissimilar_LOSS1,loss_ICL))

    def _parse_data_rgb(self, inputs):
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, modal=0):
        return self.encoder(x1, x2, modal)


