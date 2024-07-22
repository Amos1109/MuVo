import numpy as np
import torch
import torch.nn.functional as F
import ot
from src.simclr import SupervisedConLoss
#1.5 81.0 #1 80.9
group_simclr = SupervisedConLoss(temperature=1.5, base_temperature=1.5)

def get_group(feat, label):
    groups = {}
    for x, y in zip(label, feat):
        group = groups.get(x.item(), [])
        group.append(y)
        groups[x.item()] = group
    return groups

def contrastive_source_target(features, class_labels, class_centroid):
    features = F.normalize(features)
    class_centroid = F.normalize(class_centroid)
    grp_unlabeled = get_group(features, class_labels)
    l_fast = []
    l_slow = []
    for key in grp_unlabeled.keys():
        l_fast.append(torch.stack(grp_unlabeled[key]).mean(dim=0))
        l_slow.append(class_centroid[key])
    if len(l_fast) > 0:
        l_fast = torch.stack(l_fast)
        l_slow = torch.stack(l_slow)
        features = torch.cat([l_fast.unsqueeze(1), l_slow.unsqueeze(1).cuda()], dim=1)
        loss = group_simclr(features)
    return loss


def contrastive_class_to_class_learned_memory( features, class_labels, memory):
    """

    Args:
        model: segmentation model that contains the self-attention MLPs for selecting the features
        to take part in the contrastive learning optimization
        features: Nx256  feature vectors for the contrastive learning (after applying the projection and prediction head)
        class_labels: N corresponding class labels for every feature vector
        num_classes: number of classes in the dataet
        memory: memory bank [List]

    Returns:
        returns the contrastive loss between features vectors from [features] and from [memory] in a class-wise fashion.
    """




    loss = 0
    features = F.normalize(features)
    for c in range(memory.num_classes):
    #     # get features of an specific class
        mask_c = class_labels == c
        features_c = features[mask_c, :]
        memory_c = []
        memory_c.extend(memory.feature_queues[c])

        class_centroid = memory.proto_s.mo_pro
        if features_c.shape[0] > 1:
            target_mask_c = torch.zeros((memory.num_classes), dtype=float)
            target_mask_c[c] = 1
            target_mask_c = target_mask_c.repeat(features_c.shape[0], 1)
            target_mask_c = target_mask_c.cuda()
            logits = torch.einsum('nc,ck->nk', [F.normalize(features_c, dim=1), class_centroid.transpose(0, 1)])
            logits /= 0.3
            log_prob = F.normalize(logits.exp(), dim=1, p=1).log()
            loss_target = - torch.sum((target_mask_c * log_prob).sum(1)) / target_mask_c.shape[0]
            loss = loss + loss_target


        if memory_c is not None and features_c.shape[0] > 1 and len(memory.feature_queues[c]) > 1:

            # L2 normalize vectors
            # memory_c = F.normalize(memory_c, dim=1) # N, 256
            # features_c_norm = F.normalize(features_c, dim=1) # M, 256
            memory_c = torch.stack(memory_c).cuda()

            # compute similarity. All elements with all elements
            similarities = torch.mm(features_c, memory_c.transpose(1, 0))  # MxN
            # MxN
            distances = 1 - similarities # values between [0, 2] where 0 means same vectors
            # M (elements), N (memory)

            loss = loss + distances.mean()

    return loss / memory.num_classes




def infoNce(query, key, temp=0.3):
    target_mask_c = torch.diag(torch.ones(query.shape[0])).cuda()
    # target_mask_c = torch.cat((target_mask_c, target_mask_c), dim=1).cuda()
    # key = torch.cat((key, query), dim=0)
    logits = torch.einsum('nc,ck->nk', [query, key.transpose(0, 1)])
    logits /= temp
    log_prob = F.normalize(logits.exp(), dim=1, p=1).log()
    loss = - torch.sum(target_mask_c * log_prob) / target_mask_c.sum()
    return loss


def contrastive_target_source(features, class_labels, class_centroid):
    loss = 0
    features = F.normalize(features)
    grp_unlabeled = get_group(features, class_labels)
    l_fast = []
    l_slow = []
    for key in grp_unlabeled.keys():
        l_fast.append(torch.stack(grp_unlabeled[key]).mean(dim=0))
        l_slow.append(class_centroid[key])
    if len(l_fast) > 0:
        l_fast = torch.stack(l_fast)
        l_slow = torch.stack(l_slow)
        loss = infoNce(l_fast, l_slow)
        # loss = F.mse_loss(l_fast, l_slow)
        # features = torch.cat([l_fast.unsqueeze(1), l_slow.unsqueeze(1).cuda()], dim=1)
        # loss = group_simclr(features)

    return loss


def ot_loss(proto_s, feat_tu_w, feat_tu_s, num_classes):
    bs = feat_tu_s.shape[0]
    with torch.no_grad():
        M_st_weak = 1 - pairwise_cosine_sim(proto_s.mo_pro, feat_tu_w)  # postive distance
    gamma_st_weak = ot_mapping(M_st_weak.data.cpu().numpy().astype(np.float64))
    score_ot, pred_ot = gamma_st_weak.t().max(dim=1)
    Lm = center_loss_cls(proto_s.mo_pro, feat_tu_s, pred_ot, num_classes=num_classes)
    return Lm


def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat


def ot_mapping(M):
    '''
    M: (ns, nt)
    '''
    reg1 = 1
    reg2 = 1
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    gamma = ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M, reg1, reg2)
    gamma = torch.from_numpy(gamma).cuda()
    return gamma


def center_loss_cls(centers, x, labels, num_classes=65):
    classes = torch.arange(num_classes).long().cuda()
    batch_size = x.size(0)
    centers_norm = F.normalize(centers)
    x = F.normalize(x)
    distmat = -x @ centers_norm.t() + 1

    labels = labels.unsqueeze(1).expand(batch_size, num_classes)
    mask = labels.eq(classes.expand(batch_size, num_classes))

    dist = distmat * mask.float()
    loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
    return loss


