import numpy as np
import torch
import torch.nn.functional as F


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

            memory_c = torch.stack(memory_c).cuda()

            # compute similarity. All elements with all elements
            similarities = torch.mm(features_c, memory_c.transpose(1, 0))  # MxN
            # MxN
            distances = 1 - similarities # values between [0, 2] where 0 means same vectors
            # M (elements), N (memory)

            loss = loss + distances.mean()

    return loss / memory.num_classes






def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat



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


