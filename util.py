from __future__ import print_function

import math

import numpy as np
import scipy
import torch
import torch.nn.functional as F


def embedding_ETF_metrics(z, y):
    """
    Check whether ETF has occured in the embeddings.
    It should be:
    er_inter = num_classes - 1
    er_intra = 0
    """
    # ETF metrics https://arxiv.org/pdf/2305.17326
    z_c = compute_class_means(z, y)  # (C, d)
    z_mean = z.mean(dim=0, keepdim=True)  # (1, d)

    # Intra-class effective rank
    z_centered_class = (z - z_c[y]).cpu().numpy()
    classes = torch.unique(y)

    er_intra = []
    y_cpu = y.cpu()  # move y to CPU once

    for c in classes:
        c = c.item()
        idx = y_cpu == c  # CPU mask
        z_class = z_centered_class[idx]  # indexing now stays on CPU
        C = np.cov(z_class.T)
        er_intra.append(effective_rank(C))

    er_intra = np.mean(er_intra)
    # Inter-class effective rank
    z_class_mean = (z_c - z_mean).cpu().numpy()
    C = np.cov(z_class_mean.T)
    er_inter = effective_rank(C)

    return er_intra, er_inter


def NC(b, z, weight, y):
    """
    Check whether Neural Collapse has occured.
    There must be:
    w_erank = num_classes - 1
    w_class_alignment = 0
    """
    w_instance_alignment = alignment(b, z)
    C = np.cov(weight.numpy())
    w_erank = effective_rank(C)

    z_c = compute_class_means(z, y)  # (C, d)
    z_c = F.normalize(z_c, p=2, dim=1)
    w_class_alignment = alignment(weight.T, z_c)

    return w_instance_alignment, w_erank, w_class_alignment


def weight_embeddings_information(z, weight, y):
    """Measure common information among mean class embeddings and weights"""
    # ----- weights x embeddings metrics https://arxiv.org/pdf/2406.03999
    z_c = compute_class_means(z, y)
    z_c = F.normalize(z_c, p=2, dim=1)
    z_mean = z.mean(dim=0, keepdim=True)  # (1, d)
    z_mean = F.normalize(z_mean, p=2, dim=1)
    z_ = z_c - z_mean
    z_ = F.normalize(z_, p=2, dim=1)

    gram_W = compute_gram_matrix(weight).cpu()
    gram_M = compute_gram_matrix(z_.T).cpu()

    mir = matrix_mutual_information_ratio(gram_W, gram_M)
    hdr = matrix_entropy_difference_ratio(gram_W, gram_M)

    return mir, hdr


def compute_class_means(z, y):
    """
    Compute mean embedding per class.

    Args:
        z: (N, d) tensor of embeddings
        y: (N, 1) or (N,) tensor of labels (integers)

    Returns:
        z_c: (C, d) tensor where z_c[c] is the mean embedding of class c
    """
    if y.dim() == 2:
        y = y.squeeze()

    classes = torch.unique(y)
    classes = classes.sort()[0]  # sort the class labels

    z_c = []
    for c in classes:
        mask = y == c
        z_c.append(z[mask].mean(dim=0))

    return torch.stack(z_c)  # shape (C, d)


def compute_gram_matrix(z):
    """Compute normalized gram matrix as per Definition 4.1"""
    z_normalized = F.normalize(z, p=2, dim=0)  # L2 normalize each sample
    return torch.matmul(z_normalized.T, z_normalized)


def matrix_entropy(K):
    """Improved matrix entropy calculation"""
    d = K.shape[0]
    K = (K + K.T) / 2  # Ensure symmetry
    K = K / d  # Direct scaling as per Definition 3.1

    # Use eigendecomposition for numerical stability
    eigvals = torch.linalg.eigvalsh(K)
    eigvals = torch.clamp(eigvals, min=1e-10)  # Avoid log(0)
    entropy = -torch.sum(eigvals * torch.log(eigvals))

    return entropy.item()


def matrix_mutual_information(K1, K2):
    """Compute MI(K1, K2) = H(K1) + H(K2) - H(K1 ◦ K2)"""
    H1 = matrix_entropy(K1)
    H2 = matrix_entropy(K2)
    H12 = matrix_entropy(K1 * K2)  # Hadamard product
    return H1 + H2 - H12


def matrix_mutual_information_ratio(K1, K2):
    """Compute MIR(K1, K2)"""
    H1 = matrix_entropy(K1)
    H2 = matrix_entropy(K2)
    mi = matrix_mutual_information(K1, K2)
    return mi / min(H1, H2)


def matrix_entropy_difference_ratio(K1, K2):
    """Compute HDR(K1, K2)"""
    H1 = matrix_entropy(K1)
    H2 = matrix_entropy(K2)
    return abs(H1 - H2) / max(H1, H2)


def alignment(x, y, alpha=2):
    """
    Alignment calculation between  between anchor embeddings and their positives.
    Implementation from https://github.com/SsnL/align_uniform
    """
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x, t=2):
    """
    Measure uniformity of embeddings based on the potential of the gaussian kernel.
    Implementation from https://github.com/SsnL/align_uniform
    """
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def rank(z, eps=1e-5):
    """
    Calculate the rank of the covariance matrix of embeddings z.
    """
    cov = np.cov(z.T)
    return np.linalg.matrix_rank(cov, tol=eps)


def effective_rank(z, eps=1e-7):
    s = scipy.linalg.svd(z, full_matrices=False, compute_uv=False)
    while np.linalg.norm(s, 1) < eps:
        s = s * 10
    while np.linalg.norm(s, 1) > 1000:
        s = s / 10
    s = s / (np.linalg.norm(s, 1)) + eps
    entropy = scipy.stats.entropy(s)
    return np.exp(entropy)


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = (
            eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
        )
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate**steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (
            args.warm_epochs * total_batches
        )
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def set_optimizer(opt, model):
    """Set optimizer for training.

    - ResNets / CNNs: SGD (as before)
    - ViT models (name contains 'vit'): AdamW with param-wise weight decay
    """
    if "vit" in opt.model.lower():
        print("[INFO] Using AdamW optimizer for ViT model")

        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            lname = name.lower()
            # No weight decay on LayerNorm / norm / bias
            if any(nd in lname for nd in ["bias", "norm", "ln"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # AdamW settings – pretty standard ViT-style
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 0.05},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=opt.learning_rate,  # you already set this in parse_option
            betas=(0.9, 0.999),
        )
        return optimizer

    # ======= original SGD path for ResNet and others =======
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print("==> Saving...")
    state = {
        "opt": opt,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_file)
    del state
