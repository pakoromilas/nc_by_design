"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCL(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SCL, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":  # contrast one view, for supcon?
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":  # contrast all views, for simclr?
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class NormFace(nn.Module):
    def __init__(
        self, temperature=1.0, weight=None, ignore_index=-100, reduction="mean"
    ):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, features, weight, target):
        """
        input: Tensor of shape (N, C) — raw logits
        target: Tensor of shape (N,) — class indices
        """
        logits = features @ weight  # [B, C]
        logits = logits / self.temperature  # Apply temperature scaling
        return F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class NTCE(nn.Module):
    """
    Implements the NTCE Combined loss function:

    L_NTCE_combined(U, W) = (1/M) * Σ_{i=1}^M [-log(exp(w_{y_i}^T u_i / (||w_{y_i}|| ||u_i|| τ)) / Σ_{j=1}^M exp(w_{y_i}^T u_j / (||w_{y_i}|| ||u_j|| τ)))]

    Where:
    - U ∈ R^{M×D}: batch of M instance feature representations
    - W ∈ R^{K×D}: K class weight vectors
    - w_{y_i}: class weight vector for sample i's true class y_i
    - u_i, u_j: instance representations (normalized to unit vectors)
    - τ: temperature parameter
    - M: batch size, K: number of classes

    Key properties:
    1. Anchors on class weight vectors rather than instances
    2. Contrasts each class weight against ALL M instances in batch
    3. Uses cosine similarity through L2 normalization
    4. Provides M negative samples per positive pair (vs K in standard NTCE)
    """

    def __init__(
        self, temperature=1.0, weight=None, ignore_index=-100, reduction="mean"
    ):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, features, weight, target):
        """
        Args:
            features: Tensor of shape (N, D) — instance representations
            weight: Tensor of shape (C, D) or (D, C) — class weight vectors
            target: Tensor of shape (N,) — class indices
        """
        # Handle both weight orientations: (C, D) or (D, C)
        if weight.shape[0] == features.shape[1]:  # weight is (D, C)
            weight = weight.T  # transpose to (C, D)

        weight_norm = weight
        features_norm = features
        # Get the class weight vectors for each sample
        class_weights = weight_norm[target]  # [N, D]

        # Compute similarities between each class weight and all features in batch
        all_logits = class_weights @ features_norm.T  # [N, N] - all pairs

        # Apply temperature scaling
        all_logits = all_logits / self.temperature

        # Create targets for cross entropy (diagonal elements are positive pairs)
        batch_size = features.size(0)
        ce_target = torch.arange(batch_size, device=features.device)

        # Handle ignore_index
        if self.ignore_index != -100:
            mask = target != self.ignore_index
            if mask.sum() == 0:
                return torch.tensor(0.0, device=features.device, requires_grad=True)
            all_logits = all_logits[mask]
            ce_target = torch.arange(mask.sum(), device=features.device)

        # Compute cross entropy loss
        loss = F.cross_entropy(all_logits, ce_target, reduction="none")

        # Apply sample weighting if provided
        if self.weight is not None:
            weight_expanded = self.weight[target]
            if self.ignore_index != -100:
                weight_expanded = weight_expanded[mask]
            loss = loss * weight_expanded

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class NONL(nn.Module):
    def __init__(self, temperature=None, eps=1e-8, symmetric=True, bidirectional=False):
        super().__init__()
        self.eps = eps
        self.tau = temperature
        self.symmetric = symmetric
        self.bidirectional = bidirectional

    def forward(self, features, weight, target):
        # Common computations
        logits = features @ weight  # shape: [B, C]
        if self.tau:
            logits = logits / self.tau
        exp_logits = torch.exp(logits)

        # Mask to zero out the target class
        mask = torch.zeros_like(exp_logits, dtype=torch.bool)
        mask[torch.arange(logits.size(0)), target] = True
        exp_logits_masked = exp_logits.masked_fill(mask, 0)

        # Common numerator
        numerator = exp_logits.gather(1, target.view(-1, 1))  # shape: [B, 1]

        denom = torch.sum(exp_logits_masked, dim=1, keepdim=True)  # shape: [B, 1]
        # Class-wise denominator over the batch
        exp_logits_masked_T = exp_logits_masked.T  # shape: [C, B]
        denom_per_class = torch.sum(
            exp_logits_masked_T, dim=1, keepdim=True
        )  # shape: [C, 1]
        denom_sym = denom_per_class[target]  # shape: [B, 1]

        if self.bidirectional:
            denom = denom + denom_sym
        elif self.symmetric:
            denom = denom_sym

        # Final loss
        probs = numerator / denom
        log_probs = probs.log()
        loss = -log_probs.mean()
        return loss
