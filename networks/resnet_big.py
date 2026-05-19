"""ResNet in PyTorch.
Uses torchvision models directly for better compatibility.
Based on torchvision.models.resnet with cifar_head option.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def generate_simplex_etf(
    num_classes: int,
    feat_dim: int,
    target_norm: float = 1.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Build a simplex Equiangular Tight Frame (ETF) with `num_classes` vectors
    in R^feat_dim, following Eq. (1) in the paper.

    Returns:
      W: (num_classes, feat_dim)  — each row is a classifier vector.
    """
    K = num_classes
    d = feat_dim
    assert d >= K - 1, "ETF requires feat_dim >= num_classes - 1"

    device = device or torch.device("cpu")

    # Random orthonormal matrix U ∈ R^{d×K}
    # Use QR decomposition on a random Gaussian matrix.
    G = torch.randn(d, K, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(G)  # (d, K)
    U = Q[:, :K]  # (d, K), orthonormal columns

    # Centering matrix: I - (1/K) * 1 1^T
    I = torch.eye(K, device=device, dtype=dtype)
    one = torch.ones(K, 1, device=device, dtype=dtype)
    H = I - (1.0 / K) * (one @ one.t())  # (K, K)

    # Simplex ETF (d, K)
    M = (K / (K - 1)) ** 0.5 * (U @ H)

    # Transpose so each row is a classifier vector (C, D)
    W = M.t()  # (K, d)

    # Scale to desired norm
    if target_norm is not None:
        norms = W.norm(dim=1, keepdim=True) + 1e-12
        W = W / norms * target_norm

    return W


def control_block_gradients(net_block, requires_grad=False):
    for param in net_block.parameters():
        param.requires_grad = requires_grad
    return net_block


class Flatten(nn.Module):
    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        return torch.flatten(feat, start_dim=self.dim)


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""

    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupCEResNet(nn.Module):
    """encoder + classifier - supports ResNet and ViT backbones from torchvision"""

    def __init__(
        self,
        name="resnet50",
        num_classes=10,
        normalize=False,
        head="linear",
        dataset_type="imagenet",
        loss_type="CE",
    ):
        super(SupCEResNet, self).__init__()
        self.normalize = normalize
        self.is_vit = False

        # ----------------- BACKBONE -----------------
        if name == "resnet18":
            self.backbone = models.resnet18(weights=None)
            feat_dim = 512
        elif name == "resnet34":
            self.backbone = models.resnet34(weights=None)
            feat_dim = 512
        elif name == "resnet50":
            self.backbone = models.resnet50(weights=None)
            feat_dim = 2048
        elif name == "resnet101":
            self.backbone = models.resnet101(weights=None)
            feat_dim = 2048
        elif name == "resnet152":
            self.backbone = models.resnet152(weights=None)
            feat_dim = 2048

        elif name in ("vit_b_16", "vit_base"):
            self.backbone = models.vit_b_16(weights=None)
            self.is_vit = True
            feat_dim = self.backbone.hidden_dim  # 768
        elif name in ("vit_l_16", "vit_large"):
            self.backbone = models.vit_l_16(weights=None)
            self.is_vit = True
            feat_dim = self.backbone.hidden_dim  # 1024
        else:
            raise ValueError(f"Unknown model: {name}")

        # CIFAR tweaks only for ResNets
        if dataset_type == "cifar" and not self.is_vit:
            self.backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.backbone.maxpool = nn.Identity()

        use_bias = loss_type == "CE"

        # ----------------- HEAD -----------------
        if not self.is_vit:
            # ResNet heads
            if head == "linear":
                self.backbone.fc = nn.Linear(feat_dim, num_classes, bias=use_bias)
            elif head == "etf":
                self.backbone.fc = nn.Linear(feat_dim, num_classes, bias=False)
                etf_weights = generate_simplex_etf(
                    num_classes=num_classes, feat_dim=feat_dim
                )
                with torch.no_grad():
                    self.backbone.fc.weight.copy_(etf_weights)
                self.backbone.fc.weight.requires_grad = False
            else:
                raise ValueError(f"Unknown head type: {head}")
        else:
            # ViT heads
            in_dim = self.backbone.heads.head.in_features
            if head == "linear":
                self.backbone.heads.head = nn.Linear(in_dim, num_classes, bias=use_bias)
            elif head == "etf":
                self.backbone.heads.head = nn.Linear(in_dim, num_classes, bias=False)
                etf_weights = generate_simplex_etf(m=feat_dim, n=num_classes)
                etf_weights_torch = torch.from_numpy(etf_weights.T).float()
                with torch.no_grad():
                    self.backbone.heads.head.weight.copy_(etf_weights_torch)
                self.backbone.heads.head.weight.requires_grad = False
            else:
                raise ValueError(f"Unknown head type: {head}")

        print(
            f"** Using backbone={name}, head={head}, bias={use_bias}, loss_type={loss_type} **"
        )

    def forward(self, x):
        if not self.is_vit:
            # ----------------- RESNET FORWARD -----------------
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)

            if not isinstance(self.backbone.maxpool, nn.Identity):
                x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)

            features = torch.flatten(x, 1)
            output = self.backbone.fc(features)

        else:
            # ----------------- VIT FORWARD (matches torchvision) -----------------
            # 1. Patch embedding
            x = self.backbone._process_input(x)  # (N, S, D)
            n = x.shape[0]

            # 2. Add class token
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat((batch_class_token, x), dim=1)  # (N, S+1, D)

            # 3. Add positional embeddings
            x = x + self.backbone.encoder.pos_embedding

            # 4. Transformer encoder
            x = self.backbone.encoder(x)  # (N, S+1, D)

            # 5. Final layernorm (API changed slightly across versions)
            if hasattr(self.backbone, "ln"):
                x = self.backbone.ln(x)
            elif hasattr(self.backbone.encoder, "ln"):
                x = self.backbone.encoder.ln(x)

            # CLS token as feature
            features = x[:, 0]  # (N, D)

            # 6. Classification head
            output = self.backbone.heads(features)

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        return output, features

    @property
    def fc(self):
        if self.is_vit:
            return self.backbone.heads.head
        return self.backbone.fc


# Legacy support for old model creation (not used but kept for compatibility)
class ResNet18(nn.Module):
    def __init__(self, cifar_head=True):
        super().__init__()
        self.model = models.resnet18(weights=None)
        if cifar_head:
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.model.maxpool = nn.Identity()
        print("** Using avgpool **")

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        if not isinstance(self.model.maxpool, nn.Identity):
            x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet50(nn.Module):
    def __init__(self, cifar_head=True, hparams=None):
        super().__init__()
        self.model = models.resnet50(weights=None)
        if cifar_head:
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.model.maxpool = nn.Identity()
        self.hparams = hparams
        print("** Using avgpool **")

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        if not isinstance(self.model.maxpool, nn.Identity):
            x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet34(nn.Module):
    def __init__(self, cifar_head=True):
        super().__init__()
        self.model = models.resnet34(weights=None)
        if cifar_head:
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.model.maxpool = nn.Identity()
        print("** Using avgpool **")

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        if not isinstance(self.model.maxpool, nn.Identity):
            x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet101(nn.Module):
    def __init__(self, cifar_head=True, hparams=None):
        super().__init__()
        self.model = models.resnet101(weights=None)
        if cifar_head:
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.model.maxpool = nn.Identity()
        self.hparams = hparams
        print("** Using avgpool **")

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        if not isinstance(self.model.maxpool, nn.Identity):
            x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# Model dictionary for legacy support (not used by SupCEResNet)
model_dict = {
    "resnet18": [ResNet18, 512],
    "resnet34": [ResNet34, 512],
    "resnet50": [ResNet50, 2048],
    "resnet101": [ResNet101, 2048],
}


class SupConResNet(nn.Module):
    """backbone + projection head"""

    def __init__(
        self, name="resnet50", head="mlp", feat_dim=128, dataset_type="imagenet"
    ):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]

        # Set cifar_head based on dataset
        cifar_head = dataset_type == "cifar"
        self.encoder = model_fun(cifar_head=cifar_head)

        # Store head type for forward pass logic
        self.head_type = head
        self.dim_in = dim_in

        if head == "linear":
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
            )
        elif head == "none":
            # No projection head - will output encoder features directly
            self.head = None
        else:
            raise NotImplementedError("head not supported: {}".format(head))

    def forward(self, x):
        feat = self.encoder(x)

        if self.head_type == "none":
            # Directly normalize encoder features without projection
            feat = F.normalize(feat, dim=1)
        else:
            # Apply projection head then normalize
            feat = F.normalize(self.head(feat), dim=1)

        return feat


class SupConMultiHeadResNet(nn.Module):
    """backbone + projection head"""

    def __init__(
        self,
        num_classes,
        name="resnet50",
        head="mlp",
        feat_dim=128,
        dataset_type="imagenet",
    ):
        super(SupConMultiHeadResNet, self).__init__()
        self.num_classes = num_classes
        model_fun, dim_in = model_dict[name]

        # Set cifar_head based on dataset
        cifar_head = dataset_type == "cifar"
        self.encoder = model_fun(cifar_head=cifar_head)

        for c in range(num_classes):
            if head == "linear":
                setattr(self, "head_{}".format(c), nn.Linear(dim_in, feat_dim))
            elif head == "mlp":
                setattr(
                    self,
                    "head_{}".format(c),
                    nn.Sequential(
                        nn.Linear(dim_in, dim_in),
                        nn.ReLU(inplace=True),
                        nn.Linear(dim_in, feat_dim),
                    ),
                )
            else:
                raise NotImplementedError("head not supported: {}".format(head))

    def forward(self, x, projection_head):
        """projection_head: indice of the corresponding projection head"""
        feat = self.encoder(x)
        head = getattr(self, "head_{}".format(projection_head))
        feat = F.normalize(head(feat), dim=1)
        return feat


class SupConMultiHeadResNet2(nn.Module):
    """backbone + projection head"""

    def __init__(
        self,
        num_classes,
        name="resnet50",
        head="mlp",
        feat_dim=128,
        dataset_type="imagenet",
    ):
        super(SupConMultiHeadResNet2, self).__init__()
        self.num_classes = num_classes
        model_fun, dim_in = model_dict[name]

        # Set cifar_head based on dataset
        cifar_head = dataset_type == "cifar"
        self.encoder = model_fun(cifar_head=cifar_head)

        for c in range(num_classes):
            if head == "linear":
                setattr(self, "head_{}".format(c), nn.Linear(dim_in, feat_dim))
            elif head == "mlp":
                setattr(
                    self,
                    "head_{}".format(c),
                    nn.Sequential(
                        nn.Linear(dim_in, dim_in),
                        nn.ReLU(inplace=True),
                        nn.Linear(dim_in, feat_dim),
                    ),
                )
            else:
                raise NotImplementedError("head not supported: {}".format(head))

    def forward(self, x):
        """projection_head: indice of the corresponding projection head"""
        embedding = self.encoder(x)
        features = []
        for c in range(self.num_classes):
            head = getattr(self, "head_{}".format(c))
            feat = F.normalize(head(embedding), dim=1)
            features.append(feat)

        return features


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, name="resnet50", num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class PrototypeClassifier:
    """
    Efficient prototype-based classifier that maintains running averages
    of class features to compute class prototypes (mean directions).
    """

    def __init__(self, num_classes: int, feature_dim: int, device="cuda"):
        """
        Args:
            num_classes: Number of classes
            feature_dim: Dimension of feature vectors
            device: Device to store prototypes
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device

        # Running sums of normalized features for each class
        self.feature_sums = torch.zeros(num_classes, feature_dim, device=device)
        # Count of samples seen for each class
        self.class_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        # Final normalized prototypes
        self.prototypes = torch.zeros(num_classes, feature_dim, device=device)

    def update(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Update prototypes with new batch of features.

        Args:
            features: (N, D) tensor of L2-normalized features
            labels: (N,) tensor of class labels
        """
        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=1)

        # Update running sums and counts for each class
        for class_idx in range(self.num_classes):
            mask = labels == class_idx
            if mask.sum() > 0:
                class_features = features[mask]
                # Add to running sum
                self.feature_sums[class_idx] += class_features.sum(dim=0)
                # Update count
                self.class_counts[class_idx] += mask.sum()

        # Recompute prototypes (normalized mean directions)
        self._compute_prototypes()

    def _compute_prototypes(self):
        """
        Compute normalized prototypes from running sums.
        The prototype for each class is the L2-normalized mean of all features.
        """
        for class_idx in range(self.num_classes):
            if self.class_counts[class_idx] > 0:
                # Mean direction (not yet normalized)
                mean_direction = (
                    self.feature_sums[class_idx] / self.class_counts[class_idx]
                )
                # Normalize to unit length to get the mean direction
                self.prototypes[class_idx] = F.normalize(mean_direction, p=2, dim=0)
            else:
                # No samples seen for this class yet
                self.prototypes[class_idx] = torch.zeros(
                    self.feature_dim, device=self.device
                )

    def predict(self, features: torch.Tensor):
        """
        Predict classes using nearest prototype (maximum cosine similarity).

        Args:
            features: (N, D) tensor of L2-normalized features

        Returns:
            predictions: (N,) tensor of predicted class indices
            similarities: (N, C) tensor of similarities to each prototype
        """
        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=1)

        # Compute cosine similarities (dot product since both are normalized)
        similarities = torch.matmul(features, self.prototypes.T)

        # Predict class with highest similarity
        predictions = similarities.argmax(dim=1)

        return predictions, similarities

    def reset(self):
        """Reset all running statistics."""
        self.feature_sums.zero_()
        self.class_counts.zero_()
        self.prototypes.zero_()

    def get_prototypes(self):
        """Get current prototypes."""
        return self.prototypes.clone()


class EfficientPrototypeClassifier(PrototypeClassifier):
    """
    Memory-efficient version that uses Welford's online algorithm
    for computing running mean and variance.
    """

    def __init__(
        self, num_classes: int, feature_dim: int, device="cuda", momentum: float = 0.99
    ):
        """
        Args:
            momentum: Exponential moving average momentum (0.99 = slow update, 0.9 = faster update)
        """
        super().__init__(num_classes, feature_dim, device)
        self.momentum = momentum
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)

    def update(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Update prototypes using exponential moving average for memory efficiency.
        """
        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=1)

        for class_idx in range(self.num_classes):
            mask = labels == class_idx
            if mask.sum() > 0:
                class_features = features[mask]
                # Compute mean of current batch for this class
                batch_mean = class_features.mean(dim=0)

                if self.initialized[class_idx]:
                    # Update with exponential moving average
                    self.feature_sums[class_idx] = (
                        self.momentum * self.feature_sums[class_idx]
                        + (1 - self.momentum) * batch_mean
                    )
                else:
                    # First update for this class
                    self.feature_sums[class_idx] = batch_mean
                    self.initialized[class_idx] = True

                # Update count (for statistics)
                self.class_counts[class_idx] += mask.sum()

        # Recompute prototypes
        self._compute_prototypes_ema()

    def _compute_prototypes_ema(self):
        """Compute prototypes for EMA version."""
        for class_idx in range(self.num_classes):
            if self.initialized[class_idx]:
                # Normalize the accumulated direction
                self.prototypes[class_idx] = F.normalize(
                    self.feature_sums[class_idx], p=2, dim=0
                )
            else:
                self.prototypes[class_idx] = torch.zeros(
                    self.feature_dim, device=self.device
                )


class LinearClassifierNormalized(nn.Module):
    """Linear classifier without bias for normalized training"""

    def __init__(self, name="resnet50", num_classes=10):
        super().__init__()
        _, feat_dim = {
            "resnet18": [None, 512],
            "resnet34": [None, 512],
            "resnet50": [None, 2048],
            "resnet101": [None, 2048],
        }[name]
        self.fc = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, features):
        return self.fc(features)
