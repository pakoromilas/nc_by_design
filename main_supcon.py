from __future__ import print_function

import argparse
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import util
from losses import NONL, NTCE, SCL, NormFace
from networks.resnet_big import (
    EfficientPrototypeClassifier,
    LinearClassifier,
    LinearClassifierNormalized,
    SupConResNet,
)

# Custom imports
from util import (
    AverageMeter,
    TwoCropTransform,
    accuracy,
    adjust_learning_rate,
    save_model,
    set_optimizer,
    warmup_learning_rate,
)

# Optional imports
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import apex

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    apex = None


def seed_worker(worker_id):
    """Worker function for setting random seeds in DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, deterministic=False):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def optimize_memory_for_imagenet():
    """Apply memory optimizations specifically for ImageNet"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_network_interface():
    """Detect the correct network interface for NCCL on different environments."""
    import re
    import subprocess

    is_slurm = "SLURM_JOB_ID" in os.environ

    try:
        result = subprocess.run(
            ["ip", "route", "get", "8.8.8.8"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            match = re.search(r"dev (\w+)", result.stdout)
            if match:
                interface = match.group(1)
                print(f"[INFO] Detected network interface: {interface}")
                return interface
    except Exception as e:
        print(f"[WARNING] Failed to detect network interface: {e}")

    if is_slurm:
        fallback_interfaces = [
            "ib0",
            "ib1",
            "eth0",
            "ens8",
            "enp0s8",
            "ens3f0",
            "ens3f1",
        ]
    else:
        fallback_interfaces = ["eth0", "ens5", "ens3", "enp0s3"]

    for interface in fallback_interfaces:
        try:
            result = subprocess.run(
                ["ip", "addr", "show", interface],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and "inet " in result.stdout:
                print(f"[INFO] Using fallback network interface: {interface}")
                return interface
        except:
            continue

    print("[WARNING] Could not detect network interface, using eth0 as default")
    return "eth0"


def setup_ddp():
    """Initialize DDP with support for both AWS and HPC/Slurm environments"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("DDP environment variables not found, falling back to manual setup")
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        is_slurm = "SLURM_JOB_ID" in os.environ
        is_aws = any(
            keyword in os.environ.get("HOSTNAME", "").lower()
            for keyword in ["ip-", "ec2", "aws"]
        )

        if rank == 0:
            env_type = "Slurm HPC" if is_slurm else "AWS/Cloud" if is_aws else "Unknown"
            print(f"[INFO] Detected environment: {env_type}")

        # SLURM GPU MAPPING FIX
        if is_slurm:
            if torch.cuda.is_available():
                num_gpus_per_node = torch.cuda.device_count()
                local_rank = local_rank % num_gpus_per_node
                if rank == 0:
                    print(f"[INFO] Slurm detected: {num_gpus_per_node} GPUs per node")
            else:
                print("[ERROR] No CUDA devices available")
                return rank, world_size, 0

        # Base NCCL settings
        os.environ["NCCL_TREE_THRESHOLD"] = "0"

        if is_aws:
            os.environ["NCCL_IB_DISABLE"] = "1"
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_NET"] = "Socket"
        elif is_slurm:
            try:
                import subprocess

                result = subprocess.run(
                    ["ibstat"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and "Active" in result.stdout:
                    if rank == 0:
                        print("[INFO] InfiniBand detected and active")
                else:
                    if rank == 0:
                        print("[INFO] InfiniBand not available, using Ethernet")
                    os.environ["NCCL_IB_DISABLE"] = "1"
            except Exception:
                if rank == 0:
                    print("[INFO] Could not detect InfiniBand, disabling")
                os.environ["NCCL_IB_DISABLE"] = "1"
        else:
            os.environ["NCCL_IB_DISABLE"] = "1"
            os.environ["NCCL_P2P_DISABLE"] = "1"

        network_interface = get_network_interface()
        os.environ["NCCL_SOCKET_IFNAME"] = network_interface
        os.environ["NCCL_DEBUG"] = "WARN"

        if rank == 0:
            print(f"[INFO] Initializing NCCL with interface: {network_interface}")

        try:
            dist.init_process_group(backend="nccl", init_method="env://")
            if rank == 0:
                print(f"[INFO] NCCL process group initialized successfully")
        except Exception as e:
            print(f"[Rank {rank}] NCCL initialization failed: {e}")
            # Fallback with minimal settings
            for key in ["NCCL_SOCKET_IFNAME", "NCCL_P2P_DISABLE", "NCCL_NET"]:
                if key in os.environ:
                    del os.environ[key]
            os.environ["NCCL_TREE_THRESHOLD"] = "0"
            if not is_slurm:
                os.environ["NCCL_IB_DISABLE"] = "1"
            dist.init_process_group(backend="nccl", init_method="env://")
            if rank == 0:
                print(f"[INFO] NCCL initialized with fallback settings")

    return rank, world_size, local_rank


def cleanup():
    """Clean up the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes."""
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_learning_rate_for_batch_size(dataset: str, total_batch_size: int) -> float:
    """Get appropriate learning rate based on dataset and total batch size"""
    if dataset == "imagenet100":
        # ImageNet-100 learning rate scaling (more conservative)
        batch_lr_map = {
            32: 0.0125,
            64: 0.025,
            128: 0.05,
            256: 0.1,
            512: 0.2,
            1024: 0.4,
            2048: 0.8,
        }
    elif dataset == "imagenet1k":
        # ImageNet-1K learning rate scaling (standard linear scaling)
        # Base LR of 0.1 for batch size 256
        batch_lr_map = {
            128: 0.05,
            256: 0.1,
            512: 0.2,
            1024: 0.4,
            2048: 0.8,
            4096: 1.6,
        }
    else:
        # CIFAR learning rate scaling
        batch_lr_map = {
            32: 0.025,
            64: 0.05,
            128: 0.1,
            256: 0.2,
            512: 0.4,
            1024: 0.8,
            2048: 1.6,
        }

    if total_batch_size in batch_lr_map:
        return batch_lr_map[total_batch_size]
    else:
        # Linear scaling for other batch sizes
        if dataset == "imagenet100":
            base_lr = 0.1
            base_batch_size = 256
        elif dataset == "imagenet1k":
            base_lr = 0.1
            base_batch_size = 256
        else:  # CIFAR
            base_lr = 0.2
            base_batch_size = 256

        return base_lr * (total_batch_size / base_batch_size)


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""

    # Data parameters
    dataset: str = "cifar10"
    data_folder: str = "./datasets/"
    batch_size: int = 256
    num_workers: int = 16
    imagenet100_path: Optional[str] = None
    imagenet1k_path: Optional[str] = None

    # Model parameters
    model: str = "resnet18"
    method: str = "SupCon"
    loss: str = "SCL"
    temperature: float = 0.07
    temp_pos_neg_ratio: float = 10.0
    head_type: str = "mlp"
    feat_dim: int = 128

    # Contrastive training parameters
    supcon_epochs: int = 1000
    supcon_learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-4
    supcon_lr_decay_epochs: str = "700,800,900"
    supcon_lr_decay_rate: float = 0.1
    supcon_weight_decay: float = 1e-4
    supcon_momentum: float = 0.9

    # Linear training parameters
    linear_epochs: int = 100
    linear_learning_rate: float = 0.1
    linear_lr_decay_epochs: str = "60,75,90"
    linear_lr_decay_rate: float = 0.2
    linear_weight_decay: float = 0.0
    linear_momentum: float = 0.9
    linear_normalized: bool = False
    linear_loss: str = "CE"
    linear_temperature: float = 0.2

    # Training options
    cosine: bool = False
    syncBN: bool = False
    warm: bool = False
    warm_epochs: int = 10
    warmup_from: float = 0.01
    warmup_to: float = 0.05
    print_freq: int = 10
    save_freq: int = 50
    trial: str = "0"
    seed: int = 42
    deterministic: bool = False

    # Phase control - UPDATED FOR FOUR PHASES
    skip_contrastive: bool = False  # Phase 1
    skip_prototype: bool = False  # Phase 2
    skip_linear: bool = False  # Phase 3
    skip_normalized_linear: bool = False  # Phase 4
    skip_supcon: bool = False  # Deprecated, kept for compatibility
    supcon_ckpt: str = ""

    # Prototype settings - ADD THESE
    use_prototypes: bool = False  # Legacy flag, not used in four-phase
    prototype_update_freq: int = 1

    # Metrics computation control
    compute_metrics_freq: int = 10
    always_compute_final_metrics: bool = True

    # Wandb parameters
    wandb_project: str = "nc_by_design"
    wandb_entity: Optional[str] = None  # Set via --wandb_entity or env var
    wandb_run_name: Optional[str] = None

    # DDP parameters (auto-detected)
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    n_cls: int = 10  # Will be set based on dataset
    wandb_initialized: bool = False


class DatasetManager:
    """Manages dataset loading and transformations with full ImageNet support"""

    DATASET_CONFIGS = {
        "cifar10": {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
            "classes": 10,
            "size": 32,
        },
        "cifar100": {
            "mean": (0.5071, 0.4867, 0.4408),
            "std": (0.2675, 0.2565, 0.2761),
            "classes": 100,
            "size": 32,
        },
        "imagenet100": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "classes": 100,
            "size": 224,
        },
        "imagenet1k": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "classes": 1000,
            "size": 224,
        },
    }

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.dataset_info = self.DATASET_CONFIGS[config.dataset]
        self.mean = self.dataset_info["mean"]
        self.std = self.dataset_info["std"]
        self.n_classes = self.dataset_info["classes"]
        self.input_size = self.dataset_info["size"]

    def get_contrastive_transforms(self):
        """Get transforms for contrastive learning with augmentations"""
        normalize = transforms.Normalize(mean=self.mean, std=self.std)

        if self.config.dataset in ["cifar10", "cifar100"]:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        size=self.input_size, scale=(0.2, 1.0)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:  # ImageNet datasets
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.input_size, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def get_standard_transforms(self, is_train=True):
        """Get standard transforms for linear classifier training"""
        normalize = transforms.Normalize(mean=self.mean, std=self.std)

        if self.config.dataset in ["cifar10", "cifar100"]:
            if is_train:
                return transforms.Compose(
                    [
                        transforms.RandomResizedCrop(size=self.input_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            else:
                return transforms.Compose([transforms.ToTensor(), normalize])
        else:  # ImageNet datasets
            if is_train:
                return transforms.Compose(
                    [
                        transforms.RandomResizedCrop(self.input_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            else:
                return transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(self.input_size),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )

    def create_dataset(self, transform, train=True):
        """Create dataset based on configuration"""
        if self.config.dataset == "cifar10":
            return datasets.CIFAR10(
                root=self.config.data_folder,
                transform=transform,
                download=True,
                train=train,
            )
        elif self.config.dataset == "cifar100":
            return datasets.CIFAR100(
                root=self.config.data_folder,
                transform=transform,
                download=True,
                train=train,
            )
        elif self.config.dataset == "imagenet100":
            data_dir = os.path.join(
                self.config.imagenet100_path, "train" if train else "val.X"
            )
            if not os.path.exists(data_dir):
                raise RuntimeError(f"ImageNet-100 directory not found: {data_dir}")
            return datasets.ImageFolder(data_dir, transform=transform)
        elif self.config.dataset == "imagenet1k":
            data_dir = os.path.join(
                self.config.imagenet1k_path, "train" if train else "val"
            )
            if not os.path.exists(data_dir):
                raise RuntimeError(f"ImageNet-1K directory not found: {data_dir}")
            return datasets.ImageFolder(data_dir, transform=transform)
        else:
            raise ValueError(f"Dataset not supported: {self.config.dataset}")

    def get_contrastive_loader(self):
        """Get data loader for contrastive learning"""
        transform = TwoCropTransform(self.get_contrastive_transforms())
        dataset = self.create_dataset(transform, train=True)

        g = torch.Generator()
        g.manual_seed(self.config.seed + self.config.rank * 10000)

        train_sampler = DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=True,
            seed=self.config.seed,
        )

        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=False,
            generator=g,
            worker_init_fn=seed_worker,
            drop_last=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )

        return train_loader, train_sampler

    def get_linear_loaders(self):
        """Get data loaders for linear classifier training"""
        train_transform = self.get_standard_transforms(is_train=True)
        val_transform = self.get_standard_transforms(is_train=False)

        train_dataset = self.create_dataset(train_transform, train=True)
        val_dataset = self.create_dataset(val_transform, train=False)

        g = torch.Generator()
        g.manual_seed(self.config.seed + self.config.rank * 10000)

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=True,
            seed=self.config.seed,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=False,
        )

        # Determine validation batch size
        val_batch_size = {
            "cifar10": 256,
            "cifar100": 256,
            "imagenet100": 100,
            "imagenet1k": 128,
        }[self.config.dataset]

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=False,
            generator=g,
            worker_init_fn=seed_worker,
            drop_last=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=val_sampler,
            num_workers=min(2, self.config.num_workers),
            pin_memory=False,
            generator=g,
            worker_init_fn=seed_worker,
            drop_last=False,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )

        return train_loader, val_loader, train_sampler


class ModelManager:
    """Manages model creation and loading with DDP support"""

    LOSS_FUNCTIONS = {
        "SCL": SCL,
    }

    def __init__(self, config: TrainingConfig):
        self.config = config

    def create_contrastive_model(self):
        """Create model and criterion for contrastive learning"""
        # Determine dataset type for model architecture
        dataset_type = (
            "cifar" if self.config.dataset in ["cifar10", "cifar100"] else "imagenet"
        )

        # Create model with specified head type
        model = SupConResNet(
            name=self.config.model,
            head=self.config.head_type,  # Use the new head_type parameter
            feat_dim=self.config.feat_dim,  # Use the feat_dim parameter
            dataset_type=dataset_type,
        )
        criterion = self._create_contrastive_criterion()

        # Enable synchronized Batch Normalization
        if self.config.syncBN and self.config.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        return self._setup_ddp_model(model, criterion)

    def create_linear_model(self, pretrained_path=None):
        """Create model and criterion for linear classifier training"""
        dataset_type = (
            "cifar" if self.config.dataset in ["cifar10", "cifar100"] else "imagenet"
        )

        model = SupConResNet(name=self.config.model, dataset_type=dataset_type)

        if self.config.linear_normalized:
            classifier = LinearClassifierNormalized(
                name=self.config.model, num_classes=self.config.n_cls
            )

            # Set up loss exactly as main_ce.py does
            if self.config.linear_loss == "CE":
                criterion = nn.CrossEntropyLoss()
            elif self.config.linear_loss == "NormFace":
                criterion = NormFace(temperature=self.config.linear_temperature)
            elif self.config.linear_loss == "NTCE":
                criterion = NTCE(temperature=self.config.linear_temperature)
            elif self.config.linear_loss == "NONL":
                criterion = NONL(temperature=self.config.linear_temperature)
            else:
                raise ValueError(f"Unknown loss: {self.config.linear_loss}")
        else:
            classifier = LinearClassifier(
                name=self.config.model, num_classes=self.config.n_cls
            )
            criterion = nn.CrossEntropyLoss()

        if pretrained_path:
            self._load_pretrained_weights(model, pretrained_path)

        if self.config.syncBN and self.config.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        return self._setup_ddp_linear_model(model, classifier, criterion)

    def _create_contrastive_criterion(self):
        """Create contrastive loss criterion"""
        loss_class = self.LOSS_FUNCTIONS[self.config.loss]

        return loss_class(temperature=self.config.temperature)

    def _load_pretrained_weights(self, model, checkpoint_path):
        """Load pretrained weights from checkpoint - Fixed for architecture compatibility"""
        try:
            # First try with weights_only=True (secure)
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError) as e:
            if "weights_only" in str(e) or "TrainingConfig" in str(e):
                # If weights_only fails due to custom objects, fall back to weights_only=False
                # but only if we trust the checkpoint source
                if self.config.rank == 0:
                    print(
                        f"[WARNING] Loading checkpoint with weights_only=False due to custom objects."
                    )
                    print(
                        f"[WARNING] Only do this if you trust the checkpoint source: {checkpoint_path}"
                    )
                ckpt = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
            else:
                raise e

        # Extract model state dict
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Get model keys for comparison
        model_keys = set(model.state_dict().keys())
        state_keys = set(state_dict.keys())

        # Debug info
        if self.config.rank == 0:
            print(f"[DEBUG] Model expects keys like: {list(model_keys)[:5]}...")
            print(f"[DEBUG] State dict has keys like: {list(state_keys)[:5]}...")

        # CRITICAL FIX: Handle module. prefix mismatch
        # Check if model expects module. prefix but checkpoint doesn't have it (or vice versa)
        model_has_module = any(k.startswith("module.") for k in model_keys)
        state_has_module = any(k.startswith("module.") for k in state_keys)

        if model_has_module and not state_has_module:
            # Model expects module. but checkpoint doesn't have it - ADD module. prefix
            if self.config.rank == 0:
                print("[DEBUG] Adding 'module.' prefix to checkpoint keys")
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[f"module.{k}"] = v
            state_dict = new_state_dict
        elif not model_has_module and state_has_module:
            # Model doesn't expect module. but checkpoint has it - REMOVE module. prefix
            if self.config.rank == 0:
                print("[DEBUG] Removing 'module.' prefix from checkpoint keys")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v  # Remove 'module.'
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

        # Update state_keys after prefix fix
        state_keys = set(state_dict.keys())

        # Check if the checkpoint has projection head but current model doesn't (or vice versa)
        has_head_in_checkpoint = any("head" in k for k in state_keys)
        has_head_in_model = any("head" in k for k in model_keys)

        if has_head_in_checkpoint and not has_head_in_model:
            if self.config.rank == 0:
                print(
                    "[INFO] Checkpoint has projection head but model doesn't - skipping head weights"
                )
            # Remove head weights from checkpoint
            state_dict = {k: v for k, v in state_dict.items() if "head" not in k}
            # Update state_keys after removing head weights
            state_keys = set(state_dict.keys())
        elif not has_head_in_checkpoint and has_head_in_model:
            if self.config.rank == 0:
                print(
                    "[WARNING] Model expects projection head but checkpoint doesn't have it - head will be randomly initialized"
                )

        # ARCHITECTURE COMPATIBILITY FIX
        # Handle encoder.model.* vs encoder.* structure differences

        # Case 1: State dict has encoder.model.* but model expects encoder.*
        if any(k.startswith("encoder.model.") for k in state_keys) and any(
            k.startswith("encoder.conv1") for k in model_keys
        ):
            if self.config.rank == 0:
                print("[DEBUG] Converting encoder.model.* to encoder.* structure")
            fixed_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("encoder.model."):
                    # encoder.model.conv1.weight -> encoder.conv1.weight
                    new_key = k.replace("encoder.model.", "encoder.")
                    fixed_state_dict[new_key] = v
                elif k.startswith("encoder."):
                    # Keep other encoder keys as-is
                    fixed_state_dict[k] = v
                else:
                    # Keep non-encoder keys (like head weights)
                    fixed_state_dict[k] = v
            state_dict = fixed_state_dict

        # Case 2: State dict has encoder.* but model expects encoder.model.*
        elif any(k.startswith("encoder.conv1") for k in state_keys) and any(
            k.startswith("encoder.model.") for k in model_keys
        ):
            if self.config.rank == 0:
                print("[DEBUG] Converting encoder.* to encoder.model.* structure")
            fixed_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("encoder.") and not k.startswith("encoder.model."):
                    # encoder.conv1.weight -> encoder.model.conv1.weight
                    new_key = k.replace("encoder.", "encoder.model.")
                    fixed_state_dict[new_key] = v
                else:
                    # Keep other keys as-is
                    fixed_state_dict[k] = v
            state_dict = fixed_state_dict

        # FINAL COMPATIBILITY CHECK
        # Filter out keys that don't exist in the target model (like classifier heads)
        model_keys = set(model.state_dict().keys())
        filtered_state_dict = {}

        for k, v in state_dict.items():
            if k in model_keys:
                # Check tensor shape compatibility
                if model.state_dict()[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    if self.config.rank == 0:
                        print(
                            f"[WARNING] Shape mismatch for {k}: model={model.state_dict()[k].shape}, checkpoint={v.shape}"
                        )
            else:
                # Skip keys not in target model (like old classifier heads)
                if "head" not in k and "fc" not in k and "classifier" not in k:
                    if self.config.rank == 0:
                        print(f"[WARNING] Key {k} not found in target model")

        # Load the filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(
            filtered_state_dict, strict=False
        )

        if self.config.rank == 0:
            if missing_keys:
                print(
                    f"[INFO] Missing keys (will be randomly initialized): {missing_keys[:10]}..."
                )
            if unexpected_keys:
                print(f"[INFO] Unexpected keys (ignored): {unexpected_keys[:10]}...")
            print(
                f"[INFO] Successfully loaded {len(filtered_state_dict)} weights from checkpoint"
            )

    def _setup_ddp_model(self, model, criterion):
        """Setup model with DDP if needed"""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            model = model.cuda(self.config.local_rank)
            criterion = criterion.cuda(self.config.local_rank)

            if self.config.world_size > 1:
                model = DDP(
                    model,
                    device_ids=[self.config.local_rank],
                    output_device=self.config.local_rank,
                    find_unused_parameters=True,
                    broadcast_buffers=False,
                )

            cudnn.benchmark = True

        return model, criterion

    def _setup_ddp_linear_model(self, model, classifier, criterion):
        """Setup linear model with DDP if needed"""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            model = model.cuda(self.config.local_rank)
            classifier = classifier.cuda(self.config.local_rank)
            criterion = criterion.cuda(self.config.local_rank)

            if self.config.world_size > 1:
                model = DDP(
                    model,
                    device_ids=[self.config.local_rank],
                    output_device=self.config.local_rank,
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                )
                classifier = DDP(
                    classifier,
                    device_ids=[self.config.local_rank],
                    output_device=self.config.local_rank,
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                )

            cudnn.benchmark = True

        return model, classifier, criterion


class MetricsComputer:
    """Handles computation of neural collapse metrics with improved reliability"""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def should_compute_metrics(self, epoch, total_epochs):
        """Determine if metrics should be computed for this epoch"""
        if self.config.dataset not in ["cifar10", "cifar100"]:
            return False

        # Always compute on final epoch if enabled
        if epoch == total_epochs and self.config.always_compute_final_metrics:
            return True

        # Compute every N epochs
        return epoch % self.config.compute_metrics_freq == 0

    def compute_contrastive_metrics(self, model, loader):
        """Compute metrics for contrastive training phase"""
        # Only compute embedding-based metrics for contrastive phase
        if self.config.dataset not in ["cifar10", "cifar100"]:
            return {"er_intra": 0.0, "er_inter": 0.0}

        model.eval()
        all_features = []
        all_labels = []

        if self.config.rank == 0:
            print(f"Computing contrastive embedding metrics...")

        with torch.no_grad():
            for idx, (images, labels) in enumerate(loader):
                # Handle TwoCropTransform - take first crop
                if isinstance(images, list):
                    images = images[0]

                if torch.cuda.is_available():
                    images = images.cuda(self.config.local_rank, non_blocking=True)
                    labels = labels.cuda(self.config.local_rank, non_blocking=True)

                features = model(images)
                all_features.append(features.detach().cpu())
                all_labels.append(labels.detach().cpu())

                # Use more batches for better metric estimation
                if idx >= 20:  # Increased from 10 to 20 batches
                    break

        if len(all_features) > 0:
            z = torch.cat(all_features, dim=0)
            y = torch.cat(all_labels, dim=0)

            if self.config.rank == 0:
                print(f"Features shape: {z.shape}, Labels shape: {y.shape}")

            try:
                er_intra, er_inter = util.embedding_ETF_metrics(z, y)
                if self.config.rank == 0:
                    print(
                        f"Successfully computed contrastive metrics: er_intra={er_intra:.4f}, er_inter={er_inter:.4f}"
                    )
                return {"er_intra": er_intra, "er_inter": er_inter}
            except Exception as e:
                if self.config.rank == 0:
                    print(f"Warning: Embedding metrics computation failed: {e}")
                    import traceback

                    traceback.print_exc()
                return {"er_intra": 0.0, "er_inter": 0.0}

        if self.config.rank == 0:
            print("No features collected for metrics computation")
        return {"er_intra": 0.0, "er_inter": 0.0}

    def compute_linear_metrics(self, model, classifier, loader):
        """Compute metrics for linear classifier training phase - FIXED for normalization consistency"""
        if self.config.dataset not in ["cifar10", "cifar100"]:
            return {
                "mir": 0.0,
                "hdr": 0.0,
                "w_inst_alignment": 0.0,
                "w_erank": 0.0,
                "w_class_alignment": 0.0,
            }

        model.eval()
        classifier.eval()

        all_features = []
        all_weights = []
        all_labels = []

        if self.config.rank == 0:
            print(
                f"Computing linear classifier metrics (MIR, HDR, NC) with normalized={self.config.linear_normalized}..."
            )

        with torch.no_grad():
            # Get classifier weights
            if hasattr(classifier, "module"):
                fc_weight = classifier.module.fc.weight  # Shape: (C, d)
            else:
                fc_weight = classifier.fc.weight  # Shape: (C, d)

            # CRITICAL FIX: Normalize weights based on training mode
            if self.config.linear_normalized:
                # If we trained with normalized features, normalize weights too
                weight_normalized = F.normalize(fc_weight, p=2, dim=1)  # Shape: (C, d)
            else:
                # If we trained without normalization, use raw weights
                weight_normalized = fc_weight  # Shape: (C, d)

            for idx, (images, labels) in enumerate(loader):
                if torch.cuda.is_available():
                    images = images.cuda(self.config.local_rank, non_blocking=True)
                    labels = labels.cuda(self.config.local_rank, non_blocking=True)

                # Get features from the encoder
                if hasattr(model, "module"):
                    features_raw = model.module.encoder(images)
                else:
                    features_raw = model.encoder(images)

                # CRITICAL FIX: Apply same normalization as used during training
                if self.config.linear_normalized:
                    # If we trained with normalized features, normalize for metrics too
                    features_processed = F.normalize(
                        features_raw, p=2, dim=1
                    )  # Shape: (N, d)
                else:
                    # If we trained without normalization, use raw features for metrics
                    features_processed = features_raw  # Shape: (N, d)

                # Get corresponding weights for each sample
                corresponding_weights = weight_normalized[labels]  # Shape: (N, d)

                # Collect data
                all_features.append(features_processed.detach().cpu())
                all_weights.append(corresponding_weights.detach().cpu())
                all_labels.append(labels.detach().cpu())

                if idx >= 20:  # Use enough batches for stable metrics
                    break

        if len(all_features) > 0:
            z = torch.cat(all_features, dim=0)  # Features (N, d)
            b = torch.cat(all_weights, dim=0)  # Corresponding weights (N, d)
            y = torch.cat(all_labels, dim=0)  # Labels (N,)

            if self.config.rank == 0:
                print(
                    f"Computing NC metrics on {z.shape[0]} samples with {z.shape[1]} features"
                )
                print(
                    f"Feature norms - min: {torch.norm(z, p=2, dim=1).min():.4f}, "
                    f"max: {torch.norm(z, p=2, dim=1).max():.4f}, "
                    f"mean: {torch.norm(z, p=2, dim=1).mean():.4f}"
                )

            # Compute NC metrics with consistent normalization
            try:
                from utils import compute_nc_metrics

                metrics = compute_nc_metrics(z, b, y, self.config.n_cls)
                return metrics
            except Exception as e:
                if self.config.rank == 0:
                    print(f"[WARNING] Failed to compute NC metrics: {e}")
                return {
                    "mir": 0.0,
                    "hdr": 0.0,
                    "w_inst_alignment": 0.0,
                    "w_erank": 0.0,
                    "w_class_alignment": 0.0,
                }

        if self.config.rank == 0:
            print("No features collected for linear metrics computation")
        return {
            "mir": 0.0,
            "hdr": 0.0,
            "w_inst_alignment": 0.0,
            "w_erank": 0.0,
            "w_class_alignment": 0.0,
        }


class Trainer:
    """Handles training logic for both phases with comprehensive logging"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.writer = None
        self.metrics_computer = MetricsComputer(config)

    def setup_logging(self, log_dir):
        """Setup tensorboard and wandb logging"""
        if self.config.rank == 0:
            # Setup TensorBoard
            tb_path = Path(log_dir) / "tensorboard"
            tb_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_path))
            print("✓ TensorBoard initialized")

            # Setup Wandb if available
            if WANDB_AVAILABLE:
                try:
                    wandb_config = vars(self.config)
                    wandb_config["num_gpus"] = (
                        torch.cuda.device_count() if torch.cuda.is_available() else 0
                    )

                    wandb.init(
                        project=self.config.wandb_project,
                        entity=self.config.wandb_entity,
                        name=self.config.wandb_run_name,
                        config=wandb_config,
                    )

                    self.config.wandb_initialized = True
                    print("✓ Wandb initialized")
                except Exception as e:
                    print(f"⚠ Wandb initialization failed: {e}")
                    self.config.wandb_initialized = False
            else:
                self.config.wandb_initialized = False
                print("⚠ Wandb not available, skipping wandb logging")

    def train_contrastive_epoch(self, train_loader, model, criterion, optimizer, epoch):
        """Train one epoch of contrastive learning"""
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for idx, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)

            # Prepare data - handle TwoCropTransform
            images = torch.cat([images[0], images[1]], dim=0)

            if torch.cuda.is_available():
                images = images.cuda(self.config.local_rank, non_blocking=True)
                labels = labels.cuda(self.config.local_rank, non_blocking=True)

            bsz = labels.shape[0]

            # Warm-up learning rate
            if self.config.warm:
                warmup_learning_rate(
                    self.config, epoch, idx, len(train_loader), optimizer
                )

            # Forward pass
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            # Compute loss
            if self.config.method == "SupCon":
                loss = criterion(features, labels)
            elif self.config.method == "SimCLR":
                loss = criterion(features)
            else:
                raise ValueError(
                    f"Contrastive method not supported: {self.config.method}"
                )

            # Update metrics
            reduced_loss = reduce_tensor(loss, self.config.world_size)
            losses.update(reduced_loss.item(), bsz)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print progress
            if (idx + 1) % self.config.print_freq == 0 and self.config.rank == 0:
                print(
                    "Contrastive Train: [{0}][{1}/{2}]\t"
                    "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "loss {loss.val:.3f} ({loss.avg:.3f})".format(
                        epoch,
                        idx + 1,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                    )
                )
                sys.stdout.flush()

                # Log batch metrics
                if self.config.wandb_initialized and WANDB_AVAILABLE:
                    wandb.log(
                        {
                            "contrastive/batch_loss": losses.val,
                            "contrastive/batch_time": batch_time.val,
                            "contrastive/data_time": data_time.val,
                            "contrastive/step": epoch * len(train_loader) + idx,
                        }
                    )

        return losses.avg

    def train_linear_epoch(
        self, train_loader, model, classifier, criterion, optimizer, epoch
    ):
        """Train one epoch of linear classifier"""
        model.eval()
        classifier.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        end = time.time()
        for idx, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)

            if torch.cuda.is_available():
                images = images.cuda(self.config.local_rank, non_blocking=True)
                labels = labels.cuda(self.config.local_rank, non_blocking=True)

            bsz = labels.shape[0]

            if self.config.warm:
                warmup_learning_rate(
                    self.config, epoch, idx, len(train_loader), optimizer
                )

            # Get features from encoder
            with torch.no_grad():
                if hasattr(model, "module"):
                    features = model.module.encoder(images)
                else:
                    features = model.encoder(images)

            # Handle normalized vs standard training
            if self.config.linear_normalized:
                # Normalize features
                features_norm = F.normalize(features.detach(), p=2, dim=1)

                if self.config.linear_loss == "CE":
                    # For CE with normalized features, pass through classifier
                    output = classifier(features_norm)
                    loss = criterion(output, labels)
                    # Compute accuracy with output
                    acc1, _ = accuracy(output, labels, topk=(1, 5))
                else:
                    # For NormFace / NTCE / NONL - pass features and weights directly to loss
                    # Get the weight matrix from classifier
                    if hasattr(classifier, "module"):
                        weight = classifier.module.fc.weight  # Transpose to (D, C)
                    else:
                        weight = classifier.fc.weight  # Transpose to (D, C)

                    weight = torch.nn.functional.normalize(weight, p=2, dim=1).T
                    # The loss functions expect: features (N, D), weight (D, C), labels (N,)
                    loss = criterion(features_norm, weight, labels)

                    # For accuracy, compute logits manually
                    with torch.no_grad():
                        logits = torch.matmul(features_norm, weight)
                        acc1, _ = accuracy(logits, labels, topk=(1, 5))
            else:
                # Standard training
                output = classifier(features.detach())
                loss = criterion(output, labels)
                acc1, _ = accuracy(output, labels, topk=(1, 5))

            # Update metrics
            reduced_loss = reduce_tensor(loss, self.config.world_size)
            losses.update(reduced_loss.item(), bsz)

            reduced_acc1 = reduce_tensor(acc1[0], self.config.world_size)
            top1.update(reduced_acc1.item(), bsz)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % self.config.print_freq == 0 and self.config.rank == 0:
                loss_name = (
                    self.config.linear_loss if self.config.linear_normalized else "CE"
                )
                print(
                    f"Linear Train ({loss_name}): [{epoch}][{idx + 1}/{len(train_loader)}]\t"
                    f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"loss {losses.val:.3f} ({losses.avg:.3f})\t"
                    f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})"
                )

        return losses.avg, top1.avg

    # FIX 1: Update validate_linear method to handle normalization correctly
    def validate_linear(self, val_loader, model, classifier, criterion):
        """Validate linear classifier"""
        model.eval()
        classifier.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        with torch.no_grad():
            end = time.time()
            for idx, (images, labels) in enumerate(val_loader):
                if torch.cuda.is_available():
                    images = images.cuda(self.config.local_rank, non_blocking=True)
                    labels = labels.cuda(self.config.local_rank, non_blocking=True)

                bsz = labels.shape[0]

                # Get features - FIXED to handle DDP properly
                if hasattr(model, "module"):
                    features = model.module.encoder(images)
                else:
                    features = model.encoder(images)

                # CRITICAL FIX: Handle normalized vs standard validation consistently
                if self.config.linear_normalized:
                    # Always normalize features for normalized evaluation
                    features_norm = F.normalize(features, p=2, dim=1)

                    if self.config.linear_loss == "CE":
                        # For CE with normalized features, pass through classifier
                        output = classifier(features_norm)
                        loss = criterion(output, labels)
                        acc1, _ = accuracy(output, labels, topk=(1, 5))
                    else:
                        # For NormFace / NTCE / NONL - use proper normalized computation
                        if hasattr(classifier, "module"):
                            weight = classifier.module.fc.weight  # (C, D)
                        else:
                            weight = classifier.fc.weight  # (C, D)

                        # CRITICAL FIX: Normalize classifier weights too
                        weight_norm = F.normalize(weight, p=2, dim=1)  # (C, D)

                        # For normalized losses, pass normalized features and weights
                        loss = criterion(
                            features_norm, weight_norm.T, labels
                        )  # weight.T = (D, C)

                        # Compute accuracy using normalized dot product
                        logits = torch.matmul(features_norm, weight_norm.T)  # (N, C)
                        acc1, _ = accuracy(logits, labels, topk=(1, 5))
                else:
                    # Standard validation - no normalization
                    output = classifier(features)
                    loss = criterion(output, labels)
                    acc1, _ = accuracy(output, labels, topk=(1, 5))

                # Update metrics
                reduced_loss = reduce_tensor(loss, self.config.world_size)
                losses.update(reduced_loss.item(), bsz)

                reduced_acc1 = reduce_tensor(acc1[0], self.config.world_size)
                top1.update(reduced_acc1.item(), bsz)

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % self.config.print_freq == 0:
                    if self.config.rank == 0:
                        print(
                            f"Test: [{idx}/{len(val_loader)}]\t"
                            f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                            f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})"
                        )

        if self.config.rank == 0:
            print(f" * Acc@1 {top1.avg:.3f}")

        return losses.avg, top1.avg

    def log_epoch_metrics(self, phase, epoch, metrics, lr):
        """Log epoch metrics to wandb and tensorboard"""
        if self.config.rank == 0:
            # Log to wandb
            if self.config.wandb_initialized and WANDB_AVAILABLE:
                log_dict = {f"{phase}/{k}": v for k, v in metrics.items()}
                log_dict["lr"] = lr
                log_dict["epoch"] = epoch
                wandb.log(log_dict)

            # Log to tensorboard
            if self.writer is not None:
                for key, value in metrics.items():
                    self.writer.add_scalar(f"{phase}/{key}", value, epoch)
                self.writer.add_scalar("lr", lr, epoch)

    def cleanup(self):
        """Clean up logging resources"""
        if self.config.rank == 0:
            if self.writer is not None:
                self.writer.close()
            if self.config.wandb_initialized and WANDB_AVAILABLE:
                wandb.finish()


class UnifiedSupConTrainer:
    """Main class orchestrating the unified training process"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_directories()

        # Initialize managers
        self.dataset_manager = DatasetManager(config)
        self.model_manager = ModelManager(config)
        self.trainer = Trainer(config)

        # Update config with dataset info
        self.config.n_cls = self.dataset_manager.n_classes
        self.best_acc = 0.0

    def setup_directories(self):
        """Setup output directories"""
        total_batch_size = self.config.batch_size * self.config.world_size

        self.model_path = (
            Path("./save/UnifiedSupCon")
            / f"{self.config.dataset}_models_{self.config.loss}"
        )
        self.tb_path = (
            Path("./save/UnifiedSupCon")
            / f"{self.config.dataset}_tensorboard_{self.config.loss}"
        )

        # Create model name with head_type included
        model_name_parts = [
            f"Unified_{self.config.method}_{self.config.loss}_{self.config.dataset}",
            f"{self.config.model}",
            f"head_{self.config.head_type}",  # NEW: Include projection head type
        ]

        # Add feature dimension only if head is not 'none'
        if self.config.head_type != "none":
            model_name_parts.append(f"featdim_{self.config.feat_dim}")

        # Add training parameters
        model_name_parts.extend(
            [
                f"supcon_lr_{self.config.supcon_learning_rate}",
                f"linear_lr_{self.config.linear_learning_rate}",
                f"bsz_{total_batch_size}",
                f"temp_{self.config.temperature}",
                f"trial_{self.config.trial}",
                f"seed_{self.config.seed}",
            ]
        )

        self.model_name = "_".join(model_name_parts)

        if self.config.cosine:
            self.model_name += "_cosine"
        if self.config.warm:
            self.model_name += "_warm"

        # Additional configuration indicators
        if self.config.use_prototypes:
            self.model_name += "_prototypes"
        if self.config.linear_normalized:
            self.model_name += f"_norm_{self.config.linear_loss.lower()}"

        self.save_folder = self.model_path / self.model_name
        self.tb_folder = self.tb_path / self.model_name

        if self.config.rank == 0:
            self.save_folder.mkdir(parents=True, exist_ok=True)
            self.tb_folder.mkdir(parents=True, exist_ok=True)

    def run_contrastive_training(self):
        """Run contrastive pre-training phase"""
        if self.config.rank == 0:
            print("\n" + "=" * 20 + " PHASE 1: CONTRASTIVE PRE-TRAINING " + "=" * 20)

        # Setup data and model
        train_loader, train_sampler = self.dataset_manager.get_contrastive_loader()
        model, criterion = self.model_manager.create_contrastive_model()

        # Setup optimizer
        self.config.learning_rate = self.config.supcon_learning_rate
        optimizer = set_optimizer(self.config, model)
        self._configure_optimizer(optimizer, "contrastive")

        # Training loop
        for epoch in range(1, self.config.supcon_epochs + 1):
            train_sampler.set_epoch(epoch)
            self._adjust_learning_rate(optimizer, epoch, "contrastive")

            # Train one epoch
            start_time = time.time()
            train_loss = self.trainer.train_contrastive_epoch(
                train_loader, model, criterion, optimizer, epoch
            )
            elapsed_time = time.time() - start_time

            if self.config.rank == 0:
                print(
                    f"Contrastive epoch {epoch}, total time {elapsed_time:.2f}, loss: {train_loss:.4f}"
                )

            # Compute metrics based on the new logic
            if self.trainer.metrics_computer.should_compute_metrics(
                epoch, self.config.supcon_epochs
            ):
                if self.config.rank == 0:
                    print(f"Computing metrics at epoch {epoch}...")
                nc_metrics = self.trainer.metrics_computer.compute_contrastive_metrics(
                    model, train_loader
                )

                if self.config.rank == 0:
                    print(
                        f'Contrastive Metrics - ER_intra: {nc_metrics["er_intra"]:.4f}, ER_inter: {nc_metrics["er_inter"]:.4f}'
                    )

                # Log all metrics
                all_metrics = {"train_loss": train_loss, "train_time": elapsed_time}
                all_metrics.update(nc_metrics)
                self.trainer.log_epoch_metrics(
                    "contrastive", epoch, all_metrics, optimizer.param_groups[0]["lr"]
                )
            else:
                # Log basic training metrics
                basic_metrics = {"train_loss": train_loss, "train_time": elapsed_time}
                self.trainer.log_epoch_metrics(
                    "contrastive", epoch, basic_metrics, optimizer.param_groups[0]["lr"]
                )

            # Save checkpoint
            if epoch % self.config.save_freq == 0 and self.config.rank == 0:
                save_file = self.save_folder / f"contrastive_ckpt_epoch_{epoch}.pth"
                save_model(model, optimizer, self.config, epoch, str(save_file))

        # Save final model
        final_model_path = self.save_folder / "contrastive_final.pth"
        if self.config.rank == 0:
            save_model(
                model,
                optimizer,
                self.config,
                self.config.supcon_epochs,
                str(final_model_path),
            )
            print(f"Contrastive model saved to: {final_model_path}")

        return str(final_model_path)

    def run_linear_training(self, contrastive_model_path):
        """Run linear classifier training phase"""
        if self.config.rank == 0:
            print("\n" + "=" * 20 + " PHASE 2: LINEAR CLASSIFIER TRAINING " + "=" * 20)

        # Setup data and model
        train_loader, val_loader, train_sampler = (
            self.dataset_manager.get_linear_loaders()
        )
        model, classifier, criterion = self.model_manager.create_linear_model(
            contrastive_model_path
        )

        # FIX: Set learning_rate attribute for set_optimizer
        self.config.learning_rate = self.config.linear_learning_rate

        # Setup optimizer
        optimizer = set_optimizer(self.config, classifier)
        self._configure_optimizer(optimizer, "linear")

        # Training loop
        for epoch in range(1, self.config.linear_epochs + 1):
            train_sampler.set_epoch(epoch)
            self._adjust_learning_rate(optimizer, epoch, "linear")

            # Train one epoch
            start_time = time.time()
            train_loss, train_acc = self.trainer.train_linear_epoch(
                train_loader, model, classifier, criterion, optimizer, epoch
            )
            elapsed_time = time.time() - start_time

            if self.config.rank == 0:
                print(
                    f"Linear train epoch {epoch}, total time {elapsed_time:.2f}, accuracy: {train_acc:.2f}%"
                )

            # Validate
            val_loss, val_acc = self.trainer.validate_linear(
                val_loader, model, classifier, criterion
            )

            # Compute NC metrics based on the new logic
            if self.trainer.metrics_computer.should_compute_metrics(
                epoch, self.config.linear_epochs
            ):
                if self.config.rank == 0:
                    print(f"Computing linear metrics at epoch {epoch}...")
                nc_metrics = self.trainer.metrics_computer.compute_linear_metrics(
                    model, classifier, val_loader
                )

                if self.config.rank == 0:
                    print(
                        f'Linear Metrics - MIR: {nc_metrics["mir"]:.4f}, HDR: {nc_metrics["hdr"]:.4f}'
                    )
                    print(
                        f'               - W_inst: {nc_metrics["w_inst_alignment"]:.4f}, '
                        f'W_erank: {nc_metrics["w_erank"]:.4f}, W_class: {nc_metrics["w_class_alignment"]:.2f}'
                    )

                # Combine all metrics
                all_metrics = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_time": elapsed_time,
                }
                all_metrics.update({f"val_{k}": v for k, v in nc_metrics.items()})

                self.trainer.log_epoch_metrics(
                    "linear", epoch, all_metrics, optimizer.param_groups[0]["lr"]
                )
            else:
                # Log basic metrics
                basic_metrics = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_time": elapsed_time,
                }
                self.trainer.log_epoch_metrics(
                    "linear", epoch, basic_metrics, optimizer.param_groups[0]["lr"]
                )

            # Save best model
            if val_acc > self.best_acc and self.config.rank == 0:
                self.best_acc = val_acc
                self._save_best_linear_model(model, classifier, optimizer, epoch)

                if self.config.wandb_initialized and WANDB_AVAILABLE:
                    wandb.run.summary["best_val_acc"] = self.best_acc

        if self.config.rank == 0:
            print(f"Best validation accuracy: {self.best_acc:.2f}%")

    def run_prototype_evaluation_with_metrics(self, contrastive_model_path):
        """
        Run prototype-based evaluation with NC metrics computation.
        Returns: (accuracy, nc_metrics_dict)
        """
        if self.config.rank == 0:
            print("Computing class prototypes from training data...")

        # Load the pre-trained model
        train_loader, val_loader, train_sampler = (
            self.dataset_manager.get_linear_loaders()
        )
        model, criterion = self.model_manager.create_contrastive_model()

        # Load pretrained weights
        if contrastive_model_path:
            self.model_manager._load_pretrained_weights(model, contrastive_model_path)

        # Freeze the encoder
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Initialize prototype classifier
        if self.config.head_type == "none":
            feat_dim = {
                "resnet18": 512,
                "resnet34": 512,
                "resnet50": 2048,
                "resnet101": 2048,
            }[self.config.model]
        else:
            feat_dim = self.config.feat_dim

        prototype_classifier = EfficientPrototypeClassifier(
            num_classes=self.config.n_cls,
            feature_dim=feat_dim,
            device=f"cuda:{self.config.local_rank}",
        )

        # Build prototypes
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.cuda(self.config.local_rank, non_blocking=True)
                labels = labels.cuda(self.config.local_rank, non_blocking=True)

                if isinstance(images, list):
                    images = images[0]

                features = model(images)
                prototype_classifier.update(features, labels)

                if self.config.rank == 0 and (batch_idx + 1) % 50 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(train_loader)} batches")

        # Get final prototypes
        final_prototypes = prototype_classifier.get_prototypes()

        # Evaluate on validation set
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.cuda(self.config.local_rank, non_blocking=True)
                labels = labels.cuda(self.config.local_rank, non_blocking=True)

                features = model(images)
                predictions, _ = prototype_classifier.predict(features)

                correct = predictions.eq(labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

        accuracy = 100.0 * total_correct / total_samples

        # Compute NC metrics
        nc_metrics = self.compute_prototype_nc_metrics(
            model, prototype_classifier, val_loader
        )

        # Save prototypes and metrics
        if self.config.rank == 0:
            save_dict = {
                "prototypes": final_prototypes.cpu(),
                "accuracy": accuracy,
                "nc_metrics": nc_metrics,
                "config": self.config,
                "model_state": model.state_dict(),
            }
            save_path = self.save_folder / "phase2_prototypes.pth"
            torch.save(save_dict, save_path)
            print(f"Phase 2 prototypes saved to: {save_path}")

        # Log to wandb
        if self.config.rank == 0 and self.config.wandb_initialized and WANDB_AVAILABLE:
            wandb.log(
                {
                    "phase2_prototype/accuracy": accuracy,
                    "phase2_prototype/mir": nc_metrics["mir"],
                    "phase2_prototype/hdr": nc_metrics["hdr"],
                    "phase2_prototype/w_inst_alignment": nc_metrics["w_inst_alignment"],
                    "phase2_prototype/w_erank": nc_metrics["w_erank"],
                    "phase2_prototype/w_class_alignment": nc_metrics[
                        "w_class_alignment"
                    ],
                }
            )

        return accuracy, nc_metrics

    def run_prototype_evaluation(self, contrastive_model_path):
        """
        Run prototype-based evaluation instead of linear classifier training.
        Now includes NC metrics computation using prototypes as classifier weights.

        Args:
            contrastive_model_path: Path to pre-trained contrastive model
        """
        if self.config.rank == 0:
            print("\n" + "=" * 20 + " PHASE 2: PROTOTYPE-BASED EVALUATION " + "=" * 20)
            print("Computing class prototypes from training data...")

        # Load the pre-trained model
        train_loader, val_loader, train_sampler = (
            self.dataset_manager.get_linear_loaders()
        )
        model, criterion = self.model_manager.create_contrastive_model()

        # Load pretrained weights
        if contrastive_model_path:
            self.model_manager._load_pretrained_weights(model, contrastive_model_path)

        # Freeze the encoder
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Initialize prototype classifier
        # Get feature dimension based on model architecture and head type
        if self.config.head_type == "none":
            # No projection head - use encoder output dimension
            feat_dim = {
                "resnet18": 512,
                "resnet34": 512,
                "resnet50": 2048,
                "resnet101": 2048,
            }[self.config.model]
        else:
            # With projection head - use configured dimension
            feat_dim = self.config.feat_dim

        prototype_classifier = EfficientPrototypeClassifier(
            num_classes=self.config.n_cls,
            feature_dim=feat_dim,
            device=f"cuda:{self.config.local_rank}",
        )

        # Phase 1: Build prototypes from training data
        if self.config.rank == 0:
            print(f"Building prototypes from {len(train_loader)} training batches...")

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.cuda(self.config.local_rank, non_blocking=True)
                labels = labels.cuda(self.config.local_rank, non_blocking=True)

                # Handle TwoCropTransform if present
                if isinstance(images, list):
                    # Use only the first augmentation for prototype computation
                    images = images[0]

                # Get normalized features from encoder
                features = model(images)  # Already normalized in SupConResNet

                # Update prototypes
                prototype_classifier.update(features, labels)

                # Print progress
                if (
                    self.config.rank == 0
                    and (batch_idx + 1) % self.config.print_freq == 0
                ):
                    print(f"  Processed {batch_idx + 1}/{len(train_loader)} batches")

        # Get final prototypes
        final_prototypes = prototype_classifier.get_prototypes()

        if self.config.rank == 0:
            print(f"\nPrototypes computed for {self.config.n_cls} classes")
            print(f"Prototype shape: {final_prototypes.shape}")

            # Check prototype norms (should all be 1.0 or 0.0 for missing classes)
            prototype_norms = torch.norm(final_prototypes, p=2, dim=1)
            print(
                f"Prototype norms - min: {prototype_norms.min():.4f}, "
                f"max: {prototype_norms.max():.4f}, mean: {prototype_norms.mean():.4f}"
            )

        # Phase 2: Evaluate on validation set
        if self.config.rank == 0:
            print("\nEvaluating prototype classifier on validation set...")

        total_correct = 0
        total_samples = 0
        all_similarities = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.cuda(self.config.local_rank, non_blocking=True)
                labels = labels.cuda(self.config.local_rank, non_blocking=True)

                # Get features
                features = model(images)

                # Predict using prototypes
                predictions, similarities = prototype_classifier.predict(features)

                # Calculate accuracy
                correct = predictions.eq(labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

                # Store similarities for analysis
                all_similarities.append(similarities.cpu())

        # Calculate final accuracy
        accuracy = 100.0 * total_correct / total_samples

        if self.config.rank == 0:
            print(f"\n" + "=" * 50)
            print(f"PROTOTYPE CLASSIFIER RESULTS:")
            print(f"  Validation Accuracy: {accuracy:.2f}%")
            print(f"  Total Samples: {total_samples}")
            print("=" * 50)

        # ================== NEW: COMPUTE NC METRICS ==================
        # Compute Neural Collapse metrics using prototypes as classifier weights
        nc_metrics = self.compute_prototype_nc_metrics(
            model, prototype_classifier, val_loader
        )

        if self.config.rank == 0:
            print(f"\n" + "=" * 50)
            print(f"NEURAL COLLAPSE METRICS (Using Prototypes as Weights):")
            print(f"  MIR (Mutual Information Ratio): {nc_metrics['mir']:.4f}")
            print(f"  HDR (Entropy Difference Ratio): {nc_metrics['hdr']:.4f}")
            print(
                f"  W_inst (Instance Alignment): {nc_metrics['w_inst_alignment']:.4f}"
            )
            print(f"  W_erank (Effective Rank): {nc_metrics['w_erank']:.4f}")
            print(f"  W_class (Class Alignment): {nc_metrics['w_class_alignment']:.4f}")
            print("=" * 50)

            # Log NC metrics
            if self.trainer.writer is not None:
                for key, value in nc_metrics.items():
                    self.trainer.writer.add_scalar(f"prototype/{key}", value, 0)

            if self.config.wandb_initialized and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "prototype/val_acc": accuracy,
                        "prototype/mir": nc_metrics["mir"],
                        "prototype/hdr": nc_metrics["hdr"],
                        "prototype/w_inst_alignment": nc_metrics["w_inst_alignment"],
                        "prototype/w_erank": nc_metrics["w_erank"],
                        "prototype/w_class_alignment": nc_metrics["w_class_alignment"],
                        "prototype/num_classes": self.config.n_cls,
                        "prototype/feature_dim": feat_dim,
                    }
                )
        # ==============================================================

        # Save prototypes and metrics for later analysis
        if self.config.rank == 0:
            save_dict = {
                "prototypes": final_prototypes.cpu(),
                "accuracy": accuracy,
                "nc_metrics": nc_metrics,  # Add NC metrics to saved data
                "config": self.config,
                "model_state": model.state_dict(),
            }
            save_path = self.save_folder / "prototypes.pth"
            torch.save(save_dict, save_path)
            print(f"\nPrototypes and metrics saved to: {save_path}")

        return accuracy

    def compute_prototype_nc_metrics(self, model, prototype_classifier, loader):
        """
        Compute NC metrics using prototypes as classifier weights.

        Args:
            model: The encoder model
            prototype_classifier: The prototype classifier with computed prototypes
            loader: Data loader for computing metrics

        Returns:
            Dictionary of NC metrics
        """
        if self.config.dataset not in ["cifar10", "cifar100"]:
            return {
                "mir": 0.0,
                "hdr": 0.0,
                "w_inst_alignment": 0.0,
                "w_erank": 0.0,
                "w_class_alignment": 0.0,
            }

        model.eval()

        all_features = []
        all_prototype_weights = []
        all_labels = []

        if self.config.rank == 0:
            print(f"\nComputing NC metrics using prototypes as weights...")

        with torch.no_grad():
            # Get the prototypes (these act as our "classifier weights")
            prototypes = prototype_classifier.get_prototypes()  # Shape: (C, d)

            # Prototypes are already normalized, but verify
            prototypes_normalized = F.normalize(prototypes, p=2, dim=1)  # Shape: (C, d)

            for idx, (images, labels) in enumerate(loader):
                if torch.cuda.is_available():
                    images = images.cuda(self.config.local_rank, non_blocking=True)
                    labels = labels.cuda(self.config.local_rank, non_blocking=True)

                # Get normalized features from the model
                features = model(images)  # Already normalized by SupConResNet

                # Get the corresponding prototype for each sample (as if it were the classifier weight)
                # This is equivalent to getting the weight vector for each sample's true class
                corresponding_prototypes = prototypes_normalized[
                    labels
                ]  # Shape: (N, d)

                # Collect data
                all_features.append(features.detach().cpu())
                all_prototype_weights.append(corresponding_prototypes.detach().cpu())
                all_labels.append(labels.detach().cpu())

                if idx >= 20:  # Use enough batches for stable metrics
                    break

        if len(all_features) > 0:
            z = torch.cat(all_features, dim=0)  # Normalized embeddings (N, d)
            b = torch.cat(
                all_prototype_weights, dim=0
            )  # Corresponding prototype "weights" (N, d)
            y = torch.cat(all_labels, dim=0)  # Labels (N,)

            # Transpose prototypes for util functions - Shape: (d, C)
            W = prototypes_normalized.detach().cpu().T

            if self.config.rank == 0:
                print(
                    f"Features shape: {z.shape}, Prototype weights shape: {b.shape}, Labels shape: {y.shape}"
                )
                print(f"Full prototype matrix shape: {W.shape}")

                # Verify normalization
                print(
                    f"Features norm check: mean={z.norm(dim=1).mean():.4f}, std={z.norm(dim=1).std():.4f}"
                )
                print(
                    f"Prototype weights norm check: mean={b.norm(dim=1).mean():.4f}, std={b.norm(dim=1).std():.4f}"
                )

            try:
                # Compute metrics using prototypes as weights
                mir, hdr = util.weight_embeddings_information(z, W, y)
                w_instance_alignment, w_erank, w_class_alignment = util.NC(b, z, W, y)

                if self.config.rank == 0:
                    print(f"Successfully computed NC metrics with prototypes")
                    print(f"MIR: {mir:.4f}, HDR: {hdr:.4f}")
                    print(
                        f"W_inst: {w_instance_alignment:.4f}, W_erank: {w_erank:.4f}, W_class: {w_class_alignment:.4f}"
                    )

                return {
                    "mir": mir,
                    "hdr": hdr,
                    "w_inst_alignment": w_instance_alignment,
                    "w_erank": w_erank,
                    "w_class_alignment": w_class_alignment,
                }
            except Exception as e:
                if self.config.rank == 0:
                    print(f"Warning: NC metrics computation failed: {e}")
                    import traceback

                    traceback.print_exc()
                return {
                    "mir": 0.0,
                    "hdr": 0.0,
                    "w_inst_alignment": 0.0,
                    "w_erank": 0.0,
                    "w_class_alignment": 0.0,
                }

        return {
            "mir": 0.0,
            "hdr": 0.0,
            "w_inst_alignment": 0.0,
            "w_erank": 0.0,
            "w_class_alignment": 0.0,
        }

    def run_linear_training_with_phase_id(
        self, contrastive_model_path, phase_id="linear"
    ):
        """
        Run linear classifier training with phase-specific logging.
        Returns: (best_accuracy, final_nc_metrics)
        """
        phase_name = (
            "Standard Linear" if phase_id == "phase3_standard" else "Normalized Linear"
        )

        if self.config.rank == 0:
            print(f"Training {phase_name} classifier")
            if self.config.linear_normalized:
                print(f"  Loss: {self.config.linear_loss}")
                print(f"  Temperature: {self.config.linear_temperature:.3f}")
            else:
                print(f"  Loss: Cross Entropy")

        # Setup data and model
        train_loader, val_loader, train_sampler = (
            self.dataset_manager.get_linear_loaders()
        )
        model, classifier, criterion = self.model_manager.create_linear_model(
            contrastive_model_path
        )

        # Set learning_rate attribute
        self.config.learning_rate = self.config.linear_learning_rate

        # Setup optimizer
        optimizer = set_optimizer(self.config, classifier)
        self._configure_optimizer(optimizer, "linear")

        # Track best metrics
        best_accuracy = 0.0
        best_nc_metrics = None

        # Training loop
        for epoch in range(1, self.config.linear_epochs + 1):
            train_sampler.set_epoch(epoch)
            self._adjust_learning_rate(optimizer, epoch, "linear")

            # Train one epoch
            start_time = time.time()
            train_loss, train_acc = self.trainer.train_linear_epoch(
                train_loader, model, classifier, criterion, optimizer, epoch
            )
            elapsed_time = time.time() - start_time

            if self.config.rank == 0 and epoch % 10 == 0:
                print(
                    f"{phase_name} - Epoch {epoch}: train_acc={train_acc:.2f}%, time={elapsed_time:.2f}s"
                )

            # Validate
            val_loss, val_acc = self.trainer.validate_linear(
                val_loader, model, classifier, criterion
            )

            # Compute NC metrics at specified intervals
            nc_metrics = {
                "mir": 0.0,
                "hdr": 0.0,
                "w_inst_alignment": 0.0,
                "w_erank": 0.0,
                "w_class_alignment": 0.0,
            }

            if self.trainer.metrics_computer.should_compute_metrics(
                epoch, self.config.linear_epochs
            ):
                if self.config.rank == 0:
                    print(f"Computing NC metrics for {phase_name} at epoch {epoch}...")
                nc_metrics = self.trainer.metrics_computer.compute_linear_metrics(
                    model, classifier, val_loader
                )

                if self.config.rank == 0:
                    print(f"{phase_name} NC Metrics:")
                    print(
                        f'  MIR: {nc_metrics["mir"]:.4f}, HDR: {nc_metrics["hdr"]:.4f}'
                    )
                    print(
                        f'  W_inst: {nc_metrics["w_inst_alignment"]:.4f}, '
                        f'W_erank: {nc_metrics["w_erank"]:.4f}, '
                        f'W_class: {nc_metrics["w_class_alignment"]:.2f}'
                    )

            # Log metrics with phase-specific prefix
            if self.config.rank == 0:
                log_metrics = {
                    f"{phase_id}/train_loss": train_loss,
                    f"{phase_id}/train_acc": train_acc,
                    f"{phase_id}/val_loss": val_loss,
                    f"{phase_id}/val_acc": val_acc,
                    f"{phase_id}/learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }

                # Add NC metrics if computed
                if nc_metrics["mir"] > 0:
                    for key, value in nc_metrics.items():
                        log_metrics[f"{phase_id}/{key}"] = value

                # Log to tensorboard
                if self.trainer.writer is not None:
                    for key, value in log_metrics.items():
                        if key != "epoch":
                            self.trainer.writer.add_scalar(key, value, epoch)

                # Log to wandb
                if self.config.wandb_initialized and WANDB_AVAILABLE:
                    wandb.log(log_metrics)

            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_nc_metrics = nc_metrics.copy()

                if self.config.rank == 0:
                    save_file = self.save_folder / f"{phase_id}_best_model.pth"
                    save_state = {
                        "model": model.state_dict(),
                        "classifier": classifier.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_acc": best_accuracy,
                        "nc_metrics": best_nc_metrics,
                        "config": self.config,
                    }
                    torch.save(save_state, save_file)

        if self.config.rank == 0:
            print(
                f"{phase_name} training complete. Best accuracy: {best_accuracy:.2f}%"
            )

        # Ensure we have NC metrics for the best model
        if best_nc_metrics is None or best_nc_metrics["mir"] == 0:
            if self.config.rank == 0:
                print(f"Computing final NC metrics for {phase_name}...")
            best_nc_metrics = self.trainer.metrics_computer.compute_linear_metrics(
                model, classifier, val_loader
            )

        self.best_acc = best_accuracy
        return best_accuracy, best_nc_metrics

    def run_training(self):
        """Run the complete training pipeline with all four phases"""
        if self.config.rank == 0:
            print("=" * 60)
            print("UNIFIED SUPERVISED CONTRASTIVE LEARNING - FOUR PHASES")
            print("=" * 60)
            self._print_config()

        # Setup logging
        self.trainer.setup_logging(str(self.tb_folder))

        # Initialize results storage
        results = {}
        contrastive_model_path = None

        # PHASE 1: Contrastive pre-training
        if not self.config.skip_contrastive:
            if self.config.rank == 0:
                print("\n" + "=" * 50)
                print(" PHASE 1: CONTRASTIVE PRE-TRAINING")
                print("=" * 50)
            contrastive_model_path = self.run_contrastive_training()
            results["phase1_model"] = contrastive_model_path
        else:
            contrastive_model_path = self.config.supcon_ckpt
            if self.config.rank == 0:
                print(
                    f"\n[INFO] Skipping Phase 1: Using provided model: {contrastive_model_path}"
                )

        # PHASE 2: Prototype evaluation with NC metrics
        if not self.config.skip_prototype:
            if self.config.rank == 0:
                print("\n" + "=" * 50)
                print(" PHASE 2: PROTOTYPE EVALUATION")
                print("=" * 50)

            prototype_accuracy, prototype_nc_metrics = (
                self.run_prototype_evaluation_with_metrics(contrastive_model_path)
            )
            results["phase2_accuracy"] = prototype_accuracy
            results["phase2_nc_metrics"] = prototype_nc_metrics

            if self.config.rank == 0:
                print(f"\n✓ Phase 2 Complete:")
                print(f"  - Accuracy: {prototype_accuracy:.2f}%")
                print(
                    f"  - NC Metrics: MIR={prototype_nc_metrics['mir']:.4f}, HDR={prototype_nc_metrics['hdr']:.4f}"
                )
        else:
            if self.config.rank == 0:
                print(f"\n[INFO] Skipping Phase 2: Prototype evaluation")

        # PHASE 3: Standard Linear Probing (with magnitude)
        if not self.config.skip_linear:
            if self.config.rank == 0:
                print("\n" + "=" * 50)
                print(" PHASE 3: STANDARD LINEAR PROBING (CE LOSS)")
                print("=" * 50)

            # Configure for standard linear probing
            self.config.linear_normalized = False
            self.config.linear_loss = "CE"
            self.config.linear_temperature = 0.2  # Not used for CE

            # Run standard linear training
            self.best_acc = 0.0  # Reset best accuracy
            standard_acc, standard_nc_metrics = self.run_linear_training_with_phase_id(
                contrastive_model_path, phase_id="phase3_standard"
            )

            results["phase3_accuracy"] = standard_acc
            results["phase3_nc_metrics"] = standard_nc_metrics

            if self.config.rank == 0:
                print(f"\n✓ Phase 3 Complete:")
                print(f"  - Best Accuracy: {standard_acc:.2f}%")
                print(
                    f"  - NC Metrics: MIR={standard_nc_metrics['mir']:.4f}, HDR={standard_nc_metrics['hdr']:.4f}"
                )
        else:
            if self.config.rank == 0:
                print(f"\n[INFO] Skipping Phase 3: Standard linear probing")

        # PHASE 4: Normalized Linear Probing with NormFace
        if not self.config.skip_normalized_linear:
            if self.config.rank == 0:
                print("\n" + "=" * 50)
                print(" PHASE 4: NORMALIZED LINEAR PROBING (NormFace LOSS)")
                print("=" * 50)

            # Configure for normalized linear probing
            self.config.linear_normalized = True
            self.config.linear_loss = "NormFace"
            self.config.linear_temperature = (
                self.config.temperature
            )  # Use contrastive temperature

            # Reset best accuracy for this phase
            self.best_acc = 0.0

            # Run normalized linear training
            normalized_acc, normalized_nc_metrics = (
                self.run_linear_training_with_phase_id(
                    contrastive_model_path, phase_id="phase4_normalized"
                )
            )

            results["phase4_accuracy"] = normalized_acc
            results["phase4_nc_metrics"] = normalized_nc_metrics

            if self.config.rank == 0:
                print(f"\n✓ Phase 4 Complete:")
                print(f"  - Best Accuracy: {normalized_acc:.2f}%")
                print(f"  - Temperature: {self.config.temperature:.3f}")
                print(
                    f"  - NC Metrics: MIR={normalized_nc_metrics['mir']:.4f}, HDR={normalized_nc_metrics['hdr']:.4f}"
                )
        else:
            if self.config.rank == 0:
                print(f"\n[INFO] Skipping Phase 4: Normalized linear probing")

        # Final summary with comparison
        if self.config.rank == 0:
            print("\n" + "=" * 60)
            print(" TRAINING COMPLETED - SUMMARY")
            print("=" * 60)

            if "phase1_model" in results:
                print(f"\n📁 Contrastive Model: {results['phase1_model']}")

            print("\n📊 Accuracy Comparison:")
            if "phase2_accuracy" in results:
                print(f"  Phase 2 (Prototype):     {results['phase2_accuracy']:.2f}%")
            if "phase3_accuracy" in results:
                print(f"  Phase 3 (Standard CE):   {results['phase3_accuracy']:.2f}%")
            if "phase4_accuracy" in results:
                print(
                    f"  Phase 4 (Normalized Linear): {results['phase4_accuracy']:.2f}%"
                )

            print("\n📈 Neural Collapse Metrics Comparison:")
            print("  " + "-" * 50)
            print("  Phase    | MIR    | HDR    | W_inst | W_erank | W_class")
            print("  " + "-" * 50)

            for phase_num, phase_key in [(2, "phase2"), (3, "phase3"), (4, "phase4")]:
                metrics_key = f"{phase_key}_nc_metrics"
                if metrics_key in results:
                    m = results[metrics_key]
                    print(
                        f"  Phase {phase_num}  | {m['mir']:.4f} | {m['hdr']:.4f} | "
                        f"{m['w_inst_alignment']:.4f} | {m['w_erank']:.4f} | {m['w_class_alignment']:.2f}"
                    )
            print("  " + "-" * 50)

            # Log final comparison to wandb
            if self.config.wandb_initialized and WANDB_AVAILABLE:
                summary_dict = {}
                for key, value in results.items():
                    if isinstance(value, dict):  # NC metrics
                        for metric_name, metric_value in value.items():
                            summary_dict[f"{key}_{metric_name}"] = metric_value
                    else:
                        summary_dict[key] = value
                wandb.log(summary_dict)

        # Cleanup
        self.trainer.cleanup()

    def _configure_optimizer(self, optimizer, phase):
        """Configure optimizer parameters for specific phase"""
        if phase == "contrastive":
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.config.supcon_learning_rate
                param_group["weight_decay"] = self.config.supcon_weight_decay
                param_group["momentum"] = self.config.supcon_momentum
        elif phase == "linear":
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.config.linear_learning_rate
                param_group["weight_decay"] = self.config.linear_weight_decay
                param_group["momentum"] = self.config.linear_momentum

    def _adjust_learning_rate(self, optimizer, epoch, phase):
        """Adjust learning rate based on phase and epoch"""
        temp_config = argparse.Namespace(**vars(self.config))

        if phase == "contrastive":
            temp_config.learning_rate = self.config.supcon_learning_rate
            temp_config.epochs = self.config.supcon_epochs
            temp_config.weight_decay = self.config.supcon_weight_decay
            temp_config.momentum = self.config.supcon_momentum

            # ✅ Only set step decay if NOT using cosine
            if not self.config.cosine:
                temp_config.lr_decay_epochs = [
                    int(x) for x in self.config.supcon_lr_decay_epochs.split(",")
                ]
                temp_config.lr_decay_rate = self.config.supcon_lr_decay_rate
            else:
                # For cosine annealing, step decay parameters are not used by adjust_learning_rate
                # But we still need lr_decay_rate for eta_min calculation in cosine formula
                temp_config.lr_decay_epochs = (
                    []
                )  # Empty - won't be used with cosine=True
                temp_config.lr_decay_rate = getattr(
                    self.config, "supcon_lr_decay_rate", 0.1
                )

        elif phase == "linear":
            temp_config.learning_rate = self.config.linear_learning_rate
            temp_config.epochs = self.config.linear_epochs
            temp_config.weight_decay = self.config.linear_weight_decay
            temp_config.momentum = self.config.linear_momentum

            # Linear phase typically uses step decay even when contrastive uses cosine
            temp_config.lr_decay_epochs = [
                int(x) for x in self.config.linear_lr_decay_epochs.split(",")
            ]
            temp_config.lr_decay_rate = self.config.linear_lr_decay_rate

        # Set warmup parameters if needed
        if self.config.warm:
            temp_config.warmup_from = 0.01
            temp_config.warm_epochs = 5 if self.config.dataset == "imagenet1k" else 10
            if self.config.cosine:
                eta_min = temp_config.learning_rate * (temp_config.lr_decay_rate**3)
                temp_config.warmup_to = (
                    eta_min
                    + (temp_config.learning_rate - eta_min)
                    * (
                        1
                        + math.cos(
                            math.pi * temp_config.warm_epochs / temp_config.epochs
                        )
                    )
                    / 2
                )
            else:
                temp_config.warmup_to = temp_config.learning_rate
        else:
            # Set default values even when warmup is disabled
            temp_config.warmup_from = 0.01
            temp_config.warm_epochs = 0
            temp_config.warmup_to = temp_config.learning_rate

        # Call the actual learning rate adjustment
        adjust_learning_rate(temp_config, optimizer, epoch)

    def _save_best_linear_model(self, model, classifier, optimizer, epoch):
        """Save the best linear classifier model"""
        save_file = self.save_folder / "best_linear_classifier.pth"
        save_state = {
            "model": model.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": self.best_acc,
            "config": self.config,
        }
        torch.save(save_state, save_file)

    def _print_config(self):
        """Print training configuration"""
        total_batch_size = self.config.batch_size * self.config.world_size
        print("\n========== Training Configuration ==========")
        print(f"Dataset: {self.config.dataset}")
        if self.config.dataset == "imagenet100" and self.config.imagenet100_path:
            print(f"ImageNet-100 Path: {self.config.imagenet100_path}")
        elif self.config.dataset == "imagenet1k" and self.config.imagenet1k_path:
            print(f"ImageNet-1K Path: {self.config.imagenet1k_path}")

        print(f"\nModel Architecture: {self.config.model}")
        print(f"Projection Head: {self.config.head_type}")
        if self.config.head_type != "none":
            print(f"Projection Dimension: {self.config.feat_dim}")

        print(f"\n--- Four-Phase Training Plan ---")
        phases = [
            ("Phase 1 - Contrastive Pre-training", not self.config.skip_contrastive),
            ("Phase 2 - Prototype Evaluation", not self.config.skip_prototype),
            ("Phase 3 - Standard Linear (CE)", not self.config.skip_linear),
            ("Phase 4 - Normalized Linear", not self.config.skip_normalized_linear),
        ]

        for phase_name, enabled in phases:
            status = "✓ ENABLED" if enabled else "✗ SKIPPED"
            print(f"  {phase_name}: {status}")

        if not self.config.skip_contrastive:
            print(f"\nContrastive Settings:")
            print(f"  Loss: {self.config.loss}")
            print(f"  Temperature: {self.config.temperature}")
            print(f"  Epochs: {self.config.supcon_epochs}")
            print(f"  Learning Rate: {self.config.supcon_learning_rate}")
            print(f"  Cosine LR: {self.config.cosine}")
            print(f"  Warmup: {self.config.warm}")

        if not self.config.skip_linear or not self.config.skip_normalized_linear:
            print(f"\nLinear Probing Settings:")
            print(f"  Epochs: {self.config.linear_epochs}")
            print(f"  Learning Rate: {self.config.linear_learning_rate}")

            if not self.config.skip_normalized_linear:
                print(
                    f"  Phase 4 Temperature: {self.config.temperature} (matches contrastive)"
                )

        print(f"\nTraining Setup:")
        print(f"  Batch Size per GPU: {self.config.batch_size}")
        print(f"  Total Batch Size: {total_batch_size}")
        print(f"  World Size: {self.config.world_size}")
        print(
            f"  GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )
        print(f"  Seed: {self.config.seed}")
        print(
            f"  NC Metrics Frequency: Every {self.config.compute_metrics_freq} epochs"
        )

        print(f"\nOutput Path: {self.save_folder}")
        print("============================================\n")


def parse_arguments():
    """Parse command line arguments and return configuration"""
    # Stage 1: Minimal parser to capture --config
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, help="Path to config YAML file")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    # Load YAML config if specified
    config = {}
    if pre_args.config and YAML_AVAILABLE:
        with open(pre_args.config, "r") as f:
            config = yaml.safe_load(f)
    elif pre_args.config and not YAML_AVAILABLE:
        print(
            "Warning: YAML config file specified but yaml not available. Install with: pip install pyyaml"
        )

    # Stage 2: Full parser with defaults from config
    parser = argparse.ArgumentParser(
        "Unified SupCon Training - Four Phase Pipeline",
        parents=[pre_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default=config.get("dataset", "cifar10"),
        choices=["cifar10", "cifar100", "imagenet100", "imagenet1k"],
        help="dataset",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=config.get("data_folder", "./datasets/"),
        help="path to dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.get("batch_size", 256),
        help="batch_size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=config.get("num_workers", 16),
        help="num of workers",
    )

    # ImageNet paths
    parser.add_argument(
        "--imagenet100_path",
        type=str,
        default=config.get("imagenet100_path", None),
        help="Path to ImageNet-100 dataset directory",
    )
    parser.add_argument(
        "--imagenet1k_path",
        type=str,
        default=config.get("imagenet1k_path", None),
        help="Path to ImageNet-1K dataset directory",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default=config.get("model", "resnet18"),
        help="model architecture",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=config.get("method", "SupCon"),
        choices=["SupCon", "SimCLR"],
        help="choose method",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=config.get("loss", "SCL"),
        choices=["SCL"],
        help="loss function",
    )
    parser.add_argument(
        "--temp", type=float, default=config.get("temp", 0.07), help="temperature"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.get("temperature", 0.07),
        help="temperature (alias for --temp)",
    )
    parser.add_argument(
        "--temp_pos_neg_ratio",
        type=float,
        default=config.get("temp_pos_neg_ratio", 10.0),
        help="positive vs negative pairs temperature ratio",
    )

    parser.add_argument(
        "--head_type",
        type=str,
        default=config.get("head_type", "mlp"),
        choices=["mlp", "linear", "none"],
        help="Type of projection head: mlp (2-layer), linear (1-layer), or none (no projection)",
    )
    parser.add_argument(
        "--feat_dim",
        type=int,
        default=config.get("feat_dim", 128),
        help="Feature dimension for projection head (ignored if head_type is none)",
    )
    parser.add_argument(
        "--use_prototypes",
        action="store_true",
        default=config.get("use_prototypes", False),
        help="Use prototype-based evaluation instead of linear classifier",
    )
    parser.add_argument(
        "--prototype_update_freq",
        type=int,
        default=config.get("prototype_update_freq", 1),
        help="Update prototypes every N batches (1 = every batch)",
    )

    # Contrastive training parameters
    parser.add_argument(
        "--supcon_epochs",
        type=int,
        default=config.get("supcon_epochs", 1000),
        help="contrastive pre-training epochs",
    )
    parser.add_argument(
        "--supcon_learning_rate",
        type=float,
        default=config.get("supcon_learning_rate", 0.05),
        help="contrastive learning rate",
    )
    parser.add_argument(
        "--supcon_lr_decay_epochs",
        type=str,
        default=config.get("supcon_lr_decay_epochs", "700,800,900"),
        help="contrastive lr decay epochs",
    )
    parser.add_argument(
        "--supcon_lr_decay_rate",
        type=float,
        default=config.get("supcon_lr_decay_rate", 0.1),
        help="contrastive lr decay rate",
    )
    parser.add_argument(
        "--supcon_weight_decay",
        type=float,
        default=config.get("supcon_weight_decay", 1e-4),
        help="contrastive weight decay",
    )
    parser.add_argument(
        "--supcon_momentum",
        type=float,
        default=config.get("supcon_momentum", 0.9),
        help="contrastive momentum",
    )

    # Linear classifier parameters
    parser.add_argument(
        "--linear_epochs",
        type=int,
        default=config.get("linear_epochs", 100),
        help="linear classifier training epochs",
    )
    parser.add_argument(
        "--linear_learning_rate",
        type=float,
        default=config.get("linear_learning_rate", 0.1),
        help="linear learning rate",
    )
    parser.add_argument(
        "--linear_lr_decay_epochs",
        type=str,
        default=config.get("linear_lr_decay_epochs", "60,75,90"),
        help="linear lr decay epochs",
    )
    parser.add_argument(
        "--linear_lr_decay_rate",
        type=float,
        default=config.get("linear_lr_decay_rate", 0.2),
        help="linear lr decay rate",
    )
    parser.add_argument(
        "--linear_weight_decay",
        type=float,
        default=config.get("linear_weight_decay", 0.0),
        help="linear weight decay",
    )
    parser.add_argument(
        "--linear_momentum",
        type=float,
        default=config.get("linear_momentum", 0.9),
        help="linear momentum",
    )
    parser.add_argument(
        "--linear_normalized",
        action="store_true",
        default=config.get("linear_normalized", False),
        help="Use normalized features and weights for linear probing (no bias)",
    )
    parser.add_argument(
        "--linear_loss",
        type=str,
        default=config.get("linear_loss", "CE"),
        choices=["CE", "NormFace", "NTCE", "NONL"],
        help="Loss function for linear probing when using normalized features",
    )
    parser.add_argument(
        "--linear_temperature",
        type=float,
        default=config.get("linear_temperature", 0.2),
        help="Temperature for normalized linear losses (NormFace, NTCE, NONL)",
    )

    # Training options
    parser.add_argument(
        "--cosine",
        action="store_true",
        default=config.get("cosine", False),
        help="using cosine annealing",
    )
    parser.add_argument(
        "--syncBN",
        action="store_true",
        default=config.get("syncBN", False),
        help="using synchronized batch normalization",
    )
    parser.add_argument(
        "--warm",
        action="store_true",
        default=config.get("warm", False),
        help="warm-up for large batch training",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=config.get("print_freq", 10),
        help="print frequency",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=config.get("save_freq", 50),
        help="save frequency",
    )
    parser.add_argument(
        "--trial", type=str, default=config.get("trial", "0"), help="trial id"
    )
    parser.add_argument(
        "--seed", type=int, default=config.get("seed", 42), help="random seed"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=config.get("deterministic", False),
        help="deterministic behavior (slower but reproducible)",
    )

    # Phase control - UPDATED FOR FOUR PHASES
    parser.add_argument(
        "--skip_contrastive",
        action="store_true",
        default=config.get("skip_contrastive", False),
        help="Skip Phase 1: Contrastive pre-training",
    )
    parser.add_argument(
        "--skip_prototype",
        action="store_true",
        default=config.get("skip_prototype", False),
        help="Skip Phase 2: Prototype evaluation with NC metrics",
    )
    parser.add_argument(
        "--skip_linear",
        action="store_true",
        default=config.get("skip_linear", False),
        help="Skip Phase 3: Standard linear probing (CE loss)",
    )
    parser.add_argument(
        "--skip_normalized_linear",
        action="store_true",
        default=config.get("skip_normalized_linear", False),
        help="Skip Phase 4: Normalized linear probing",
    )

    # Backward compatibility
    parser.add_argument(
        "--skip_supcon",
        action="store_true",
        help="(Deprecated) Same as --skip_contrastive",
    )

    parser.add_argument(
        "--supcon_ckpt",
        type=str,
        default=config.get("supcon_ckpt", ""),
        help="Path to pre-trained contrastive model (required if skipping Phase 1)",
    )

    # Metrics computation control
    parser.add_argument(
        "--compute_metrics_freq",
        type=int,
        default=config.get("compute_metrics_freq", 10),
        help="compute neural collapse metrics every N epochs",
    )
    parser.add_argument(
        "--always_compute_final_metrics",
        action="store_true",
        default=config.get("always_compute_final_metrics", True),
        help="always compute metrics on final epoch",
    )

    # Wandb parameters
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=config.get("wandb_project", "nc_by_design"),
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=config.get("wandb_entity", None),
        help="wandb entity",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=config.get("wandb_run_name", None),
        help="wandb run name",
    )

    args = parser.parse_args(remaining_argv)

    # Handle backward compatibility
    if args.skip_supcon:
        args.skip_contrastive = True
        print("[INFO] --skip_supcon is deprecated. Use --skip_contrastive instead.")

    # Auto-detect DDP settings from environment variables
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.rank = int(os.environ.get("RANK", 0))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Dataset-specific defaults and path handling
    if args.dataset == "imagenet100":
        if args.imagenet100_path is None:
            args.imagenet100_path = os.environ.get(
                "IMAGENET-100_PATH", "/path/to/imagenet100"
            )
        # ImageNet-100 specific defaults
        if "supcon_epochs" not in config and "--supcon_epochs" not in remaining_argv:
            args.supcon_epochs = 800
        if "linear_epochs" not in config and "--linear_epochs" not in remaining_argv:
            args.linear_epochs = 90
        if (
            "supcon_lr_decay_epochs" not in config
            and "--supcon_lr_decay_epochs" not in remaining_argv
        ):
            args.supcon_lr_decay_epochs = "600,700,750"
        if (
            "linear_lr_decay_epochs" not in config
            and "--linear_lr_decay_epochs" not in remaining_argv
        ):
            args.linear_lr_decay_epochs = "30,60,80"

    elif args.dataset == "imagenet1k":
        if args.imagenet1k_path is None:
            args.imagenet1k_path = os.environ.get(
                "IMAGENET_PATH", "/path/to/imagenet1k"
            )
        # ImageNet-1K specific defaults
        if "supcon_epochs" not in config and "--supcon_epochs" not in remaining_argv:
            args.supcon_epochs = 800
        if "linear_epochs" not in config and "--linear_epochs" not in remaining_argv:
            args.linear_epochs = 90
        if (
            "supcon_lr_decay_epochs" not in config
            and "--supcon_lr_decay_epochs" not in remaining_argv
        ):
            args.supcon_lr_decay_epochs = "600,700,750"
        if (
            "linear_lr_decay_epochs" not in config
            and "--linear_lr_decay_epochs" not in remaining_argv
        ):
            args.linear_lr_decay_epochs = "30,60,80"
        if "model" not in config and "--model" not in remaining_argv:
            args.model = "resnet50"  # ResNet50 for ImageNet-1K

        # Force cosine scheduler and warmup for ImageNet-1K
        args.cosine = True
        args.warm = True
    else:
        # CIFAR defaults
        if "supcon_epochs" not in config and "--supcon_epochs" not in remaining_argv:
            args.supcon_epochs = 1000
        if "linear_epochs" not in config and "--linear_epochs" not in remaining_argv:
            args.linear_epochs = 100
        if (
            "supcon_lr_decay_epochs" not in config
            and "--supcon_lr_decay_epochs" not in remaining_argv
        ):
            args.supcon_lr_decay_epochs = "700,800,900"
        if (
            "linear_lr_decay_epochs" not in config
            and "--linear_lr_decay_epochs" not in remaining_argv
        ):
            args.linear_lr_decay_epochs = "60,75,90"

    # Validate arguments
    if args.skip_contrastive and not args.supcon_ckpt:
        raise ValueError(
            "If skipping contrastive training (Phase 1), must provide --supcon_ckpt path"
        )

    # Set learning rates based on batch size
    total_batch_size = args.batch_size * args.world_size

    # Auto-set learning rates if not explicitly provided
    if (
        "supcon_learning_rate" not in config
        and "--supcon_learning_rate" not in remaining_argv
    ):
        args.supcon_learning_rate = get_learning_rate_for_batch_size(
            args.dataset, total_batch_size
        )

    # Set linear learning rate based on dataset
    if (
        "linear_learning_rate" not in config
        and "--linear_learning_rate" not in remaining_argv
    ):
        if args.dataset in ["cifar10", "cifar100"]:
            args.linear_learning_rate = 5.0  # Higher LR for CIFAR linear evaluation
        else:
            args.linear_learning_rate = get_learning_rate_for_batch_size(
                args.dataset, total_batch_size
            )

    # Enable warm-up for large batch sizes or ImageNet datasets
    if args.dataset in ["imagenet100", "imagenet1k"] or total_batch_size > 256:
        args.warm = True

    # Handle temperature parameter (support both --temp and --temperature)
    if hasattr(args, "temperature") and args.temperature != 0.07:
        final_temperature = args.temperature
    else:
        final_temperature = args.temp

    if args.rank == 0:
        print(f"[INFO] Four-Phase Training Configuration:")
        print(f"  Phase 1 (Contrastive): {'SKIP' if args.skip_contrastive else 'RUN'}")
        print(f"  Phase 2 (Prototype): {'SKIP' if args.skip_prototype else 'RUN'}")
        print(f"  Phase 3 (Linear CE): {'SKIP' if args.skip_linear else 'RUN'}")
        print(
            f"  Phase 4 (Normalized Linear): {'SKIP' if args.skip_normalized_linear else 'RUN'}"
        )
        print(
            f"[INFO] Set contrastive LR to {args.supcon_learning_rate} for total batch size {total_batch_size}"
        )
        print(
            f"[INFO] Set linear LR to {args.linear_learning_rate} for total batch size {total_batch_size}"
        )
        print(f"[INFO] Using temperature: {final_temperature}")
        print(f"[INFO] Cosine scheduling: {args.cosine}")
        print(f"[INFO] Warmup: {args.warm}")

    # Convert to TrainingConfig
    return TrainingConfig(
        dataset=args.dataset,
        data_folder=args.data_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        imagenet100_path=args.imagenet100_path,
        imagenet1k_path=args.imagenet1k_path,
        model=args.model,
        head_type=args.head_type,
        feat_dim=args.feat_dim,
        method=args.method,
        loss=args.loss,
        temperature=final_temperature,
        temp_pos_neg_ratio=args.temp_pos_neg_ratio,
        supcon_epochs=args.supcon_epochs,
        supcon_learning_rate=args.supcon_learning_rate,
        supcon_lr_decay_epochs=args.supcon_lr_decay_epochs,
        supcon_lr_decay_rate=args.supcon_lr_decay_rate,
        supcon_weight_decay=args.supcon_weight_decay,
        supcon_momentum=args.supcon_momentum,
        linear_epochs=args.linear_epochs,
        linear_learning_rate=args.linear_learning_rate,
        linear_lr_decay_epochs=args.linear_lr_decay_epochs,
        linear_lr_decay_rate=args.linear_lr_decay_rate,
        linear_weight_decay=args.linear_weight_decay,
        linear_momentum=args.linear_momentum,
        linear_normalized=args.linear_normalized,
        linear_loss=args.linear_loss,
        linear_temperature=args.linear_temperature,
        use_prototypes=args.use_prototypes,
        prototype_update_freq=args.prototype_update_freq,
        cosine=args.cosine,
        syncBN=args.syncBN,
        warm=args.warm,
        warm_epochs=5 if args.dataset == "imagenet1k" else 10,
        warmup_from=0.01,
        warmup_to=args.supcon_learning_rate,  # Will be adjusted in _adjust_learning_rate
        print_freq=args.print_freq,
        save_freq=args.save_freq,
        trial=args.trial,
        seed=args.seed,
        deterministic=args.deterministic,
        skip_contrastive=args.skip_contrastive,  # Phase 1
        skip_prototype=args.skip_prototype,  # Phase 2
        skip_linear=args.skip_linear,  # Phase 3
        skip_normalized_linear=args.skip_normalized_linear,  # Phase 4
        supcon_ckpt=args.supcon_ckpt,
        compute_metrics_freq=args.compute_metrics_freq,
        always_compute_final_metrics=args.always_compute_final_metrics,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        rank=args.rank,
        world_size=args.world_size,
        local_rank=args.local_rank,
    )


def main_worker(config):
    """Main worker function for each process"""
    # Validate local_rank before setting device
    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        if config.local_rank >= num_available_gpus:
            if config.rank == 0:
                print(
                    f"[ERROR] Local rank {config.local_rank} >= available GPUs {num_available_gpus}"
                )
            config.local_rank = config.local_rank % num_available_gpus

        torch.cuda.set_device(config.local_rank)
        if config.rank == 0:
            print(f"[Rank {config.rank}] Using GPU {config.local_rank}")
    else:
        if config.rank == 0:
            print(f"[Rank {config.rank}] No CUDA available, using CPU")
        config.local_rank = 0

    # Set random seed for this process
    set_seed(config.seed + config.rank, deterministic=config.deterministic)
    if config.rank == 0:
        print(f"[INFO] Random seed set to: {config.seed} + {config.rank} (rank)")
        print(f"[INFO] Deterministic mode: {'ON' if config.deterministic else 'OFF'}")

    # Print GPU information (only from rank 0)
    if config.rank == 0:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"[INFO] GPUs available: {num_gpus}")
            for i in range(num_gpus):
                print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"[INFO] CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
        else:
            print("[INFO] No GPUs available, using CPU")

    # Create and run trainer
    try:
        trainer = UnifiedSupConTrainer(config)
        trainer.run_training()
    except Exception as e:
        print(f"[Rank {config.rank}] Training failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """Main function"""
    print("Starting Unified SupCon Training...")

    # Apply memory optimizations
    optimize_memory_for_imagenet()

    # Parse arguments
    try:
        config = parse_arguments()
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)

    # Check if we're in a distributed environment
    if "WORLD_SIZE" in os.environ:
        # Already launched with torchrun or srun
        try:
            rank, world_size, local_rank = setup_ddp()
            config.rank = rank
            config.world_size = world_size
            config.local_rank = local_rank

            if rank == 0:
                print(
                    f"[INFO] DDP initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}"
                )

            main_worker(config)
        except Exception as e:
            print(f"DDP setup failed: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        # Single GPU or need to spawn processes
        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
            if world_size > 1:
                print("For multi-GPU training, please use torchrun:")
                print(
                    f"torchrun --nproc_per_node={world_size} unified_supcon.py [your_args]"
                )
                sys.exit(1)
            else:
                # Single GPU
                config.rank = 0
                config.world_size = 1
                config.local_rank = 0
                print("[INFO] Single GPU training")
                main_worker(config)
        else:
            # CPU training
            config.rank = 0
            config.world_size = 1
            config.local_rank = 0
            print("[INFO] CPU training")
            main_worker(config)

    # Clean up
    cleanup()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
