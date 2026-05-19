from __future__ import print_function

import argparse
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import losses
import util
import wandb
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from networks.resnet_big import SupCEResNet
from util import (
    AverageMeter,
    accuracy,
    adjust_learning_rate,
    save_model,
    set_optimizer,
    warmup_learning_rate,
)

try:
    pass
except ImportError:
    pass


def seed_worker(worker_id):
    """Worker function for setting random seeds in DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# class ClassBalancedBatchSampler:
#     """
#     Class-balanced batch sampler that tries to cover (almost) all samples
#     in the dataset once per epoch, while keeping:
#         - classes_per_batch distinct classes per batch
#         - samples_per_class samples per selected class

#     It:
#       1. Shuffles indices *within* each class every epoch.
#       2. Splits each class's indices into chunks of size samples_per_class
#          (last chunk padded by repeating indices if needed).
#       3. Builds batches by picking classes_per_batch DISTINCT classes and
#          taking one chunk from each.
#       4. Then shreds the global list of batches across ranks.

#     Some samples may be repeated (because of padding), but every sample
#     should be seen at least once per epoch, up to leftovers when we
#     cannot make a full batch out of remaining classes.
#     """

#     def __init__(self, dataset, classes_per_batch, samples_per_class,
#                  num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=True):
#         self.classes_per_batch = classes_per_batch
#         self.samples_per_class = samples_per_class
#         self.batch_size = classes_per_batch * samples_per_class  # per-rank batch size
#         self.shuffle = shuffle
#         self.seed = seed
#         self.epoch = 0
#         self.drop_last = drop_last

#         # Distributed settings
#         if num_replicas is None:
#             if dist.is_available() and dist.is_initialized():
#                 num_replicas = dist.get_world_size()
#             else:
#                 num_replicas = 1
#         if rank is None:
#             if dist.is_available() and dist.is_initialized():
#                 rank = dist.get_rank()
#             else:
#                 rank = 0

#         self.num_replicas = num_replicas
#         self.rank = rank

#         # Get labels from dataset
#         if hasattr(dataset, 'targets'):
#             labels = np.array(dataset.targets)
#         elif hasattr(dataset, 'labels'):
#             labels = np.array(dataset.labels)
#         elif hasattr(dataset, 'samples'):
#             labels = np.array([s[1] for s in dataset.samples])
#         else:
#             raise ValueError("Dataset must have 'targets', 'labels', or 'samples' attribute")

#         # Build class-to-indices mapping (force int keys)
#         self.class_indices = defaultdict(list)
#         for idx, label in enumerate(labels):
#             self.class_indices[int(label)].append(idx)

#         self.num_classes = len(self.class_indices)
#         self.classes = sorted(self.class_indices.keys())

#         if self.classes_per_batch > self.num_classes:
#             raise ValueError(
#                 f"classes_per_batch ({self.classes_per_batch}) cannot be greater "
#                 f"than num_classes ({self.num_classes})"
#             )

#         self.num_samples = len(labels)

#         # Rough estimate for length; will be refined on first __iter__.
#         # Approx: per class, ceil(n_c / M) chunks, then group chunks into batches of C.
#         total_chunks = 0
#         for c in self.classes:
#             n_c = len(self.class_indices[c])
#             total_chunks += int(math.ceil(n_c / float(self.samples_per_class)))

#         est_num_world_batches = total_chunks // self.classes_per_batch
#         if not drop_last and total_chunks % self.classes_per_batch != 0:
#             est_num_world_batches += 1

#         # Make this divisible across replicas in the estimate
#         if drop_last:
#             total_batches_after_drop = (est_num_world_batches // self.num_replicas) * self.num_replicas
#             self.num_batches_per_rank = total_batches_after_drop // self.num_replicas
#         else:
#             total_batches_after_pad = int(math.ceil(est_num_world_batches / self.num_replicas)) * self.num_replicas
#             self.num_batches_per_rank = total_batches_after_pad // self.num_replicas

#         if self.rank == 0:
#             print(f"\n[INFO] Class-balanced BatchSampler (v2) initialized:")
#             print(f"  - Classes per batch: {self.classes_per_batch}")
#             print(f"  - Samples per class: {self.samples_per_class}")
#             print(f"  - Per-rank batch size: {self.batch_size}")
#             print(f"  - Total classes: {self.num_classes}")
#             print(f"  - Total samples: {self.num_samples}")
#             print(f"  - Estimated world batches per epoch: {est_num_world_batches}")
#             print(f"  - Estimated batches per rank: {self.num_batches_per_rank}")
#             print(f"  - drop_last: {self.drop_last}")

#     def __iter__(self):
#         # RNG for this epoch
#         g = np.random.default_rng(self.seed + self.epoch)

#         # 1) Shuffle indices WITHIN each class and split into chunks of size M
#         class_to_chunks = {}
#         total_chunks = 0

#         for class_idx in self.classes:
#             indices = self.class_indices[class_idx].copy()
#             if self.shuffle:
#                 g.shuffle(indices)

#             chunks = []
#             for i in range(0, len(indices), self.samples_per_class):
#                 chunk = indices[i:i + self.samples_per_class]
#                 # Pad last chunk to exactly samples_per_class by repeating indices
#                 if len(chunk) < self.samples_per_class:
#                     needed = self.samples_per_class - len(chunk)
#                     if len(indices) > 0:
#                         if self.shuffle:
#                             extra = g.choice(indices, size=needed, replace=True).tolist()
#                         else:
#                             extra = [indices[j % len(indices)] for j in range(needed)]
#                         chunk = chunk + extra
#                 chunks.append(chunk)

#             class_to_chunks[class_idx] = chunks
#             total_chunks += len(chunks)

#         # 2) Build world-level batches: each batch = classes_per_batch distinct classes
#         batches = []
#         # We maintain a set/list of classes that still have chunks
#         available_classes = [c for c in self.classes if len(class_to_chunks[c]) > 0]

#         while True:
#             current_classes = [c for c in available_classes if len(class_to_chunks[c]) > 0]
#             if len(current_classes) < self.classes_per_batch:
#                 break  # cannot form another full C-class batch

#             if self.shuffle:
#                 chosen_classes = g.choice(
#                     current_classes, size=self.classes_per_batch, replace=False
#                 ).tolist()
#             else:
#                 chosen_classes = current_classes[:self.classes_per_batch]

#             batch = []
#             for c in chosen_classes:
#                 # Pop chunk from this class
#                 if self.shuffle:
#                     # random pop: take last
#                     chunk = class_to_chunks[c].pop()
#                 else:
#                     # deterministic: pop from front
#                     chunk = class_to_chunks[c].pop(0)
#                 batch.extend(chunk)

#             batches.append(batch)

#         # 3) Optionally drop or pad batches BEFORE sharding
#         if self.drop_last:
#             # Truncate to multiple of num_replicas
#             target_num_batches = (len(batches) // self.num_replicas) * self.num_replicas
#             batches = batches[:target_num_batches]
#         else:
#             # Pad by cycling existing batches until divisible
#             target_num_batches = int(math.ceil(len(batches) / self.num_replicas)) * self.num_replicas
#             original_num_batches = len(batches)
#             while len(batches) < target_num_batches and original_num_batches > 0:
#                 idx = len(batches) % original_num_batches
#                 batches.append(batches[idx])

#         # 4) Shuffle batches if needed
#         if self.shuffle:
#             g.shuffle(batches)

#         # 5) Shard across ranks
#         batches_for_rank = batches[self.rank::self.num_replicas]

#         # Update actual length based on what we just built
#         self.num_batches_per_rank = len(batches_for_rank)

#         for batch in batches_for_rank:
#             yield batch

#     def __len__(self):
#         # Will be updated after first __iter__; before that, return estimate
#         return self.num_batches_per_rank

#     def set_epoch(self, epoch):
#         self.epoch = epoch

## slow oti na nai
# class ClassBalancedBatchSampler:
#     """
#     Fast class-balanced sampler.

#     - Per GPU:
#         batch_size = classes_per_batch * samples_per_class
#     - Across GPUs:
#         total_slots = classes_per_batch * num_replicas class-slots.

#     Behavior per global step (super-batch):
#       * If total_slots <= num_classes:
#             choose total_slots DISTINCT classes (no overlap globally).
#       * If total_slots > num_classes:
#             use each available class once, then fill the remaining
#             slots by sampling classes WITH replacement.
#             (So you get "1 per class + extras".)

#     Chunks: each time a class appears in the class-slot list, we pop
#     one chunk of 'samples_per_class' indices from that class.
#     """

#     def __init__(self, dataset, classes_per_batch, samples_per_class,
#                  num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=True):
#         self.classes_per_batch = classes_per_batch
#         self.samples_per_class = samples_per_class
#         self.batch_size = classes_per_batch * samples_per_class  # per-rank
#         self.shuffle = shuffle
#         self.seed = seed
#         self.epoch = 0
#         self.drop_last = drop_last

#         # Distributed
#         if num_replicas is None:
#             if dist.is_available() and dist.is_initialized():
#                 num_replicas = dist.get_world_size()
#             else:
#                 num_replicas = 1
#         if rank is None:
#             if dist.is_available() and dist.is_initialized():
#                 rank = dist.get_rank()
#             else:
#                 rank = 0

#         self.num_replicas = num_replicas
#         self.rank = rank

#         # Labels
#         if hasattr(dataset, "targets"):
#             labels = np.array(dataset.targets)
#         elif hasattr(dataset, "labels"):
#             labels = np.array(dataset.labels)
#         elif hasattr(dataset, "samples"):
#             labels = np.array([s[1] for s in dataset.samples])
#         else:
#             raise ValueError("Dataset must have 'targets', 'labels', or 'samples'")

#         # Class -> indices
#         self.class_indices = defaultdict(list)
#         for idx, label in enumerate(labels):
#             self.class_indices[int(label)].append(idx)

#         self.num_classes = len(self.class_indices)
#         self.classes = sorted(self.class_indices.keys())
#         self.num_samples = len(labels)

#         # Total class slots per super-batch (all GPUs)
#         self.total_slots = self.classes_per_batch * self.num_replicas

#         # Rough estimate of length (like your original)
#         total_chunks = 0
#         for c in self.classes:
#             n_c = len(self.class_indices[c])
#             total_chunks += int(math.ceil(n_c / float(self.samples_per_class)))
#         est_num_superbatches = total_chunks // max(self.total_slots, 1)
#         self.num_batches_per_rank = est_num_superbatches

#         if self.rank == 0:
#             print(f"\n[INFO] Fast ClassBalancedBatchSampler initialized:")
#             print(f"  - Classes per batch (per GPU): {self.classes_per_batch}")
#             print(f"  - Samples per class: {self.samples_per_class}")
#             print(f"  - Per-rank batch size: {self.batch_size}")
#             print(f"  - Num classes: {self.num_classes}")
#             print(f"  - Num samples: {self.num_samples}")
#             print(f"  - World size: {self.num_replicas}")
#             print(f"  - Class slots per super-batch (world): {self.total_slots}")
#             print(f"  - Estimated batches per rank: {self.num_batches_per_rank}")
#             print(f"  - drop_last: {self.drop_last}")

#     def __iter__(self):
#         g = np.random.default_rng(self.seed + self.epoch)

#         # Build chunks per class once per epoch
#         class_to_chunks = {}
#         for c in self.classes:
#             idxs = self.class_indices[c].copy()
#             if self.shuffle:
#                 g.shuffle(idxs)

#             chunks = []
#             for i in range(0, len(idxs), self.samples_per_class):
#                 chunk = idxs[i:i + self.samples_per_class]
#                 if len(chunk) < self.samples_per_class:
#                     # pad from within the same class
#                     needed = self.samples_per_class - len(chunk)
#                     extra = g.choice(idxs, size=needed, replace=True).tolist()
#                     chunk = chunk + extra
#                 chunks.append(chunk)
#             class_to_chunks[c] = chunks

#         batches_for_rank = []

#         while True:
#             # Classes that still have at least one chunk
#             available = [c for c in self.classes if len(class_to_chunks[c]) > 0]
#             if len(available) == 0:
#                 break

#             # If we can't even fill one per-rank batch, stop (or partial if not drop_last)
#             if len(available) < self.classes_per_batch and self.drop_last:
#                 break

#             # Build list of class IDs for this super-batch (world)
#             chosen_classes = self._build_superbatch_classes(
#                 available, class_to_chunks, self.total_slots, g
#             )
#             if len(chosen_classes) == 0:
#                 break

#             # Slice for this rank
#             start = self.rank * self.classes_per_batch
#             end = start + self.classes_per_batch
#             rank_classes = chosen_classes[start:end]

#             batch = []
#             for c in rank_classes:
#                 # Class might have run out of chunks if heavily reused; skip if so
#                 if len(class_to_chunks[c]) == 0:
#                     continue
#                 # Pop one chunk (O(1))
#                 chunk = class_to_chunks[c].pop()
#                 batch.extend(chunk)

#             if len(batch) == 0:
#                 break

#             batches_for_rank.append(batch)

#         self.num_batches_per_rank = len(batches_for_rank)

#         for b in batches_for_rank:
#             yield b

#     def _build_superbatch_classes(self, available_classes, class_to_chunks, total_slots, g):
#         """
#         Returns a list of class IDs (length <= total_slots).

#         If total_slots <= len(available_classes):
#             sample total_slots distinct classes.

#         Else (your ImageNet 2048 > 1000 case):
#             - take each available_class once (if it still has chunks),
#             - then fill remaining slots by sampling from those classes
#               WITH replacement.
#         """
#         pool = [c for c in available_classes if len(class_to_chunks[c]) > 0]
#         if len(pool) == 0:
#             return []

#         if self.shuffle:
#             g.shuffle(pool)

#         # First: distinct classes
#         chosen = []
#         for c in pool:
#             if len(chosen) >= total_slots:
#                 break
#             chosen.append(c)

#         if len(chosen) >= total_slots:
#             # Trim and shuffle
#             chosen = chosen[:total_slots]
#             if self.shuffle:
#                 g.shuffle(chosen)
#             return chosen

#         # Need more slots -> sample with replacement
#         slots_left = total_slots - len(chosen)
#         extra = g.choice(pool, size=slots_left, replace=True).tolist()
#         chosen.extend(extra)
#         if self.shuffle:
#             g.shuffle(chosen)
#         return chosen

#     def __len__(self):
#         return self.num_batches_per_rank

#     def set_epoch(self, epoch):
#        self.epoch = epoch


class ClassBalancedBatchSampler:
    """
    Class-balanced DDP sampler with:

      * World-aware batches (concatenate all ranks).
      * Every *global* batch contains every class at least once
        (assuming total_slots >= num_classes).
      * No sample is ever dropped: all dataset indices are used
        at least once per epoch (some may be repeated for balance).

    IMPORTANT:
      This is achieved by **oversampling** classes when necessary.
      For highly imbalanced datasets and very strict constraints,
      it may still be mathematically impossible to avoid repeats.
    """

    def __init__(
        self,
        dataset,
        classes_per_batch,
        samples_per_class,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=True,
    ):

        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.batch_size = classes_per_batch * samples_per_class  # per rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last  # kept for compatibility, but we don't drop data

        # DDP boilerplate (unchanged)
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        self.num_replicas = num_replicas
        self.rank = rank

        # Get labels
        if hasattr(dataset, "targets"):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, "labels"):
            labels = np.array(dataset.labels)
        elif hasattr(dataset, "samples"):
            labels = np.array([s[1] for s in dataset.samples])
        else:
            raise ValueError("Dataset must have 'targets', 'labels', or 'samples'")

        # class -> indices
        self.class_indices = defaultdict(list)
        for idx, lab in enumerate(labels):
            self.class_indices[int(lab)].append(idx)

        self.classes = sorted(self.class_indices.keys())
        self.num_classes = len(self.classes)
        self.num_samples = len(labels)

        self.total_slots = self.classes_per_batch * self.num_replicas

        if self.total_slots < self.num_classes:
            raise ValueError(
                f"Need at least one slot per class globally: "
                f"classes_per_batch * num_replicas = {self.total_slots} "
                f"< num_classes ({self.num_classes})"
            )

        # Target: same number of steps as a plain sampler
        world_batch_size = self.batch_size * self.num_replicas
        self.num_global_steps = int(
            math.ceil(self.num_samples / float(world_batch_size))
        )
        self.num_batches_per_rank = self.num_global_steps

        if self.rank == 0:
            print(
                f"\n[INFO] World-aware ClassBalancedBatchSampler (no-drop, oversampling) initialized:"
            )
            print(f"  - classes_per_batch (per GPU): {self.classes_per_batch}")
            print(f"  - samples_per_class: {self.samples_per_class}")
            print(f"  - per-rank batch size: {self.batch_size}")
            print(f"  - num_classes: {self.num_classes}")
            print(f"  - num_samples: {self.num_samples}")
            print(f"  - world_size: {self.num_replicas}")
            print(f"  - total slots per global step: {self.total_slots}")
            print(f"  - target global steps/epoch: {self.num_global_steps}")
            print(f"  - NOTE: minority classes will be oversampled if necessary.\n")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches_per_rank

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)

        # For each class, maintain a cyclic shuffled order of indices.
        class_order = {}
        class_pos = {}
        for c in self.classes:
            idxs = self.class_indices[c].copy()
            if self.shuffle:
                g.shuffle(idxs)
            class_order[c] = idxs
            class_pos[c] = 0

        # Track how many times each index has been used, to ensure
        # we cover all instances at least once.
        used_counts = np.zeros(self.num_samples, dtype=np.int32)

        def get_chunk_for_class(c):
            """
            Return samples_per_class indices for class c, making sure
            we iterate over all its indices before heavily repeating
            any of them.
            """
            idxs = class_order[c]
            pos = class_pos[c]
            chosen = []

            for _ in range(self.samples_per_class):
                # If we exhausted the current permutation, reshuffle and restart
                if pos >= len(idxs):
                    if self.shuffle:
                        g.shuffle(idxs)
                    pos = 0
                idx = idxs[pos]
                pos += 1
                chosen.append(idx)
                used_counts[idx] += 1

            class_pos[c] = pos
            return chosen

        batches_for_rank = []

        for step in range(self.num_global_steps):
            # Build world-class slots with full coverage
            chosen_classes = self._build_superbatch_classes_full_coverage_no_capacity(
                self.total_slots, g, used_counts
            )

            start_slot = self.rank * self.classes_per_batch
            end_slot = start_slot + self.classes_per_batch

            batch = []
            for slot_idx, c in enumerate(chosen_classes):
                chunk = get_chunk_for_class(c)
                if start_slot <= slot_idx < end_slot:
                    batch.extend(chunk)

            batches_for_rank.append(batch)

        self.num_batches_per_rank = len(batches_for_rank)

        for b in batches_for_rank:
            yield b

    def _build_superbatch_classes_full_coverage_no_capacity(
        self, total_slots, g, used_counts
    ):
        """
        Build a list of class IDs of length total_slots such that:

          - Every class appears at least once.
          - Extra slots are biased toward classes whose samples have
            been used less often so far (to help cover all instances).

        We ignore "capacity" because we allow oversampling.
        """
        # Start with 1 slot per class for coverage
        base = self.classes.copy()
        if self.shuffle:
            g.shuffle(base)

        chosen = list(base)

        if total_slots == self.num_classes:
            return chosen

        slots_left = total_slots - self.num_classes

        # Compute an approximate 'need' score per class:
        # classes whose indices have been used less get more extra slots.
        avg_used_per_class = {}
        for c in self.classes:
            idxs = self.class_indices[c]
            if len(idxs) == 0:
                avg_used_per_class[c] = 0.0
            else:
                avg_used_per_class[c] = float(used_counts[idxs].mean())

        # We want to give extra slots to classes with *smaller* avg_used.
        # Turn that into a probability distribution.
        needs = np.array(
            [1.0 / (1.0 + avg_used_per_class[c]) for c in self.classes],
            dtype=np.float64,
        )
        probs = needs / needs.sum()

        class_array = np.array(self.classes)

        while slots_left > 0:
            if self.shuffle:
                # Draw according to "need"
                c = g.choice(class_array, p=probs)
            else:
                c = self.classes[0]
            chosen.append(int(c))
            slots_left -= 1

        if self.shuffle:
            g.shuffle(chosen)

        return chosen


def get_network_interface():
    """Detect the correct network interface for NCCL on different environments."""
    import re
    import subprocess

    # Check if we're on a Slurm cluster
    is_slurm = "SLURM_JOB_ID" in os.environ

    try:
        # Get default route interface
        result = subprocess.run(
            ["ip", "route", "get", "8.8.8.8"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Extract interface name from output like "dev eth0"
            match = re.search(r"dev (\w+)", result.stdout)
            if match:
                interface = match.group(1)
                print(f"[INFO] Detected network interface: {interface}")
                return interface
    except Exception as e:
        print(f"[WARNING] Failed to detect network interface: {e}")

    # Different fallback options for different environments
    if is_slurm:
        # Common HPC interfaces
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
        # AWS and other cloud interfaces
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

    # CRITICAL FIX: For Slurm, adjust local_rank based on available GPUs
    if world_size > 1:
        # Check environment type
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
            # Get the number of GPUs available on this node
            if torch.cuda.is_available():
                num_gpus_per_node = torch.cuda.device_count()
                # Map local_rank to available GPU devices
                local_rank = local_rank % num_gpus_per_node
                if rank == 0:
                    print(f"[INFO] Slurm detected: {num_gpus_per_node} GPUs per node")
                    print(f"[INFO] Adjusted local_rank mapping for Slurm")
            else:
                print("[ERROR] No CUDA devices available")
                return rank, world_size, 0

        # Base NCCL settings that work everywhere
        os.environ["NCCL_TREE_THRESHOLD"] = "0"  # Disable tree algorithms

        if is_aws:
            # AWS-specific settings
            os.environ["NCCL_IB_DISABLE"] = "1"  # No InfiniBand on AWS
            os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P for stability
            # Suppress AWS OFI warnings (these are harmless)
            os.environ["NCCL_NET"] = "Socket"  # Force socket backend
        elif is_slurm:
            # HPC/Slurm settings - might have InfiniBand
            if rank == 0:
                print("[INFO] HPC environment detected, checking for InfiniBand...")
            # Check if InfiniBand is available
            try:
                import subprocess

                result = subprocess.run(
                    ["ibstat"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and "Active" in result.stdout:
                    if rank == 0:
                        print("[INFO] InfiniBand detected and active")
                    # Don't disable IB if it's available
                else:
                    if rank == 0:
                        print("[INFO] InfiniBand not available, using Ethernet")
                    os.environ["NCCL_IB_DISABLE"] = "1"
            except Exception:
                if rank == 0:
                    print("[INFO] Could not detect InfiniBand, disabling")
                os.environ["NCCL_IB_DISABLE"] = "1"
        else:
            # Generic cloud/other settings
            os.environ["NCCL_IB_DISABLE"] = "1"
            os.environ["NCCL_P2P_DISABLE"] = "1"

        # Auto-detect network interface
        network_interface = get_network_interface()
        os.environ["NCCL_SOCKET_IFNAME"] = network_interface

        # Reduce NCCL verbosity to suppress harmless warnings
        os.environ["NCCL_DEBUG"] = "WARN"

        if rank == 0:
            print(f"[INFO] Initializing NCCL with interface: {network_interface}")
            print(
                f"[INFO] Rank {rank}, World Size {world_size}, Local Rank {local_rank}"
            )

        try:
            dist.init_process_group(backend="nccl", init_method="env://")
            if rank == 0:
                print(f"[INFO] NCCL process group initialized successfully")
        except Exception as e:
            print(f"[Rank {rank}] NCCL initialization failed: {e}")
            print(f"[Rank {rank}] Trying with minimal NCCL settings...")

            # Fallback with minimal settings
            for key in ["NCCL_SOCKET_IFNAME", "NCCL_P2P_DISABLE", "NCCL_NET"]:
                if key in os.environ:
                    del os.environ[key]

            # Try again with just the essential settings
            os.environ["NCCL_TREE_THRESHOLD"] = "0"
            if not is_slurm:  # Only disable IB on non-HPC systems
                os.environ["NCCL_IB_DISABLE"] = "1"

            dist.init_process_group(backend="nccl", init_method="env://")
            if rank == 0:
                print(f"[INFO] NCCL initialized with fallback settings")

    return rank, world_size, local_rank


def cleanup():
    """Clean up the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed, deterministic=False):
    """Set seed for reproducibility.

    Args:
        seed: Random seed
        deterministic: If True, force deterministic behavior (slower but more reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # For atomic operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For transforms
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Use benchmark mode for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def optimize_memory_for_imagenet():
    """Apply memory optimizations specifically for ImageNet-1K"""
    # Enable memory pool expansion to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Reduce TensorFlow memory if using mixed frameworks
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Enable CUDA memory caching optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def debug_slurm_environment():
    """Debug function to print Slurm environment variables"""
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE",
        "SLURM_NODEID",
        "SLURM_NNODES",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS_PER_NODE",
        "SLURM_GPUS_PER_TASK",
        "CUDA_VISIBLE_DEVICES",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
    ]

    print("\n=== SLURM ENVIRONMENT DEBUG ===")
    for var in slurm_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"{var}: {value}")

    if torch.cuda.is_available():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(
            f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}"
        )
    print("================================\n")


def parse_option():
    # Stage 1: Minimal parser to capture --config
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, help="Path to config YAML file")
    pre_parser.add_argument(
        "--wandb_project",
        type=str,
        default="nc_by_design",
        help="Weights & Biases project name",
    )
    # Set via --wandb_entity or env var
    pre_parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Weights & Biases entity"
    )
    pre_parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Weights & Biases run name"
    )
    pre_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    pre_parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic behavior (slower but more reproducible across hardware)",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()

    # Load YAML config if specified
    config = {}
    if pre_args.config:
        with open(pre_args.config, "r") as f:
            config = yaml.safe_load(f)

    # Stage 2: Full parser with defaults from config
    parser = argparse.ArgumentParser(
        "argument for training",
        parents=[pre_parser],  # includes --config etc
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Basic training options
    parser.add_argument("--print_freq", type=int, default=config.get("print_freq", 10))
    parser.add_argument("--save_freq", type=int, default=config.get("save_freq", 50))
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 256))
    parser.add_argument(
        "--num_workers", type=int, default=config.get("num_workers", 16)
    )
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 500))

    # Class-balanced batch sampling
    parser.add_argument(
        "--class_balanced",
        action="store_true",
        default=config.get("class_balanced", False),
        help="Use class-balanced batch sampling",
    )
    parser.add_argument(
        "--classes_per_batch",
        type=int,
        default=config.get("classes_per_batch", None),
        help="Number of classes per batch (C) for class-balanced sampling",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=config.get("samples_per_class", None),
        help="Number of samples per class (M) for class-balanced sampling",
    )

    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default=config.get("lr_decay_epochs", "350,400,450"),
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=config.get("lr_decay_rate", 0.1)
    )
    parser.add_argument(
        "--weight_decay", type=float, default=config.get("weight_decay", 1e-4)
    )
    parser.add_argument("--momentum", type=float, default=config.get("momentum", 0.9))

    parser.add_argument(
        "--normalize", action="store_true", default=config.get("normalize", False)
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=config.get("loss", "CE"),
        choices=["CE", "NormFace", "NTCE", "NONL"],
    )
    parser.add_argument(
        "--temperature", type=float, default=config.get("temperature", 0.2)
    )
    parser.add_argument(
        "--symmetric", action="store_true", default=config.get("symmetric", False)
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=config.get("bidirectional", False),
    )

    parser.add_argument("--model", type=str, default=config.get("model", "resnet18"))
    parser.add_argument(
        "--etf_classifier",
        action="store_true",
        default=config.get("etf_classifier", False),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=config.get("dataset", "cifar10"),
        choices=["cifar10", "cifar100", "imagenet100", "imagenet1k"],
    )

    # ImageNet paths
    parser.add_argument(
        "--imagenet1k_path",
        type=str,
        default=config.get("imagenet1k_path", None),
        help="Path to ImageNet-1K dataset directory (train/ and val/ subdirs)",
    )
    parser.add_argument(
        "--imagenet100_path",
        type=str,
        default=config.get("imagenet100_path", None),
        help="Path to ImageNet-100 dataset directory (train/ and val.X/ subdirs)",
    )

    parser.add_argument(
        "--cosine", action="store_true", default=config.get("cosine", False)
    )
    parser.add_argument(
        "--syncBN", action="store_true", default=config.get("syncBN", False)
    )
    parser.add_argument(
        "--warm", action="store_true", default=config.get("warm", False)
    )
    parser.add_argument("--trial", type=str, default=config.get("trial", "0"))

    # Imbalanced CIFAR options
    parser.add_argument(
        "--imb_ratio",
        type=float,
        default=config.get("imb_ratio", None),
        help="Imbalance ratio n_max/n_min for CIFAR (e.g. 100 -> imb_factor=0.01)",
    )
    parser.add_argument(
        "--imb_factor",
        type=float,
        default=config.get("imb_factor", None),
        help="Direct LDAM-style imbalance factor (n_min/n_max). "
        "If both imb_ratio and imb_factor are set, imb_factor takes precedence.",
    )
    parser.add_argument(
        "--imb_type",
        type=str,
        default=config.get("imb_type", "exp"),
        choices=["exp", "step", "none"],
        help="Type of CIFAR imbalance (matches LDAM-DRW: exp or step).",
    )

    # Final parse with CLI override
    opt = parser.parse_args(remaining_argv)

    # Auto-detect DDP settings from environment variables
    opt.world_size = int(os.environ.get("WORLD_SIZE", 1))
    opt.rank = int(os.environ.get("RANK", 0))
    opt.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ------------------------------------------------------------------
    # Derive final imbalance settings (CIFAR only)
    # ------------------------------------------------------------------
    if opt.imb_factor is not None:
        if opt.rank == 0:
            print(
                f"[INFO] Using explicit imb_factor={opt.imb_factor} "
                f"(n_min/n_max ≈ {opt.imb_factor:.4f}) for CIFAR."
            )
    elif opt.imb_ratio is not None and opt.imb_ratio > 1:
        opt.imb_factor = 1.0 / opt.imb_ratio
        if opt.rank == 0:
            print(
                f"[INFO] Using imb_ratio={opt.imb_ratio} (n_max/n_min), "
                f"so imb_factor={opt.imb_factor:.4f}."
            )
    else:
        opt.imb_factor = None
        if opt.rank == 0 and opt.dataset in ["cifar10", "cifar100"]:
            print("[INFO] No CIFAR imbalance requested (balanced dataset).")

    # ------------------------------------------------------------------
    # Dataset-specific defaults
    # ------------------------------------------------------------------
    if opt.dataset == "imagenet100":
        if "batch_size" not in config and "--batch_size" not in remaining_argv:
            opt.batch_size = 256
        if "epochs" not in config and "--epochs" not in remaining_argv:
            opt.epochs = 90
        if (
            "lr_decay_epochs" not in config
            and "--lr_decay_epochs" not in remaining_argv
        ):
            opt.lr_decay_epochs = "30,60,80"

        if opt.imagenet100_path is None:
            opt.imagenet100_path = os.environ.get(
                "IMAGENET-100_PATH", "/path/to/imagenet100"
            )

    elif opt.dataset == "imagenet1k":
        if "batch_size" not in config and "--batch_size" not in remaining_argv:
            opt.batch_size = 256
        if "epochs" not in config and "--epochs" not in remaining_argv:
            opt.epochs = 90
        if (
            "lr_decay_epochs" not in config
            and "--lr_decay_epochs" not in remaining_argv
        ):
            opt.lr_decay_epochs = "30,60,80"
        if "model" not in config and "--model" not in remaining_argv:
            opt.model = "resnet50"

        # Force cosine scheduler and warmup for ImageNet-1K baseline
        opt.cosine = True
        opt.warm = True

        if opt.imagenet1k_path is None:
            opt.imagenet1k_path = os.environ.get("IMAGENET_PATH", "/path/to/imagenet1k")
    else:
        # CIFAR defaults
        if (
            "lr_decay_epochs" not in config
            and "--lr_decay_epochs" not in remaining_argv
        ):
            opt.lr_decay_epochs = "350,400,450"

    # ------------------------------------------------------------------
    # Learning rate scaling based on total batch size
    # ------------------------------------------------------------------
    if opt.class_balanced:
        effective_bsz_per_rank = opt.classes_per_batch * opt.samples_per_class
        total_batch_size = effective_bsz_per_rank * opt.world_size
    else:
        total_batch_size = opt.batch_size * opt.world_size

    if opt.dataset == "imagenet100":
        batch_lr_map = {
            32: 0.0125,
            64: 0.025,
            128: 0.05,
            256: 0.1,
            512: 0.2,
            1024: 0.4,
            2048: 0.8,
        }
    elif opt.dataset == "imagenet1k":
        batch_lr_map = {
            128: 0.05,
            256: 0.1,
            512: 0.2,
            1024: 0.4,
            2048: 0.8,
            4096: 1.6,
            8192: 3.2,
        }
    else:
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
        opt.learning_rate = batch_lr_map[total_batch_size]
    else:
        base_lr = 0.1 if opt.dataset in ["imagenet100", "imagenet1k"] else 0.2
        opt.learning_rate = base_lr * (total_batch_size / 256)
        if opt.rank == 0:
            print(
                f"[INFO] Using linear scaling for total batch size {total_batch_size}"
            )

    if opt.rank == 0:
        if opt.class_balanced:
            per_gpu_bsz = effective_bsz_per_rank
        else:
            per_gpu_bsz = opt.batch_size
        print(
            f"[INFO] Set learning rate to {opt.learning_rate} based on total batch size {total_batch_size} "
            f"(per GPU: {per_gpu_bsz}{' (class-balanced)' if opt.class_balanced else ''})"
        )

    # ------------------------------------------------------------------
    # Paths, model_name, dataset info
    # ------------------------------------------------------------------
    opt.data_folder = "./datasets/"
    opt.model_path = "./save/SupCon/{}_models".format(opt.dataset)
    opt.tb_path = "./save/SupCon/{}_tensorboard".format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = [int(it) for it in iterations]

    opt.model_name = "{}_{}_{}_lr_{}_decay_{}_bsz_{}_seed_{}_trial_{}".format(
        opt.loss,
        opt.dataset,
        opt.model,
        opt.learning_rate,
        opt.weight_decay,
        total_batch_size,
        opt.seed,
        opt.trial,
    )

    if opt.cosine:
        opt.model_name += "_cosine"

    # n_cls
    if opt.dataset == "cifar10":
        opt.n_cls = 10
    elif opt.dataset in ["cifar100", "imagenet100"]:
        opt.n_cls = 100
    else:  # imagenet1k
        opt.n_cls = 1000

    # ------------------------------------------------------------------
    # ViT-specific training adjustments (SGD-compatible, but can also use AdamW via set_optimizer)
    # ------------------------------------------------------------------
    if "vit" in opt.model.lower():
        if opt.rank == 0:
            print(f"\n[INFO] Detected ViT model - applying ViT-specific settings")

        is_vit_large = ("large" in opt.model.lower()) or ("vit_l" in opt.model.lower())

        # Recompute total batch size (in case of class-balanced)
        if opt.class_balanced:
            effective_bsz_per_rank = opt.classes_per_batch * opt.samples_per_class
            total_batch_size = effective_bsz_per_rank * opt.world_size
        else:
            total_batch_size = opt.batch_size * opt.world_size

        if is_vit_large:
            # ViT-Large
            if opt.dataset == "imagenet100":
                vit_batch_lr_map = {
                    256: 0.03,
                    512: 0.06,
                    1024: 0.12,
                    2048: 0.24,
                }
            elif opt.dataset == "imagenet1k":
                vit_batch_lr_map = {
                    256: 0.03,
                    512: 0.06,
                    1024: 0.12,
                    2048: 0.24,
                }
            else:  # CIFAR
                vit_batch_lr_map = {
                    256: 0.05,
                    512: 0.1,
                    1024: 0.2,
                }
            base_lr = 0.03
        else:
            # ViT-Base
            if opt.dataset == "imagenet100":
                vit_batch_lr_map = {
                    256: 0.05,
                    512: 0.1,
                    1024: 0.2,
                    2048: 0.4,
                }
            elif opt.dataset == "imagenet1k":
                vit_batch_lr_map = {
                    256: 0.05,
                    512: 0.1,
                    1024: 0.2,
                    2048: 0.4,
                }
            else:  # CIFAR
                vit_batch_lr_map = {
                    256: 0.1,
                    512: 0.2,
                    1024: 0.4,
                }
            base_lr = 0.05

        if total_batch_size in vit_batch_lr_map:
            opt.learning_rate = vit_batch_lr_map[total_batch_size]
        else:
            opt.learning_rate = base_lr * (total_batch_size / 256)

        # ViT warmup & cosine
        opt.cosine = True
        opt.warm = True

        if is_vit_large:
            if opt.dataset == "imagenet100":
                opt.warm_epochs = 20
            elif opt.dataset == "imagenet1k":
                opt.warm_epochs = 30
            else:
                opt.warm_epochs = 15
        else:
            if opt.dataset == "imagenet100":
                opt.warm_epochs = 15
            elif opt.dataset == "imagenet1k":
                opt.warm_epochs = 20
            else:
                opt.warm_epochs = 10

        # Start ViT from 0 (or tiny) lr
        opt.warmup_from = 0.0

        if opt.rank == 0:
            model_size = "Large" if is_vit_large else "Base"
            print(
                f"[INFO] ViT-{model_size} specific settings applied (SGD-compatible):"
            )
            print(f"  - Learning rate: {opt.learning_rate}")
            print(f"  - Warmup epochs: {opt.warm_epochs}")
            print(f"  - Weight decay: {opt.weight_decay}")
            print(f"  - Warmup from: {opt.warmup_from}")

    # ------------------------------------------------------------------
    # Warmup settings (FINAL, after ViT overrides)
    # ------------------------------------------------------------------
    # Auto-enable warmup for big batch ImageNet if not explicitly disabled
    if opt.dataset in ["imagenet100", "imagenet1k"] or total_batch_size > 256:
        opt.warm = True or opt.warm  # keep True if already True

    if opt.warm:
        opt.model_name += "_warm"

        # If ViT already set warmup_from / warm_epochs, keep them
        if not hasattr(opt, "warmup_from"):
            opt.warmup_from = 0.01
        if not hasattr(opt, "warm_epochs"):
            opt.warm_epochs = 5 if opt.dataset == "imagenet1k" else 10

        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate**3)
            opt.warmup_to = (
                eta_min
                + (opt.learning_rate - eta_min)
                * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs))
                / 2
            )
        else:
            opt.warmup_to = opt.learning_rate

    # ------------------------------------------------------------------
    # Imbalanced CIFAR tag in model_name
    # ------------------------------------------------------------------
    if opt.dataset in ["cifar10", "cifar100"] and opt.imb_factor is not None:
        opt.model_name += f"_imb_{opt.imb_type}_f_{opt.imb_factor}"

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if opt.rank == 0:
        os.makedirs(opt.tb_folder, exist_ok=True)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if opt.rank == 0:
        os.makedirs(opt.save_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # Class-balanced sanity checks
    # ------------------------------------------------------------------
    if opt.class_balanced:
        if opt.classes_per_batch is None or opt.samples_per_class is None:
            raise ValueError(
                "When using --class_balanced, both --classes_per_batch and --samples_per_class must be specified"
            )

        if opt.classes_per_batch > opt.n_cls:
            raise ValueError(
                f"--classes_per_batch ({opt.classes_per_batch}) cannot exceed number of classes ({opt.n_cls})"
            )

        effective_bsz_per_rank = opt.classes_per_batch * opt.samples_per_class
        if opt.rank == 0:
            print(f"\n[INFO] Class-balanced sampling enabled:")
            print(f"  - Classes per batch (C): {opt.classes_per_batch}")
            print(f"  - Samples per class (M): {opt.samples_per_class}")
            print(f"  - Effective per-rank batch size: {effective_bsz_per_rank}")
            print(
                f"  - Effective world batch size: {effective_bsz_per_rank * opt.world_size}"
            )
            print(
                f"  - Note: --batch_size ({opt.batch_size}) is ignored with --class_balanced\n"
            )

    return opt


def set_loader(opt):
    if opt.dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif opt.dataset == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:  # imagenet100 or imagenet1k
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.dataset in ["cifar10", "cifar100"]:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:  # imagenet100 or imagenet1k
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

    if opt.dataset == "cifar10":
        # ---------------- CIFAR-10 (balanced or imbalanced) ----------------
        if opt.imb_factor is not None and opt.imb_type != "none":
            # Long-tailed CIFAR-10, LDAM-DRW style
            train_dataset = IMBALANCECIFAR10(
                root=opt.data_folder,
                imb_type=opt.imb_type,  # usually 'exp'
                imb_factor=opt.imb_factor,  # e.g. 0.01 for ratio 100
                rand_number=opt.seed,  # seed for which samples are kept
                train=True,
                transform=train_transform,
                download=True,
            )
            if opt.rank == 0:
                ratio = 1.0 / opt.imb_factor
                print(
                    f"[INFO] Using IMBALANCECIFAR10 with imb_type={opt.imb_type}, "
                    f"imb_factor={opt.imb_factor:.4f} (≈ ratio {ratio:.1f})."
                )
        else:
            # Standard balanced CIFAR-10
            train_dataset = datasets.CIFAR10(
                root=opt.data_folder,
                train=True,
                transform=train_transform,
                download=True,
            )

        # Validation set always stays balanced
        val_dataset = datasets.CIFAR10(
            root=opt.data_folder,
            train=False,
            transform=val_transform,
        )

    elif opt.dataset == "cifar100":
        # ---------------- CIFAR-100 (balanced or imbalanced) ----------------
        if opt.imb_factor is not None and opt.imb_type != "none":
            # Long-tailed CIFAR-100, LDAM-DRW style
            train_dataset = IMBALANCECIFAR100(
                root=opt.data_folder,
                imb_type=opt.imb_type,
                imb_factor=opt.imb_factor,
                rand_number=opt.seed,
                train=True,
                transform=train_transform,
                download=True,
            )
            if opt.rank == 0:
                ratio = 1.0 / opt.imb_factor
                print(
                    f"[INFO] Using IMBALANCECIFAR100 with imb_type={opt.imb_type}, "
                    f"imb_factor={opt.imb_factor:.4f} (≈ ratio {ratio:.1f})."
                )
        else:
            # Standard balanced CIFAR-100
            train_dataset = datasets.CIFAR100(
                root=opt.data_folder,
                train=True,
                transform=train_transform,
                download=True,
            )

        # Validation set always stays balanced
        val_dataset = datasets.CIFAR100(
            root=opt.data_folder,
            train=False,
            transform=val_transform,
        )
    elif opt.dataset == "imagenet100":
        traindir = os.path.join(opt.imagenet100_path, "train")
        valdir = os.path.join(opt.imagenet100_path, "val.X")

        if not os.path.exists(traindir):
            raise RuntimeError(f"ImageNet-100 train directory not found: {traindir}")
        if not os.path.exists(valdir):
            raise RuntimeError(f"ImageNet-100 val directory not found: {valdir}")

        train_dataset = datasets.ImageFolder(traindir, transform=train_transform)
        val_dataset = datasets.ImageFolder(valdir, transform=val_transform)

        if opt.rank == 0:
            print(
                f"[INFO] Loaded ImageNet-100 with {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
            )
    else:  # imagenet1k
        traindir = os.path.join(opt.imagenet1k_path, "train")
        valdir = os.path.join(opt.imagenet1k_path, "val")

        if not os.path.exists(traindir):
            raise RuntimeError(f"ImageNet-1K train directory not found: {traindir}")
        if not os.path.exists(valdir):
            raise RuntimeError(f"ImageNet-1K val directory not found: {valdir}")

        train_dataset = datasets.ImageFolder(traindir, transform=train_transform)
        val_dataset = datasets.ImageFolder(valdir, transform=val_transform)

        if opt.rank == 0:
            print(
                f"[INFO] Loaded ImageNet-1K with {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
            )

    # CRITICAL: Use different seeds for each process to avoid identical augmentations
    g = torch.Generator()
    g.manual_seed(opt.seed + opt.rank * 10000)  # Large separation between ranks

    # Keep original resource usage
    num_workers = opt.num_workers
    if opt.dataset in ["cifar10", "cifar100"]:
        val_batch_size = 256
    elif opt.dataset == "imagenet100":
        val_batch_size = 100
    else:  # imagenet1k
        val_batch_size = 128

    # Create samplers/batch_samplers based on class_balanced flag
    if opt.class_balanced:
        # Validate required parameters
        if opt.classes_per_batch is None or opt.samples_per_class is None:
            raise ValueError(
                "--classes_per_batch and --samples_per_class must be specified when using --class_balanced"
            )

        # Create class-balanced batch sampler for training
        train_batch_sampler = ClassBalancedBatchSampler(
            train_dataset,
            classes_per_batch=opt.classes_per_batch,
            samples_per_class=opt.samples_per_class,
            num_replicas=opt.world_size,
            rank=opt.rank,
            shuffle=True,
            seed=opt.seed,
            drop_last=True,
        )

        # Use regular sampler for validation
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=opt.world_size, rank=opt.rank, shuffle=False
        )

        # Create DataLoader with batch_sampler for training
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,  # Use batch_sampler instead of sampler+batch_size
            num_workers=num_workers,
            pin_memory=False,
            generator=g,
            worker_init_fn=seed_worker,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Return batch_sampler for set_epoch calls
        train_sampler = train_batch_sampler

    else:
        # Standard distributed sampling
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank,
            shuffle=True,
            seed=opt.seed,  # Same seed for consistent epoch shuffling across ranks
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=opt.world_size, rank=opt.rank, shuffle=False
        )

        # Create DataLoader with standard sampler for training
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=False,
            generator=g,
            worker_init_fn=seed_worker,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        )

    # Validation loader is always the same
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=val_sampler,
        num_workers=min(2, num_workers),  # Only reduce validation workers slightly
        pin_memory=False,
        generator=g,
        worker_init_fn=seed_worker,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, train_sampler


def set_model(opt):
    if opt.etf_classifier:
        head = "etf"
    else:
        head = "linear"

    # Determine dataset type for architecture selection
    if opt.dataset in ["cifar10", "cifar100"]:
        dataset_type = "cifar"
    else:  # imagenet100, imagenet1k
        dataset_type = "imagenet"

    # CRITICAL FIX: Pass loss_type to model constructor
    model = SupCEResNet(
        name=opt.model,
        num_classes=opt.n_cls,
        normalize=opt.normalize,
        head=head,
        dataset_type=dataset_type,
        loss_type=opt.loss,
    )

    if opt.loss == "CE":
        criterion = torch.nn.CrossEntropyLoss()
    elif opt.loss == "NormFace":
        criterion = losses.NormFace(temperature=opt.temperature)
    elif opt.loss == "NTCE":
        criterion = losses.NTCE(temperature=opt.temperature)
    elif opt.loss == "NONL":
        criterion = losses.NONL(
            temperature=opt.temperature,
            symmetric=opt.symmetric,
            bidirectional=opt.bidirectional,
        )
    else:
        raise ValueError(f"Unknown loss: {opt.loss}")

    # CRITICAL: Convert BatchNorm to SyncBatchNorm BEFORE moving to GPU
    if opt.syncBN and opt.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Move model to GPU BEFORE wrapping with DDP
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.local_rank)
        model = model.cuda(opt.local_rank)
        criterion = criterion.cuda(opt.local_rank)

    # Wrap model with DDP only if world_size > 1
    if opt.world_size > 1:
        model = DDP(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,  # Can keep False now since no unused parameters
            broadcast_buffers=False,  # Reduce communication overhead - CRITICAL for stability
        )

    return model, criterion


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes."""
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def compute_val_metrics(model, val_loader, opt):
    """
    Compute neural collapse metrics on validation set during training.
    Only computes NC metrics for CIFAR datasets to avoid hanging on ImageNet.
    """
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Only collect features for NC metrics if dataset is CIFAR
    collect_features = opt.dataset in ["cifar10", "cifar100"]

    if collect_features:
        # Each GPU collects its own features and sends to CPU
        all_features = []
        all_weights = []
        all_labels = []

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()
    end = time.time()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            data_time.update(time.time() - end)

            images = images.cuda(opt.local_rank, non_blocking=True)
            labels = labels.cuda(opt.local_rank, non_blocking=True)

            output, features = model(images)

            if opt.world_size > 1:
                weight = torch.nn.functional.normalize(
                    model.module.fc.weight, p=2, dim=1
                ).T
            else:
                weight = torch.nn.functional.normalize(model.fc.weight, p=2, dim=1).T

            loss = criterion(output, labels)

            # ---- NEW: accuracy computation ----
            if opt.loss == "CE":
                logits_for_acc = output
            else:
                logits_for_acc = torch.matmul(features, weight)

            _, predicted = logits_for_acc.max(1)
            correct = predicted.eq(labels).sum().item()

            bsz = labels.size(0)
            total_loss += loss.item() * bsz
            total_correct += correct
            total_samples += bsz

            reduced_loss = reduce_tensor(loss, opt.world_size)
            reduced_correct = reduce_tensor(
                torch.tensor(correct, device="cuda", dtype=torch.float), opt.world_size
            )

            losses.update(reduced_loss.item(), bsz)
            top1.update((reduced_correct.item() / bsz) * 100.0, bsz)

            # Only collect features for CIFAR datasets
            if collect_features:
                all_features.append(features.detach().cpu())
                all_weights.append(weight.T[labels].detach().cpu())
                all_labels.append(labels.detach().cpu())

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % opt.print_freq == 0 and opt.rank == 0:
                print(
                    "Val:   [{0}/{1}]\t"
                    "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        idx + 1,
                        len(val_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        top1=top1,
                    )
                )
                sys.stdout.flush()

        # CRITICAL: Ensure all processes finish the validation loop together
        if opt.world_size > 1:
            dist.barrier()

    # Distributed reduction (total stats)
    if opt.world_size > 1:
        total_loss_tensor = torch.tensor(total_loss, device="cuda")
        total_correct_tensor = torch.tensor(total_correct, device="cuda")
        total_samples_tensor = torch.tensor(total_samples, device="cuda")

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        total_loss = total_loss_tensor.item()
        total_correct = total_correct_tensor.item()
        total_samples = total_samples_tensor.item()

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    # Only compute NC metrics for CIFAR datasets
    if not collect_features:
        if opt.rank == 0:
            print(f"[INFO] Skipping NC metrics for {opt.dataset} dataset")
        return {
            "val/loss": avg_loss,
            "val/acc": accuracy,
            "val/w_inst_alignment": 0.0,
            "val/w_erank": 0.0,
            "val/w_class_alignment": 0.0,
            "val/mir": 0.0,
            "val/hdr": 0.0,
            "val/er_intra": 0.0,
            "val/er_inter": 0.0,
        }

    # Rest of the function remains the same for CIFAR datasets
    # Gather all features from all GPUs
    if opt.world_size > 1:
        if opt.rank == 0:
            print(f"[INFO] Gathering features from all {opt.world_size} GPUs...")

        # Convert lists to tensors for gathering
        if len(all_features) > 0:
            local_features = torch.cat(all_features, dim=0)
            local_weights = torch.cat(all_weights, dim=0)
            local_labels = torch.cat(all_labels, dim=0)
        else:
            # Handle empty case
            feature_dim = (
                model.module.fc.weight.shape[1]
                if opt.world_size > 1
                else model.fc.weight.shape[1]
            )
            local_features = torch.empty(0, feature_dim)
            local_weights = torch.empty(0, feature_dim)
            local_labels = torch.empty(0, dtype=torch.long)

        # Gather sizes first
        local_size = torch.tensor(
            local_features.shape[0], dtype=torch.long, device="cuda"
        )
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(opt.world_size)]
        dist.all_gather(gathered_sizes, local_size)
        gathered_sizes = [size.item() for size in gathered_sizes]

        # Only gather if there's data to gather
        if sum(gathered_sizes) > 0:
            # Gather features
            max_size = max(gathered_sizes)
            if local_features.shape[0] < max_size:
                padding = torch.zeros(
                    max_size - local_features.shape[0], local_features.shape[1]
                )
                local_features_padded = torch.cat([local_features, padding], dim=0)
                local_weights_padded = torch.cat([local_weights, padding], dim=0)
                local_labels_padded = torch.cat(
                    [local_labels, torch.zeros(padding.shape[0], dtype=torch.long)],
                    dim=0,
                )
            else:
                local_features_padded = local_features
                local_weights_padded = local_weights
                local_labels_padded = local_labels

            # Move to GPU for gathering
            local_features_gpu = local_features_padded.cuda(opt.local_rank)
            local_weights_gpu = local_weights_padded.cuda(opt.local_rank)
            local_labels_gpu = local_labels_padded.cuda(opt.local_rank)

            gathered_features = [
                torch.zeros_like(local_features_gpu) for _ in range(opt.world_size)
            ]
            gathered_weights = [
                torch.zeros_like(local_weights_gpu) for _ in range(opt.world_size)
            ]
            gathered_labels = [
                torch.zeros_like(local_labels_gpu) for _ in range(opt.world_size)
            ]

            dist.all_gather(gathered_features, local_features_gpu)
            dist.all_gather(gathered_weights, local_weights_gpu)
            dist.all_gather(gathered_labels, local_labels_gpu)

            # Only rank 0 computes NC metrics
            if opt.rank == 0:
                # Combine and trim gathered data
                all_z = []
                all_b = []
                all_y = []

                for rank_idx in range(opt.world_size):
                    actual_size = gathered_sizes[rank_idx]
                    if actual_size > 0:
                        all_z.append(gathered_features[rank_idx][:actual_size].cpu())
                        all_b.append(gathered_weights[rank_idx][:actual_size].cpu())
                        all_y.append(gathered_labels[rank_idx][:actual_size].cpu())

                if len(all_z) > 0:
                    z = torch.cat(all_z, dim=0)
                    b = torch.cat(all_b, dim=0)
                    y = torch.cat(all_y, dim=0)

                    try:
                        if opt.world_size > 1:
                            weight_cpu = (
                                torch.nn.functional.normalize(
                                    model.module.fc.weight, p=2, dim=1
                                )
                                .detach()
                                .cpu()
                                .T
                            )
                        else:
                            weight_cpu = (
                                torch.nn.functional.normalize(
                                    model.fc.weight, p=2, dim=1
                                )
                                .detach()
                                .cpu()
                                .T
                            )

                        er_intra, er_inter = util.embedding_ETF_metrics(z, y)
                        mir, hdr = util.weight_embeddings_information(z, weight_cpu, y)
                        w_instance_alignment, w_erank, w_class_alignment = util.NC(
                            b, z, weight_cpu, y
                        )

                        metrics = {
                            "val/loss": avg_loss,
                            "val/acc": accuracy,
                            "val/w_inst_alignment": w_instance_alignment,
                            "val/w_erank": w_erank,
                            "val/w_class_alignment": w_class_alignment,
                            "val/mir": mir,
                            "val/hdr": hdr,
                            "val/er_intra": er_intra,
                            "val/er_inter": er_inter,
                        }

                        # Signal completion to other ranks
                        completion_signal = torch.tensor([1.0], device="cuda")
                    except Exception as e:
                        print(f"[WARNING] Failed to compute NC metrics: {e}")
                        completion_signal = torch.tensor([0.0], device="cuda")
                        metrics = {
                            "val/loss": avg_loss,
                            "val/acc": accuracy,
                            "val/w_inst_alignment": 0.0,
                            "val/w_erank": 0.0,
                            "val/w_class_alignment": 0.0,
                            "val/mir": 0.0,
                            "val/hdr": 0.0,
                            "val/er_intra": 0.0,
                            "val/er_inter": 0.0,
                        }

                    # Broadcast completion signal
                    dist.broadcast(completion_signal, src=0)
                else:
                    metrics = {
                        "val/loss": avg_loss,
                        "val/acc": accuracy,
                        "val/w_inst_alignment": 0.0,
                        "val/w_erank": 0.0,
                        "val/w_class_alignment": 0.0,
                        "val/mir": 0.0,
                        "val/hdr": 0.0,
                        "val/er_intra": 0.0,
                        "val/er_inter": 0.0,
                    }
            else:
                # Wait for rank 0 to finish computing
                completion_signal = torch.tensor([0.0], device="cuda")
                dist.broadcast(completion_signal, src=0)

                metrics = {
                    "val/loss": avg_loss,
                    "val/acc": accuracy,
                    "val/w_inst_alignment": 0.0,
                    "val/w_erank": 0.0,
                    "val/w_class_alignment": 0.0,
                    "val/mir": 0.0,
                    "val/hdr": 0.0,
                    "val/er_intra": 0.0,
                    "val/er_inter": 0.0,
                }
        else:
            # Signal no computation needed
            if opt.rank == 0:
                completion_signal = torch.tensor([0.0], device="cuda")
                dist.broadcast(completion_signal, src=0)
            else:
                completion_signal = torch.tensor([0.0], device="cuda")
                dist.broadcast(completion_signal, src=0)

            metrics = {
                "val/loss": avg_loss,
                "val/acc": accuracy,
                "val/w_inst_alignment": 0.0,
                "val/w_erank": 0.0,
                "val/w_class_alignment": 0.0,
                "val/mir": 0.0,
                "val/hdr": 0.0,
                "val/er_intra": 0.0,
                "val/er_inter": 0.0,
            }
    else:
        # Single GPU case
        if len(all_features) > 0:
            z = torch.cat(all_features, dim=0)
            b = torch.cat(all_weights, dim=0)
            y = torch.cat(all_labels, dim=0)

            weight_cpu = (
                torch.nn.functional.normalize(model.fc.weight, p=2, dim=1)
                .detach()
                .cpu()
                .T
            )

            er_intra, er_inter = util.embedding_ETF_metrics(z, y)
            mir, hdr = util.weight_embeddings_information(z, weight_cpu, y)
            w_instance_alignment, w_erank, w_class_alignment = util.NC(
                b, z, weight_cpu, y
            )

            metrics = {
                "val/loss": avg_loss,
                "val/acc": accuracy,
                "val/w_inst_alignment": w_instance_alignment,
                "val/w_erank": w_erank,
                "val/w_class_alignment": w_class_alignment,
                "val/mir": mir,
                "val/hdr": hdr,
                "val/er_intra": er_intra,
                "val/er_inter": er_inter,
            }
        else:
            metrics = {
                "val/loss": avg_loss,
                "val/acc": accuracy,
                "val/w_inst_alignment": 0.0,
                "val/w_erank": 0.0,
                "val/w_class_alignment": 0.0,
                "val/mir": 0.0,
                "val/hdr": 0.0,
                "val/er_intra": 0.0,
                "val/er_inter": 0.0,
            }

    # CRITICAL: Final barrier to ensure all ranks finish together
    if opt.world_size > 1:
        dist.barrier()

    return metrics


def compute_all_features_and_metrics(model, loader, opt, split_name="train"):
    """
    Compute all features for the entire dataset and calculate neural collapse metrics.
    Only computes NC metrics for CIFAR datasets to avoid hanging on ImageNet.
    """
    model.eval()

    # Only collect features for NC metrics if dataset is CIFAR
    collect_features = opt.dataset in ["cifar10", "cifar100"]

    if collect_features:
        # Each GPU collects its own features and sends to CPU
        all_features = []
        all_weights = []
        all_labels = []
        all_outputs = []

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()

    if opt.rank == 0:
        if collect_features:
            print(f"\n[INFO] Computing features and NC metrics for {split_name} set...")
        else:
            print(
                f"\n[INFO] Computing basic metrics for {split_name} set (NC metrics skipped for {opt.dataset})..."
            )

    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images = images.cuda(opt.local_rank, non_blocking=True)
            labels = labels.cuda(opt.local_rank, non_blocking=True)

            # Forward pass
            output, features = model(images)

            if collect_features:
                # Get normalized weights
                if opt.world_size > 1:
                    weight = torch.nn.functional.normalize(
                        model.module.fc.weight, p=2, dim=1
                    ).T
                else:
                    weight = torch.nn.functional.normalize(
                        model.fc.weight, p=2, dim=1
                    ).T

            # Compute loss
            loss = criterion(output, labels)

            # ---- NEW: accuracy computation ----
            if opt.loss == "CE":
                logits_for_acc = output
            else:
                # For ImageNet we may not collect_features, so recompute weight if needed
                if not collect_features:
                    if opt.world_size > 1:
                        weight = torch.nn.functional.normalize(
                            model.module.fc.weight, p=2, dim=1
                        ).T
                    else:
                        weight = torch.nn.functional.normalize(
                            model.fc.weight, p=2, dim=1
                        ).T
                logits_for_acc = torch.matmul(features, weight)

            _, predicted = logits_for_acc.max(1)
            correct = predicted.eq(labels).sum().item()

            # Accumulate results
            total_loss += loss.item() * labels.size(0)
            total_correct += correct
            total_samples += labels.size(0)

            # Only collect features for CIFAR datasets
            if collect_features:
                all_features.append(features.detach().cpu())
                all_weights.append(weight.T[labels].detach().cpu())
                all_labels.append(labels.detach().cpu())
                all_outputs.append(output.detach().cpu())

            if idx % 100 == 0 and opt.rank == 0:
                print(f"  Processed {idx}/{len(loader)} batches")

    # Reduce across all processes for basic metrics
    if opt.world_size > 1:
        # Convert to tensors for reduction
        total_loss_tensor = torch.tensor(total_loss, device="cuda")
        total_correct_tensor = torch.tensor(total_correct, device="cuda")
        total_samples_tensor = torch.tensor(total_samples, device="cuda")

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        total_loss = total_loss_tensor.item()
        total_correct = total_correct_tensor.item()
        total_samples = total_samples_tensor.item()

    # Compute average metrics
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    # If not collecting features (ImageNet), return basic metrics
    if not collect_features:
        if opt.rank == 0:
            print(f"\n{split_name.upper()} SET BASIC METRICS:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.2f}%")

        return {
            f"{split_name}/loss": avg_loss,
            f"{split_name}/acc": accuracy,
            f"{split_name}/prototype_acc": 0.0,
            f"{split_name}/w_inst_alignment": 0.0,
            f"{split_name}/w_erank": 0.0,
            f"{split_name}/w_class_alignment": 0.0,
            f"{split_name}/mir": 0.0,
            f"{split_name}/hdr": 0.0,
            f"{split_name}/er_intra": 0.0,
            f"{split_name}/er_inter": 0.0,
        }

    # Gather all features from all GPUs for NC computation (CIFAR only)
    if opt.world_size > 1:
        if opt.rank == 0:
            print(f"  Gathering features from all {opt.world_size} GPUs...")

        # Convert lists to tensors for gathering
        if len(all_features) > 0:
            local_features = torch.cat(all_features, dim=0)
            local_weights = torch.cat(all_weights, dim=0)
            local_labels = torch.cat(all_labels, dim=0)
            local_outputs = torch.cat(all_outputs, dim=0)
        else:
            # Handle empty case
            feature_dim = (
                model.module.fc.weight.shape[1]
                if opt.world_size > 1
                else model.fc.weight.shape[1]
            )
            num_classes = (
                model.module.fc.weight.shape[0]
                if opt.world_size > 1
                else model.fc.weight.shape[0]
            )
            local_features = torch.empty(0, feature_dim)
            local_weights = torch.empty(0, feature_dim)
            local_labels = torch.empty(0, dtype=torch.long)
            local_outputs = torch.empty(0, num_classes)

        # Gather sizes first
        local_size = torch.tensor(
            local_features.shape[0], dtype=torch.long, device="cuda"
        )
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(opt.world_size)]
        dist.all_gather(gathered_sizes, local_size)
        gathered_sizes = [size.item() for size in gathered_sizes]

        if opt.rank == 0:
            print(f"  Features per GPU: {gathered_sizes}")

        # Only gather if there's data to gather
        if sum(gathered_sizes) > 0:
            # Pad to same size for gathering
            max_size = max(gathered_sizes)
            if local_features.shape[0] < max_size:
                padding_size = max_size - local_features.shape[0]
                feature_padding = torch.zeros(padding_size, local_features.shape[1])
                output_padding = torch.zeros(padding_size, local_outputs.shape[1])
                label_padding = torch.zeros(padding_size, dtype=torch.long)

                local_features_padded = torch.cat(
                    [local_features, feature_padding], dim=0
                )
                local_weights_padded = torch.cat(
                    [local_weights, feature_padding], dim=0
                )
                local_labels_padded = torch.cat([local_labels, label_padding], dim=0)
                local_outputs_padded = torch.cat([local_outputs, output_padding], dim=0)
            else:
                local_features_padded = local_features
                local_weights_padded = local_weights
                local_labels_padded = local_labels
                local_outputs_padded = local_outputs

            # Move to GPU for gathering
            local_features_gpu = local_features_padded.cuda(opt.local_rank)
            local_weights_gpu = local_weights_padded.cuda(opt.local_rank)
            local_labels_gpu = local_labels_padded.cuda(opt.local_rank)
            local_outputs_gpu = local_outputs_padded.cuda(opt.local_rank)

            gathered_features = [
                torch.zeros_like(local_features_gpu) for _ in range(opt.world_size)
            ]
            gathered_weights = [
                torch.zeros_like(local_weights_gpu) for _ in range(opt.world_size)
            ]
            gathered_labels = [
                torch.zeros_like(local_labels_gpu) for _ in range(opt.world_size)
            ]
            gathered_outputs = [
                torch.zeros_like(local_outputs_gpu) for _ in range(opt.world_size)
            ]

            dist.all_gather(gathered_features, local_features_gpu)
            dist.all_gather(gathered_weights, local_weights_gpu)
            dist.all_gather(gathered_labels, local_labels_gpu)
            dist.all_gather(gathered_outputs, local_outputs_gpu)

            # Only rank 0 computes final NC metrics
            if opt.rank == 0:
                print(f"  Computing neural collapse metrics on combined data...")

                # Combine and trim gathered data
                all_z = []
                all_b = []
                all_y = []
                all_out = []

                for rank_idx in range(opt.world_size):
                    actual_size = gathered_sizes[rank_idx]
                    if actual_size > 0:
                        all_z.append(gathered_features[rank_idx][:actual_size].cpu())
                        all_b.append(gathered_weights[rank_idx][:actual_size].cpu())
                        all_y.append(gathered_labels[rank_idx][:actual_size].cpu())
                        all_out.append(gathered_outputs[rank_idx][:actual_size].cpu())

                if len(all_z) > 0:
                    z = torch.cat(all_z, dim=0)
                    b = torch.cat(all_b, dim=0)
                    y = torch.cat(all_y, dim=0)
                    outputs = torch.cat(all_out, dim=0)

                    print(
                        f"  Final combined data: z.shape={z.shape}, y.shape={y.shape}"
                    )

                    # Get final weights for metrics computation
                    if opt.world_size > 1:
                        weight_cpu = (
                            torch.nn.functional.normalize(
                                model.module.fc.weight, p=2, dim=1
                            )
                            .detach()
                            .cpu()
                            .T
                        )
                    else:
                        weight_cpu = (
                            torch.nn.functional.normalize(model.fc.weight, p=2, dim=1)
                            .detach()
                            .cpu()
                            .T
                        )

                    # Compute neural collapse metrics
                    er_intra, er_inter = util.embedding_ETF_metrics(z, y)
                    mir, hdr = util.weight_embeddings_information(z, weight_cpu, y)
                    w_instance_alignment, w_erank, w_class_alignment = util.NC(
                        b, z, weight_cpu, y
                    )

                    # Compute prototype accuracy
                    prototypes = []
                    for class_idx in range(opt.n_cls):
                        class_mask = y == class_idx
                        if class_mask.sum() > 0:
                            class_features = z[class_mask]
                            class_prototype = torch.nn.functional.normalize(
                                class_features.mean(dim=0), dim=0
                            )
                            prototypes.append(class_prototype)
                        else:
                            prototypes.append(torch.zeros(z.shape[1]))

                    prototypes = torch.stack(prototypes)  # (n_classes, feature_dim)

                    # Compute prototype-based accuracy
                    prototype_logits = torch.matmul(z, prototypes.T)
                    _, prototype_predicted = prototype_logits.max(1)
                    prototype_accuracy = (
                        100.0 * prototype_predicted.eq(y).sum().item() / len(y)
                    )

                    metrics = {
                        f"{split_name}/loss": avg_loss,
                        f"{split_name}/acc": accuracy,
                        f"{split_name}/prototype_acc": prototype_accuracy,
                        f"{split_name}/w_inst_alignment": w_instance_alignment,
                        f"{split_name}/w_erank": w_erank,
                        f"{split_name}/w_class_alignment": w_class_alignment,
                        f"{split_name}/mir": mir,
                        f"{split_name}/hdr": hdr,
                        f"{split_name}/er_intra": er_intra,
                        f"{split_name}/er_inter": er_inter,
                    }

                    print(f"\n{split_name.upper()} SET FINAL METRICS:")
                    print(f"  Loss: {avg_loss:.4f}")
                    print(f"  Accuracy: {accuracy:.2f}%")
                    print(f"  Prototype Accuracy: {prototype_accuracy:.2f}%")
                    print(f"  W Instance Alignment: {w_instance_alignment:.4f}")
                    print(f"  W Effective Rank: {w_erank:.4f}")
                    print(f"  W Class Alignment: {w_class_alignment:.2f}")
                    print(f"  MIR: {mir:.4f}")
                    print(f"  HDR: {hdr:.4f}")
                    print(f"  Embedding Intra-class: {er_intra:.4f}")
                    print(f"  Embedding Inter-class: {er_inter:.4f}")
                else:
                    print("  No data to process for NC metrics")
                    metrics = {
                        f"{split_name}/loss": avg_loss,
                        f"{split_name}/acc": accuracy,
                        f"{split_name}/prototype_acc": 0.0,
                        f"{split_name}/w_inst_alignment": 0.0,
                        f"{split_name}/w_erank": 0.0,
                        f"{split_name}/w_class_alignment": 0.0,
                        f"{split_name}/mir": 0.0,
                        f"{split_name}/hdr": 0.0,
                        f"{split_name}/er_intra": 0.0,
                        f"{split_name}/er_inter": 0.0,
                    }
            else:
                # Non-rank 0 processes return basic metrics
                metrics = {
                    f"{split_name}/loss": avg_loss,
                    f"{split_name}/acc": accuracy,
                    f"{split_name}/prototype_acc": 0.0,
                    f"{split_name}/w_inst_alignment": 0.0,
                    f"{split_name}/w_erank": 0.0,
                    f"{split_name}/w_class_alignment": 0.0,
                    f"{split_name}/mir": 0.0,
                    f"{split_name}/hdr": 0.0,
                    f"{split_name}/er_intra": 0.0,
                    f"{split_name}/er_inter": 0.0,
                }
        else:
            print("  No features to gather")
            metrics = {
                f"{split_name}/loss": avg_loss,
                f"{split_name}/acc": accuracy,
                f"{split_name}/prototype_acc": 0.0,
                f"{split_name}/w_inst_alignment": 0.0,
                f"{split_name}/w_erank": 0.0,
                f"{split_name}/w_class_alignment": 0.0,
                f"{split_name}/mir": 0.0,
                f"{split_name}/hdr": 0.0,
                f"{split_name}/er_intra": 0.0,
                f"{split_name}/er_inter": 0.0,
            }
    else:
        # Single GPU case - compute directly for CIFAR
        if opt.rank == 0:
            print(
                f"  Computing neural collapse metrics on {len(all_features)} batches..."
            )

        # Concatenate all features and labels
        z = torch.cat(all_features, dim=0)
        b = torch.cat(all_weights, dim=0)
        y = torch.cat(all_labels, dim=0)
        outputs = torch.cat(all_outputs, dim=0)

        # Get final weights for metrics computation
        weight_cpu = (
            torch.nn.functional.normalize(model.fc.weight, p=2, dim=1).detach().cpu().T
        )

        # Compute neural collapse metrics
        er_intra, er_inter = util.embedding_ETF_metrics(z, y)
        mir, hdr = util.weight_embeddings_information(z, weight_cpu, y)
        w_instance_alignment, w_erank, w_class_alignment = util.NC(b, z, weight_cpu, y)

        # Compute prototype accuracy
        prototypes = []
        for class_idx in range(opt.n_cls):
            class_mask = y == class_idx
            if class_mask.sum() > 0:
                class_features = z[class_mask]
                class_prototype = torch.nn.functional.normalize(
                    class_features.mean(dim=0), dim=0
                )
                prototypes.append(class_prototype)
            else:
                prototypes.append(torch.zeros(z.shape[1]))

        prototypes = torch.stack(prototypes)  # (n_classes, feature_dim)

        # Compute prototype-based accuracy
        prototype_logits = torch.matmul(z, prototypes.T)
        _, prototype_predicted = prototype_logits.max(1)
        prototype_accuracy = 100.0 * prototype_predicted.eq(y).sum().item() / len(y)

        metrics = {
            f"{split_name}/loss": avg_loss,
            f"{split_name}/acc": accuracy,
            f"{split_name}/prototype_acc": prototype_accuracy,
            f"{split_name}/w_inst_alignment": w_instance_alignment,
            f"{split_name}/w_erank": w_erank,
            f"{split_name}/w_class_alignment": w_class_alignment,
            f"{split_name}/mir": mir,
            f"{split_name}/hdr": hdr,
            f"{split_name}/er_intra": er_intra,
            f"{split_name}/er_inter": er_inter,
        }

        print(f"\n{split_name.upper()} SET FINAL METRICS:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Prototype Accuracy: {prototype_accuracy:.2f}%")
        print(f"  W Instance Alignment: {w_instance_alignment:.4f}")
        print(f"  W Effective Rank: {w_erank:.4f}")
        print(f"  W Class Alignment: {w_class_alignment:.2f}")
        print(f"  MIR: {mir:.4f}")
        print(f"  HDR: {hdr:.4f}")
        print(f"  Embedding Intra-class: {er_intra:.4f}")
        print(f"  Embedding Inter-class: {er_inter:.4f}")

    return metrics


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """
    Training loop - only computes loss and accuracy, no neural collapse metrics.
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(opt.local_rank, non_blocking=True)
        labels = labels.cuda(opt.local_rank, non_blocking=True)
        bsz = labels.shape[0]

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # Get normalized weights
        if opt.world_size > 1:
            weight = torch.nn.functional.normalize(model.module.fc.weight, p=2, dim=1).T
        else:
            weight = torch.nn.functional.normalize(model.fc.weight, p=2, dim=1).T

        output, features = model(images)

        if opt.loss == "CE":
            loss = criterion(output, labels)
        else:
            loss = criterion(features, weight, labels)

        # Reduce loss across all processes for logging
        reduced_loss = reduce_tensor(loss, opt.world_size)
        losses.update(reduced_loss.item(), bsz)

        # ---- NEW: accuracy computed like in robustness.py ----
        if opt.loss == "CE":
            logits_for_acc = output
        else:
            # features: (B, feat_dim), weight: (feat_dim, num_classes)
            # logits = cosine similarities between normalized features and normalized weights
            logits_for_acc = torch.matmul(features, weight)

        acc1, _ = accuracy(logits_for_acc, labels, topk=(1, 5))
        reduced_acc1 = reduce_tensor(acc1[0] / 100.0, opt.world_size)
        top1.update(reduced_acc1.item() * 100.0, bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0 and opt.rank == 0:
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )
            sys.stdout.flush()

            # Log batch metrics to wandb (only on rank 0)
            if hasattr(opt, "wandb_initialized") and opt.wandb_initialized:
                wandb.log(
                    {
                        "train/batch_loss": losses.val,
                        "train/batch_acc": top1.val,
                        "train/batch_time": batch_time.val,
                        "train/data_time": data_time.val,
                        "step": epoch * len(train_loader) + idx,
                    }
                )

    return losses.avg, top1.avg


def main_worker(opt):
    """Main worker function for each process."""
    # DDP is already setup in main()

    # CRITICAL FIX: Validate local_rank before setting device
    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        if opt.local_rank >= num_available_gpus:
            print(
                f"[ERROR] Local rank {opt.local_rank} >= available GPUs {num_available_gpus}"
            )
            print(
                f"[INFO] Adjusting local_rank from {opt.local_rank} to {opt.local_rank % num_available_gpus}"
            )
            opt.local_rank = opt.local_rank % num_available_gpus

        torch.cuda.set_device(opt.local_rank)
        device = torch.cuda.current_device()
        print(f"[Rank {opt.rank}] Using GPU {device} (local_rank {opt.local_rank})")
    else:
        print(f"[Rank {opt.rank}] No CUDA available, using CPU")
        opt.local_rank = 0

    # Set random seed for this process (CRITICAL for reproducibility)
    set_seed(opt.seed + opt.rank, deterministic=opt.deterministic)
    if opt.rank == 0:
        print(f"\n[INFO] Random seed set to: {opt.seed} (base) + {opt.rank} (rank)")
        print(f"[INFO] Deterministic mode: {'ON' if opt.deterministic else 'OFF'}")

    # Get GPU information (only print from rank 0)
    if opt.rank == 0:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"[INFO] Number of GPUs available: {num_gpus}")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"[INFO] GPU {i}: {gpu_name}")
            print(f"[INFO] CUDA Version: {torch.version.cuda}")
            print(f"[INFO] PyTorch Version: {torch.__version__}")
        else:
            print("[INFO] No GPUs available, using CPU")

    # Initialize Weights & Biases (only on rank 0)
    opt.wandb_initialized = False
    if opt.rank == 0:
        wandb_config = vars(opt)
        wandb_config["num_gpus"] = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        wandb_config["world_size"] = opt.world_size

        wandb.init(
            project=opt.wandb_project,
            entity=opt.wandb_entity,
            name=opt.wandb_run_name if opt.wandb_run_name else opt.model_name,
            config=wandb_config,
        )

        # Log hardware info to wandb
        if torch.cuda.is_available():
            gpu_info = ", ".join(
                [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
            )
        else:
            gpu_info = "CPU"

        wandb.config.update(
            {
                "gpu": gpu_info,
                "num_gpus": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
                "cuda_version": (
                    torch.version.cuda if torch.cuda.is_available() else "N/A"
                ),
                "pytorch_version": torch.__version__,
                "deterministic": opt.deterministic,
                "world_size": opt.world_size,
            }
        )

        opt.wandb_initialized = True

    # Set up data loaders BEFORE model creation
    train_loader, val_loader, train_sampler = set_loader(opt)

    # Set up model and criterion
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)

    # Debug: Print optimizer settings
    if opt.rank == 0:
        print(f"\n[DEBUG] Optimizer Configuration:")
        print(f"  Type: {type(optimizer).__name__}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Weight decay: {optimizer.param_groups[0]['weight_decay']}")
        if "momentum" in optimizer.param_groups[0]:
            print(f"  Momentum: {optimizer.param_groups[0]['momentum']}")
        if "betas" in optimizer.param_groups[0]:
            print(f"  Betas: {optimizer.param_groups[0]['betas']}")
        print()

    # For wandb.watch, use only on rank 0
    if opt.rank == 0 and opt.wandb_initialized:
        if opt.world_size > 1:
            wandb.watch(model.module, log="all", log_freq=opt.print_freq)
        else:
            wandb.watch(model, log="all", log_freq=opt.print_freq)

    # --- Logging the configuration (only on rank 0) ---
    if opt.rank == 0:
        total_batch_size = opt.batch_size * opt.world_size
        print("\n========== Training Configuration ==========")
        print(f"Dataset: {opt.dataset}")
        if opt.dataset == "imagenet100":
            print(f"ImageNet-100 Path: {opt.imagenet100_path}")
        elif opt.dataset == "imagenet1k":
            print(f"ImageNet-1K Path: {opt.imagenet1k_path}")
        print(f"Model Architecture: {opt.model}")
        print(f"Loss Function: {opt.loss}")
        print(f"Normalize Features: {opt.normalize}")
        print(f"Use Cosine LR Schedule: {opt.cosine}")

        if opt.class_balanced:
            per_gpu_bsz = opt.classes_per_batch * opt.samples_per_class
        else:
            per_gpu_bsz = opt.batch_size
        total_batch_size = per_gpu_bsz * opt.world_size

        print(
            f"Batch Size per GPU: {per_gpu_bsz} "
            f"{'(class-balanced)' if opt.class_balanced else ''}"
        )
        print(f"Total Batch Size: {total_batch_size}")

        print(f"World Size: {opt.world_size}")
        print(f"Epochs: {opt.epochs}")
        print(f"Learning Rate: {opt.learning_rate}")
        print(f"LR Decay Epochs: {opt.lr_decay_epochs}")
        print(f"Weight Decay: {opt.weight_decay}")
        print(f"Momentum: {opt.momentum}")
        print(f"Temperature: {opt.temperature}")
        print(f"Warmup: {opt.warm}")
        if opt.warm:
            print(f"  - Warmup epochs: {opt.warm_epochs}")
            print(f"  - Warmup from: {opt.warmup_from}")
            print(f"  - Warmup to: {opt.warmup_to}")
        print(f"Sync BatchNorm: {opt.syncBN}")
        print(f"Trial ID: {opt.trial}")
        print(f"Random Seed: {opt.seed}")
        print(f"Deterministic: {opt.deterministic}")
        print(
            f"Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )
        print(f"Model Save Path: {opt.save_folder}")
        print("============================================\n")

    # Set up TensorBoard writer (only on rank 0)
    writer = None
    if opt.rank == 0:
        writer = SummaryWriter(log_dir=opt.tb_folder)

    best_acc = 0
    for epoch in range(1, opt.epochs + 1):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(opt, optimizer, epoch)

        time1 = time.time()
        train_loss, train_acc = train(
            train_loader, model, criterion, optimizer, epoch, opt
        )
        time2 = time.time()

        if opt.rank == 0:
            print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # Compute validation metrics with neural collapse metrics
        val_metrics = compute_val_metrics(model, val_loader, opt)
        val_loss = val_metrics["val/loss"]
        val_acc = val_metrics["val/acc"]

        if opt.rank == 0:
            # Print validation results
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Only print detailed NC metrics for CIFAR datasets when they are actually computed
            if (
                opt.dataset in ["cifar10", "cifar100"]
                and val_metrics["val/w_inst_alignment"] != 0.0
            ):
                print(
                    f'Val Metrics - W_inst: {val_metrics["val/w_inst_alignment"]:.4f}, '
                    f'W_erank: {val_metrics["val/w_erank"]:.4f}, '
                    f'W_class: {val_metrics["val/w_class_alignment"]:.2f}, '
                    f'MIR: {val_metrics["val/mir"]:.4f}, '
                    f'HDR: {val_metrics["val/hdr"]:.4f}, '
                    f'ER_intra: {val_metrics["val/er_intra"]:.4f}, '
                    f'ER_inter: {val_metrics["val/er_inter"]:.4f}'
                )
            elif opt.dataset not in ["cifar10", "cifar100"]:
                print(f"[INFO] NC metrics skipped for {opt.dataset} dataset")

            # Log epoch metrics to wandb
            if opt.wandb_initialized:
                log_dict = {
                    "train/epoch_loss": train_loss,
                    "train/epoch_acc": train_acc,
                    "train/epoch_time": time2 - time1,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
                # Add validation metrics
                log_dict.update(val_metrics)
                wandb.log(log_dict)

            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar("train/loss", train_loss, epoch)
                writer.add_scalar("train/acc", train_acc, epoch)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

                # Add validation metrics to tensorboard
                for key, value in val_metrics.items():
                    writer.add_scalar(key, value, epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                if opt.wandb_initialized:
                    wandb.run.summary["best_val_acc"] = best_acc

            # Save model checkpoints (only on rank 0)
            if epoch % opt.save_freq == 0:
                save_file = os.path.join(opt.save_folder, f"ckpt_epoch_{epoch}.pth")
                save_model(model, optimizer, opt, epoch, save_file)

    # === FINAL EVALUATION: Compute all metrics on both train and val sets ===
    if opt.rank == 0:
        print("\n" + "=" * 80)
        print("FINAL EVALUATION: Computing all metrics on fixed trained model")
        print("=" * 80)

    # Compute final metrics on training set
    train_final_metrics = compute_all_features_and_metrics(
        model, train_loader, opt, "train_final"
    )

    # Compute final metrics on validation set
    val_final_metrics = compute_all_features_and_metrics(
        model, val_loader, opt, "val_final"
    )

    if opt.rank == 0:
        # Log final metrics to wandb
        if opt.wandb_initialized:
            final_log_dict = {}
            final_log_dict.update(train_final_metrics)
            final_log_dict.update(val_final_metrics)
            final_log_dict["epoch"] = opt.epochs
            wandb.log(final_log_dict)

            # Update wandb summary with final metrics
            wandb.run.summary.update(
                {
                    "best_val_acc": best_acc,
                    "final_train_acc": train_final_metrics["train_final/acc"],
                    "final_val_acc": val_final_metrics["val_final/acc"],
                    "final_train_prototype_acc": train_final_metrics[
                        "train_final/prototype_acc"
                    ],
                    "final_val_prototype_acc": val_final_metrics[
                        "val_final/prototype_acc"
                    ],
                }
            )

        # Log final metrics to TensorBoard
        if writer is not None:
            for key, value in train_final_metrics.items():
                writer.add_scalar(key, value, opt.epochs)
            for key, value in val_final_metrics.items():
                writer.add_scalar(key, value, opt.epochs)

        # Save final model
        save_file = os.path.join(opt.save_folder, "last.pth")
        save_model(model, optimizer, opt, opt.epochs, save_file)

        if writer is not None:
            writer.close()

        print(f"\nFINAL RESULTS:")
        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        print(f'Final Train Accuracy: {train_final_metrics["train_final/acc"]:.2f}%')
        print(f'Final Val Accuracy: {val_final_metrics["val_final/acc"]:.2f}%')
        print(
            f'Final Train Prototype Accuracy: {train_final_metrics["train_final/prototype_acc"]:.2f}%'
        )
        print(
            f'Final Val Prototype Accuracy: {val_final_metrics["val_final/prototype_acc"]:.2f}%'
        )

        if opt.wandb_initialized:
            wandb.finish()

    # Clean up the process group
    cleanup()


def main():
    # Apply memory optimizations
    optimize_memory_for_imagenet()

    # Parse arguments
    opt = parse_option()

    if "SLURM_JOB_ID" in os.environ and opt.rank == 0:
        debug_slurm_environment()

    # Check if we're in a distributed environment
    if "WORLD_SIZE" in os.environ:
        # Already launched with torchrun or srun
        rank, world_size, local_rank = setup_ddp()
        opt.rank = rank
        opt.world_size = world_size
        opt.local_rank = local_rank
        main_worker(opt)
    else:
        # Single GPU or need to spawn processes
        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
            if world_size > 1:
                # Use torchrun instead of mp.spawn for better stability
                print("Please use torchrun for multi-GPU training:")
                print(f"torchrun --nproc_per_node={world_size} train.py [your_args]")
                return
            else:
                # Single GPU
                opt.rank = 0
                opt.world_size = 1
                opt.local_rank = 0
                main_worker(opt)
        else:
            # CPU training
            opt.rank = 0
            opt.world_size = 1
            opt.local_rank = 0
            main_worker(opt)


if __name__ == "__main__":
    main()
