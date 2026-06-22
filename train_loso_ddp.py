"""Distributed LOSO training script for SAC2-Net.

This script supports both single-process training and multi-GPU training with
PyTorch DistributedDataParallel (DDP). For multi-GPU training, launch it with
`torchrun`, for example:

    torchrun --nproc_per_node=2 train_loso_ddp.py --dataset_annotation_path ...

Notes:
    - `--batch_size` is the per-GPU batch size when using DDP.
    - Only rank 0 runs validation and writes the final Excel file.
"""

import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from models.SAC2Net import SAC2Net as create_model
from utils.my_dataset import EmotionDataset
from utils.utils import create_lr_scheduler, evaluate, train_one_epoch


CLASS_COUNTS = {
    "megc2019_cd": [250, 109, 83],
    "samm_5cls": [57, 12, 26, 26, 15],
    "casme2_5cls": [63, 32, 99, 27, 28],
    "casme_cube_7cls": [64, 250, 86, 55, 161, 57, 187],
    "casme_cube_4cls": [457, 161, 55, 187],
}


# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def print_rank0(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed(args: argparse.Namespace) -> torch.device:
    """Initialize DDP when the script is launched by torchrun."""
    args.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training with this script requires CUDA GPUs.")

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
        )
        dist.barrier()

        print_rank0(
            f"Distributed training initialized: "
            f"world_size={args.world_size}, backend={args.dist_backend}"
        )
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

        if args.device == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)

        print_rank0(f"Single-process training on device: {device}")

    return device


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------------------------------------------------------
# Data and model helpers
# -----------------------------------------------------------------------------
def create_label_mapping(
    df: pd.DataFrame,
    label_col: str = "Estimated Emotion",
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create deterministic label-index mappings from the annotation file."""
    labels = sorted(df[label_col].unique())
    label2idx = {label: idx for idx, label in enumerate(labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    print_rank0(f"label2idx: {label2idx}")
    print_rank0(f"idx2label: {idx2label}")
    return label2idx, idx2label


def build_transforms(img_size: int = 224):
    """Build transforms for motion-magnified images and optical-flow images."""
    train_transform = {
        "mag": transforms.Compose([
            # Images are expected to have been resized during preprocessing.
            # transforms.Resize((img_size, img_size)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.46, 0.36, 0.33], std=[0.20, 0.17, 0.17]),
        ]),
        "flow": transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.87, 0.88, 0.90], std=[0.21, 0.19, 0.16]),
        ]),
    }

    val_transform = {
        "mag": transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.46, 0.36, 0.33], std=[0.20, 0.17, 0.17]),
        ]),
        "flow": transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.87, 0.88, 0.90], std=[0.21, 0.19, 0.16]),
        ]),
    }

    return train_transform, val_transform


def remove_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the `module.` prefix that may appear in DDP checkpoints."""
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_initial_weights(
    model: torch.nn.Module,
    weights_path: str,
    device: torch.device,
    load_sasa_only: bool = True,
) -> None:
    """Load pretrained weights.

    By default, this keeps the behavior of the original script: only the visual
    encoders and SASA head are loaded from the pretrained checkpoint. This is
    appropriate when `weights_path` points to SASA-pretrained weights.
    """
    if not weights_path:
        return

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = remove_module_prefix(state_dict)

    if load_sasa_only:
        keep_prefixes = ("mag_encoder", "flow_encoder", "sasa_head")
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(keep_prefixes)
        }

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print_rank0(
        f"Loaded weights from: {weights_path}\n"
        f"  load_sasa_only={load_sasa_only}\n"
        f"  missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
    )


def freeze_encoder_and_sasa(model: torch.nn.Module) -> None:
    """Freeze the visual encoders and SASA projection head."""
    freeze_prefixes = ("mag_encoder", "flow_encoder", "sasa_head")
    for name, param in model.named_parameters():
        if name.startswith(freeze_prefixes):
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)


def build_data_loaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    label2idx: Dict[str, int],
    train_transform,
    val_transform,
):
    train_dataset = EmotionDataset(
        train_df,
        dataset_root=args.dataset_root,
        label2idx=label2idx,
        transform=train_transform,
    )
    val_dataset = EmotionDataset(
        test_df,
        dataset_root=args.dataset_root,
        label2idx=label2idx,
        transform=val_transform,
    )

    train_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=False,
        )

    num_workers = min(args.num_workers, os.cpu_count() or args.num_workers)
    generator = torch.Generator()
    generator.manual_seed(args.seed + get_rank())

    print_rank0(
        f"DataLoader workers per process: {num_workers}. "
        f"Per-GPU batch size: {args.batch_size}. "
        f"Effective batch size: {args.batch_size * get_world_size()}."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=args.pin_memory,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # Validation is performed only on rank 0 using the full validation set.
    # Therefore, no DistributedSampler is used here.
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return train_loader, val_loader, train_sampler


def broadcast_early_stop(should_stop: bool, device: torch.device) -> bool:
    """Broadcast the early-stopping decision from rank 0 to all ranks."""
    if not is_dist_avail_and_initialized():
        return should_stop

    stop_tensor = torch.tensor([int(should_stop)], dtype=torch.int32, device=device)
    dist.broadcast(stop_tensor, src=0)
    return bool(stop_tensor.item())


# -----------------------------------------------------------------------------
# Main training procedure
# -----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    device = setup_distributed(args)
    set_seed(args.seed + get_rank(), deterministic=args.deterministic)

    if args.benchmark not in CLASS_COUNTS:
        raise ValueError(
            f"Unknown benchmark: {args.benchmark}. "
            f"Supported benchmarks: {list(CLASS_COUNTS.keys())}"
        )

    if not os.path.exists(args.dataset_annotation_path):
        raise FileNotFoundError(
            f"Annotation file does not exist: {args.dataset_annotation_path}"
        )

    df = pd.read_excel(args.dataset_annotation_path, converters={"Subject": str})
    required_cols = {"Subject", "Filename", "Apex", "Estimated Emotion"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Excel file is missing required columns: {missing_cols}. "
            f"Expected at least these columns: {required_cols}."
        )

    label2idx, idx2label = create_label_mapping(df, label_col="Estimated Emotion")
    train_transform, val_transform = build_transforms(img_size=224)

    subjects = sorted(df["Subject"].unique())
    print_rank0(f"Found {len(subjects)} subjects: {subjects}")

    all_results = []
    num_classes = len(label2idx)
    class_counts = CLASS_COUNTS[args.benchmark]

    for subj in subjects:
        print_rank0(f"\n===== LOSO fold: test subject = {subj} =====")

        train_df = df[df["Subject"] != subj].reset_index(drop=True)
        test_df = df[df["Subject"] == subj].reset_index(drop=True)
        print_rank0(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

        train_loader, val_loader, train_sampler = build_data_loaders(
            train_df=train_df,
            test_df=test_df,
            args=args,
            label2idx=label2idx,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        raw_model = create_model(num_classes=num_classes).to(device)
        load_initial_weights(
            model=raw_model,
            weights_path=args.weights,
            device=device,
            load_sasa_only=args.load_sasa_only,
        )

        if args.freeze_layers:
            freeze_encoder_and_sasa(raw_model)
            print_rank0("Frozen modules: mag_encoder, flow_encoder, sasa_head")

        param_groups = raw_model.get_trainable_param_groups(lr=args.lr)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd)
        lr_scheduler = create_lr_scheduler(
            optimizer,
            len(train_loader),
            args.epochs,
            warmup=True,
            warmup_epochs=args.warmup_epochs,
        )

        if args.distributed:
            train_model = DDP(
                raw_model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=args.find_unused_parameters,
            )
        else:
            train_model = raw_model

        epochs_no_improve = 0
        best_acc = -1.0
        best_epoch = -1
        best_preds_for_subj: Dict[str, List] = {}

        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss, train_acc, _ = train_one_epoch(
                model=train_model,
                optimizer=optimizer,
                data_loader=train_loader,
                device=device,
                epoch=epoch,
                total_epochs=args.epochs,
                lr_scheduler=lr_scheduler,
                au_prompt_templates_path=args.au_prompt_templates_path,
                num_classes=num_classes,
                class_counts=class_counts,
            )

            should_stop = False
            if is_main_process():
                # Use the unwrapped model to avoid DDP collectives during rank-0-only evaluation.
                val_loss, val_acc, val_preds, val_probs, val_labels = evaluate(
                    model=raw_model,
                    data_loader=val_loader,
                    device=device,
                    epoch=epoch,
                )

                print(
                    f"Subject {subj} | Epoch [{epoch:03d}/{args.epochs:03d}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}"
                )

                if val_acc > best_acc:
                    epochs_no_improve = 0
                    best_acc = val_acc
                    best_epoch = epoch
                    best_preds_for_subj = {
                        "pred_indices": val_preds,
                        "pred_probs": val_probs,
                        "true_indices": val_labels,
                    }
                else:
                    epochs_no_improve += 1
                    should_stop = epochs_no_improve > args.patience

            should_stop = broadcast_early_stop(should_stop, device=device)
            if should_stop:
                print_rank0(
                    f"Early stopping at epoch {epoch}; "
                    f"no improvement for more than {args.patience} epochs."
                )
                break

        if is_main_process():
            if not best_preds_for_subj:
                raise RuntimeError(
                    f"No validation predictions were recorded for subject {subj}. "
                    "Check whether --epochs is greater than 0."
                )

            print(
                f"Best epoch for subject {subj}: "
                f"{best_epoch} with Test Acc = {best_acc:.4f}"
            )

            subj_result_df = test_df.copy()
            subj_result_df["True_Label_Index"] = best_preds_for_subj["true_indices"]
            subj_result_df["Pred_Label_Index"] = best_preds_for_subj["pred_indices"]
            subj_result_df["Pred_Label_Name"] = [
                idx2label[idx] for idx in best_preds_for_subj["pred_indices"]
            ]
            subj_result_df["Pred_Confidence"] = best_preds_for_subj["pred_probs"]
            subj_result_df["Correct"] = (
                subj_result_df["True_Label_Index"] == subj_result_df["Pred_Label_Index"]
            )
            subj_result_df["Best_Epoch"] = best_epoch
            all_results.append(subj_result_df)

        if args.distributed:
            dist.barrier()

    if is_main_process():
        final_results_df = pd.concat(all_results, axis=0, ignore_index=True)
        overall_acc = final_results_df["Correct"].mean()
        print(f"\nOverall LOSO accuracy across all subjects: {overall_acc:.4f}")

        output_dir = os.path.dirname(args.output_excel)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        final_results_df.to_excel(args.output_excel, index=False)
        print(f"Saved LOSO prediction results to: {args.output_excel}")


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LOSO training script for SAC2-Net with optional DDP support."
    )

    parser.add_argument(
        "--dataset_annotation_path",
        type=str,
        required=True,
        help="Path to the Excel annotation file.",
    )
    parser.add_argument(
        "--au_prompt_templates_path",
        type=str,
        default="./utils/au_textual_prompt_templates.json",
        help="Path to the AU textual prompt template JSON file.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory of the processed dataset.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(CLASS_COUNTS.keys()),
        help="Benchmark name used for class-count configuration.",
    )
    parser.add_argument(
        "--output_excel",
        type=str,
        required=True,
        help="Path used to save LOSO prediction results.",
    )

    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per fold.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size per GPU/process. Effective batch size = batch_size * world_size.",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Base learning rate.")
    parser.add_argument("--wd", type=float, default=5e-2, help="Weight decay.")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warm-up epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=200,
        help="Early-stopping patience measured in epochs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers per process.",
    )
    parser.add_argument(
        "--pin_memory",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Use pinned memory in DataLoader.",
    )

    parser.add_argument("--weights", type=str, default="", help="Initial weights path.")
    parser.add_argument(
        "--load_sasa_only",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "If true, load only mag_encoder, flow_encoder, and sasa_head from --weights. "
            "Set this to false when loading a full model checkpoint."
        ),
    )
    parser.add_argument(
        "--freeze_layers",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Freeze mag_encoder, flow_encoder, and sasa_head before downstream training.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--deterministic",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable deterministic cuDNN behavior for reproducibility.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device used for single-process training. Ignored when launched by torchrun.",
    )

    parser.add_argument(
        "--dist_backend",
        default="nccl",
        help="Distributed backend. Use nccl for CUDA multi-GPU training.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        help="URL used to initialize distributed training.",
    )
    parser.add_argument(
        "--find_unused_parameters",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Set true if DDP reports unused parameters.",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    try:
        main(args)
    finally:
        cleanup_distributed()
