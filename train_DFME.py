import os
import random
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.my_dataset import EmotionDataset
from models.SAC2Net import SAC2Net as create_model
from utils.utils import create_lr_scheduler, train_one_epoch, evaluate


# Set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(42)


# Label mapping: map emotion class to index
def create_label_mapping(df: pd.DataFrame, label_col: str = "Estimated Emotion"):
    labels = sorted(df[label_col].unique())
    label2idx = {lbl: i for i, lbl in enumerate(labels)}
    idx2label = {i: lbl for lbl, i in label2idx.items()}

    print(label2idx)
    print(idx2label)
    return label2idx, idx2label


# Build data transformation
def build_transforms(img_size: int = 224):
    train_transform = {
        "mag": transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46, 0.36, 0.33],
                std=[0.20, 0.17, 0.17]),
        ]),
        "flow": transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.87, 0.88, 0.90],
                std=[0.21, 0.19, 0.16]),
        ]),
    }

    val_transform = {
        "mag": transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46, 0.36, 0.33],
                std=[0.20, 0.17, 0.17]),
        ]),
        "flow": transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.87, 0.88, 0.90],
                std=[0.21, 0.19, 0.16]),
        ]),
    }

    return train_transform, val_transform


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Read Dataset Annotation Excel File
    assert os.path.exists(args.train_dataset_annotation_path), "Annotations file: {} not exist.".format(
        args.train_dataset_annotation_path)
    train_df = pd.read_excel(args.train_dataset_annotation_path)

    assert os.path.exists(args.val_dataset_annotation_path), "Annotations file: {} not exist.".format(
        args.val_dataset_annotation_path)
    val_df = pd.read_excel(args.val_dataset_annotation_path)

    required_cols = {"Subject", "Filename", "Apex", "Estimated Emotion"}
    if not required_cols.issubset(train_df.columns):
        missing = required_cols - set(train_df.columns)
        raise ValueError(
            f"Excel file is missing required columns: {missing}. "
            f"Expected at least these columns: {required_cols}"
        )

    label2idx, idx2label = create_label_mapping(train_df, label_col="Estimated Emotion")
    train_transform, val_transform = build_transforms(img_size=224)

    train_dataset = EmotionDataset(
        train_df, dataset_root=args.dataset_root, label2idx=label2idx, transform=train_transform
    )
    val_dataset = EmotionDataset(
        val_df, dataset_root=args.dataset_root, label2idx=label2idx, transform=val_transform
    )

    # Number of workers
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, args.num_workers])
    print(f"Using {num_workers} dataloader workers every process.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )

    num_classes = len(label2idx)
    class_counts = [161, 100, 548, 265, 206, 278, 298]  # DFME
    model = create_model(num_classes=num_classes).to(device)

    if args.weights != "":
        # Loading pretrained weights
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # delete the weights of cross-modal fusion module and classifier module, save sasa-related weights
        encoder_module_names = {'mag_encoder', 'flow_encoder', "sasa_head"}
        del_keys = []
        for name, param in model.named_parameters():
            if any(name.startswith(enc) for enc in encoder_module_names):
                continue
            else:
                del_keys.append(name)

        for k in del_keys:
            if k not in weights_dict:
                continue
            del weights_dict[k]

        # print(model.load_state_dict(weights_dict, strict=False))
        model.load_state_dict(weights_dict, strict=False)

    if args.freeze_layers:
        # Freeze the weights of feature encoders and sasa projection header, train fusion module'
        encoder_module_names = {'mag_encoder', 'flow_encoder', "sasa_head"}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if any(name.startswith(enc) for enc in encoder_module_names):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    pg = model.get_trainable_param_groups(lr=args.lr)
    optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=args.warmup_epochs)

    for epoch in range(args.epochs):
        train_loss, train_acc, logger = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            lr_scheduler=lr_scheduler,
            au_prompt_templates_path=args.au_prompt_templates_path,
            num_classes=num_classes,
            class_counts=class_counts
        )

        val_loss, val_acc, val_preds, val_probs, true_labels = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}"
        )

    # Build a result DataFrame
    result_df = val_df.copy()
    result_df["True_Label_Index"] = true_labels
    result_df["Pred_Label_Index"] = val_preds
    result_df["Pred_Label_Name"] = [
        idx2label[i] for i in val_preds
    ]
    result_df["Pred_Confidence"] = val_probs
    result_df["Correct"] = (
            result_df["True_Label_Index"] == result_df["Pred_Label_Index"]
    )

    # Optionally, compute overall metrics
    overall_acc = result_df["Correct"].mean()
    print(f"\nOverall accuracy: {overall_acc:.4f}")

    # Save to Excel
    result_df.to_excel(args.output_excel, index=False)
    print(f"Saved prediction results to: {args.output_excel}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFME benchmark training script (PyTorch)")
    parser.add_argument(
        "--train_dataset_annotation_path",
        type=str,
        required=True,
        help="Path to the Excel file containing annotations of train dataset.",
    )
    parser.add_argument(
        "--val_dataset_annotation_path",
        type=str,
        required=True,
        help="Path to the Excel file containing annotations of validation dataset.",
    )
    parser.add_argument(
        "--au_prompt_templates_path",
        type=str,
        default=r"./utils/au_textual_prompt_templates.json",
        help="Path to the AU textual prompt templates file.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Directory where all images are stored.",
    )
    parser.add_argument(
        "--output_excel",
        type=str,
        required=True,
        help="Path used to save prediction results Excel file.",
    )

    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Base learning rate.")
    parser.add_argument("--wd", type=float, default=5e-2, help="Weight decay.")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warm-up epochs.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of DataLoader worker processes.",
    )

    parser.add_argument("--weights", type=str, default="", help="Initial weights path.")
    parser.add_argument(
        "--freeze_layers",
        type=bool,
        default=False,
        help="Freeze layers before training.",
    )

    parser.add_argument(
        '--device',
        default='cuda:0',
        help='device id (i.e. 0 or 0,1 or cpu)'
    )

    opt = parser.parse_args()

    main(opt)
