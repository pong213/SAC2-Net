"""This script is used for SASA pre-training using external datasets. During this stage, only the HFE visual encoders
and SASA-related projection layers are optimized, while the subsequent fusion module and classifier are excluded. The
pretrained weights are then transferred to downstream MER datasets for task-specific training.

"""


import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.my_dataset import EmotionDataset
from models.SAC2Net_without_CCF import SAC2Net_without_CCF as create_model
from utils.utils import create_lr_scheduler
from utils.sasa_loss import SASALoss

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


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

    # print(label2idx)
    # print(idx2label)
    return label2idx, idx2label


# Generate AU textual prompts from activated AU list
def generate_prompts(template_path: str, active_au_list: list):
    """
    active_au_list: list of integers, e.g., [["1", "4", "15"], ["4", "7"]]
    """
    assert os.path.exists(template_path), "AU textual prompts templates file {} does not exist".format(template_path)
    with open(template_path, "r") as f:
        au_nlp_desc = json.load(f)

    if not active_au_list:
        return "A neutral face with no distinct muscle movement."

    au_prompts = []
    for aus in active_au_list:
        # Step 1: Collect descriptions for active AUs
        descriptions = []
        for au_idx in aus:
            if au_idx in au_nlp_desc:
                # RANDOM SELECTION for robust learning
                desc = random.choice(au_nlp_desc[au_idx])
                descriptions.append(desc)
            else:
                descriptions.append("")

        # Step 2: Join them grammatically
        if len(descriptions) == 1:
            joined_desc = descriptions[0]
        else:
            # "A, B, and C"
            joined_desc = ", ".join(descriptions[:-1]) + ", and " + descriptions[-1]

        # Step 3: Wrap in a sentence template
        # RANDOM SELECTION of the wrapper
        template = random.choice(au_nlp_desc["base_templates"])
        final_sentence = template.format(joined_desc)

        au_prompts.append(final_sentence)

    return au_prompts


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
    assert os.path.exists(args.dataset_annotation_path), "Annotations file: {} not exist.".format(
        args.dataset_annotation_path)
    df = pd.read_excel(args.dataset_annotation_path, converters={u'Subject': str})

    label2idx, idx2label = create_label_mapping(df, label_col="Estimated Emotion")
    train_transform, val_transform = build_transforms(img_size=224)

    dataset = EmotionDataset(
        df, dataset_root=args.dataset_root, label2idx=label2idx, transform=train_transform
    )

    # Number of workers
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, args.num_workers])
    print(f"Using {num_workers} dataloader workers every process.")

    dataset_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = create_model().to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # delete the weights of cross-modal fusion module and classifier module,
        # save the weights of feature encoders and projection heads'
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

        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        # Freeze the weights of cross-modal fusion module and classifier module, train feature encoders'
        encoder_module_names = {'mag_encoder', 'flow_encoder', "sasa_head"}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if any(name.startswith(enc) for enc in encoder_module_names):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(dataset_loader), args.epochs,
                                       warmup=True, warmup_epochs=args.warmup_epochs)

    # Train SASA
    for epoch in range(args.epochs):
        model.train()
        sasa_loss_function = SASALoss(temperature=0.07, use_hierarchical=True)
        sasa_loss = torch.zeros(1).to(device)
        optimizer.zero_grad()

        data_loader = tqdm(dataset_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            mag_imgs, flow_imgs, au_labels, label_idxes = data
            au_prompts = generate_prompts(args.au_prompt_templates_path, au_labels)

            outputs_train = model(mag_imgs.to(device), flow_imgs.to(device), texts=au_prompts)
            loss, loss_dict = sasa_loss_function(
                outputs_train["f_mag_proj"],
                outputs_train["f_flow_proj"],
                outputs_train["f_text_proj"],
                au_labels
            )
            loss.backward()
            sasa_loss += loss.detach()

            data_loader.desc = "[train epoch {}] loss: {:.3f}, lr: {:.5f}".format(
                epoch,
                sasa_loss.item() / (step + 1),
                optimizer.param_groups[0]["lr"]
            )

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()
            # update lr
            lr_scheduler.step()

    torch.save(model.state_dict(), args.output_weights_path)
    print(f"Saved the sasa pretrained weights to {args.output_weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic anchoring soft alignment pre-training script (PyTorch)")
    parser.add_argument(
        "--dataset_annotation_path",
        type=str,
        default=r"../datasets/annotation_files/ck_plus.xlsx",
        required=True,
        help="Path to the Excel file containing annotations.",
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
        default=r"../datasets/ck_plus",
        required=True,
        help="Directory where all images are stored.",
    )
    parser.add_argument(
        "--output_weights_path",
        type=str,
        default=r"./pretrained_weights/sasa_pretrained_weights.pth",
        required=True,
        help="Path to save the weights dict.",
    )

    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=5e-2, help="Weight decay.")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warm-up epochs.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Number of DataLoader worker processes.",
    )

    parser.add_argument("--weights", type=str, default="", help="Initial weights path.")
    parser.add_argument(
        "--freeze_layers",
        type=bool,
        default=True,
        help="Freeze layers before training.",
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='device id (i.e. 0 or 0,1 or cpu)'
    )

    opt = parser.parse_args()

    main(opt)
