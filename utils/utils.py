import os
import sys
import json
import math
import pickle
import random
from typing import List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.optimization import LossLogger, CombinedLoss
from utils.sasa_loss import SASALoss


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


# Dynamic loss scheduling
def get_loss_weights(epoch: int, total_epochs: int = 200, lambda_sum: float = 2.0):
    """
    Compute dynamic loss weights using cosine annealing.

    The sum lambda_cls + lambda_SASA is kept constant at `lambda_sum`.
    lambda_SASA follows a cosine decay from `lambda_sum` to 0, while
    lambda_cls follows the complementary ascent from 0 to `lambda_sum`.

    Args:
        epoch: Current epoch index (0-based)
        total_epochs: Total number of training epochs
        lambda_sum: Total budget shared by cls and SASA (default: 2.0)

    Returns:
        lambda_cls:  Classification loss weight at this epoch
        lambda_sasa: SASA alignment loss weight at this epoch

    Example:
        lc, ls = get_loss_weights(epoch=0, total_epochs=200)
        print(f"Start:  cls={lc:.3f}  sasa={ls:.3f}")
        Start:  cls=0.000  sasa=2.000

        lc, ls = get_loss_weights(epoch=100, total_epochs=200)
        print(f"Middle: cls={lc:.3f}  sasa={ls:.3f}")
        Middle: cls=1.000  sasa=1.000

        lc, ls = get_loss_weights(epoch=199, total_epochs=200)
        print(f"End:    cls={lc:.3f}  sasa={ls:.3f}")
        End:    cls=1.9999...  sasa=0.0000...
    """
    # Clamp epoch to [0, total_epochs - 1] to handle edge cases
    t = max(0, min(epoch, total_epochs - 1))

    # Progress through training: [0, 1]
    progress = t / max(1, total_epochs - 1)

    # Cosine decay from 1 -> 0 (used for SASA weight)
    # cos_factor = 1 at progress=0, cos_factor = 0 at progress=1
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    # SASA weight: starts at lambda_sum, decays to 0
    lambda_sasa = lambda_sum * cos_factor

    # Classification weight: starts at 0, grows to lambda_sum
    lambda_cls = lambda_sum - lambda_sasa

    return lambda_cls, lambda_sasa


def train_one_epoch(
        model,
        optimizer,
        data_loader,
        device,
        epoch, total_epochs,
        lr_scheduler,
        au_prompt_templates_path,
        num_classes, class_counts=None,
        max_grad_norm=1.0,
):
    """
    Train for one epoch.

    Args:
        model: MultimodalFERModel
        optimizer: Optimizer
        data_loader: Training dataloader yielding
                    (mag_images, flow_images, texts, labels, au_labels)
        device: Device string
        epoch: Current epoch
        total_epochs: Total number of training epochs
        lr_scheduler: Optional LR scheduler (step per batch)
        au_prompt_templates_path: Path where au_prompts templates are stored
        num_classes: Number of classes
        class_counts: List of class counts
        max_grad_norm: Gradient clipping norm (stabilizes training)
    Returns:
        loss accumulation: Loss accumulated over all batches
        accuracy: Training accuracy for the epoch
        loss_logger: LossLogger with per-batch loss history
    """
    model.train()

    sasa_loss_fn = SASALoss(temperature=0.07, use_hierarchical=True)
    lc, ls = get_loss_weights(epoch=epoch, total_epochs=total_epochs)
    criterion = CombinedLoss(num_classes=num_classes, class_counts=class_counts,
                             lambda_cls=lc, lambda_sasa=ls)  # CombinedLoss instance
    logger = LossLogger()

    accu_loss = torch.zeros(1).to(device)  # Accumulated losses
    correct = torch.zeros(1).to(device)  # Cumulative number of correctly predicted samples
    total = 0

    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        mag_imgs, flow_imgs, au_labels, labels = data
        au_prompts = generate_prompts(au_prompt_templates_path, au_labels)

        # Forward pass (training mode: returns SASA features + logits + reliability)
        outputs = model(mag_imgs.to(device), flow_imgs.to(device), texts=au_prompts)

        # Compute SASA loss externally (needs au_labels)
        sasa_loss, _ = sasa_loss_fn(
            outputs['f_mag_proj'],
            outputs['f_flow_proj'],
            outputs['f_text_proj'],
            au_labels,
        )

        # Compute combined loss
        total_loss, loss_dict = criterion(outputs, labels.to(device), sasa_loss=sasa_loss)

        # Backward pass
        total_loss.backward()
        accu_loss += total_loss.detach()

        # Gradient clipping (important for stability with multiple losses)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Track accuracy
        pred_classes = torch.max(outputs['logits'], dim=1)[1]
        correct += torch.eq(pred_classes, labels.to(device)).sum()
        total += labels.size(0)

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            correct.item() / total,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(total_loss):
            print('WARNING: non-finite loss, ending training ', total_loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()
        # Log losses
        logger.update(loss_dict)

    return accu_loss.item() / (step + 1), correct.item() / total, logger


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    """
    Evaluate model on test/validation set.

    Note: Text is NOT needed during evaluation. The model in eval mode
    only requires visual inputs and returns logits.

    Args:
        model: MultimodalFERModel
        data_loader: Test dataloader yielding (mag_images, flow_images, labels)
        device: Device string
        epoch: Current epoch

    Returns:
        loss accumulation: Loss accumulated over all batches
        accuracy: Test accuracy for the epoch
        all_preds: Predicted labels from model on test set
        all_probs: Predicted probabilities from model on test set
        all_labels: Ground truth labels
    """
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_loss = torch.zeros(1).to(device)  # Accumulated loss
    correct = torch.zeros(1).to(device)
    total = 0

    all_preds: List[int] = []  # Predicted label
    all_probs: List[float] = []  # Predicted Probability
    all_labels: List[int] = []  # True label

    softmax = nn.Softmax(dim=1)
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        mag_imgs, flow_imgs, au_labels, labels = data

        # Eval mode: no texts needed, returns logits only
        outputs = model(mag_imgs.to(device), flow_imgs.to(device))
        probs = softmax(outputs['logits'])
        confs, pred_classes = torch.max(probs, dim=1)

        correct += torch.eq(pred_classes, labels.to(device)).sum()
        total += labels.size(0)
        loss = loss_function(outputs['logits'], labels.to(device))
        accu_loss += loss

        all_preds.extend(pred_classes.cpu().tolist())
        all_probs.extend(confs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            correct.item() / total
        )

    return accu_loss.item() / (step + 1), correct.item() / total, all_preds, all_probs, all_labels


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup: bool = True,
                        warmup_epochs: int = 1,
                        warmup_factor: float = 1e-3,
                        end_factor: float = 1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        The program returns a learning rate scaling factor based on the number of steps.
        Note that PyTorch calls the `lr_scheduler.step()` method once before training begins.
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # During warmup, the lr factor changes from warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # After warmup, the lr factor changes from 1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # Record the weight parameters to be trained by optimize
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # Record the corresponding weight name
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
