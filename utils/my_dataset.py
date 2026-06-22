import os
import pandas as pd
from PIL import Image
from typing import Dict

import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """
    Dataset that builds image filenames as:
        {Subject}_{Filename}_{Apex}.jpg
    and uses the "Estimated Emotion" column as label.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            dataset_root: str,
            label2idx: Dict[str, int],
            transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _build_filename(self, row: pd.Series) -> str:
        # dataset = str(row["Dataset"]).strip()   # For combination datasets
        subject = str(row["Subject"]).strip()
        fname = str(row["Filename"]).strip()
        apex = str(row["Apex"]).strip()

        filename = f"{subject}_{fname}_{apex}.jpg"
        # filename = f"{dataset}_{subject}_{fname}_{apex}.jpg"    # Combination dataset filename

        return filename

    def _ge_au_labels(self, row: pd.Series) -> list:
        aus = str(row["AU"]).split("+")
        au_labels = [au.strip().lower() for au in aus]

        return au_labels

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        au_labels = self._ge_au_labels(row)
        img_name = self._build_filename(row)

        mag_img_path = os.path.join(self.dataset_root, "magnification", img_name)
        flow_img_path = os.path.join(self.dataset_root, "decflow", img_name)

        if not os.path.exists(mag_img_path) or not os.path.exists(flow_img_path):
            raise FileNotFoundError(f"Magnified image and/or optical image not found: {mag_img_path}, {flow_img_path}")

        mag_img = Image.open(mag_img_path).convert("RGB")
        flow_img = Image.open(flow_img_path).convert("RGB")

        if self.transform is not None:
            mag_img = self.transform["mag"](mag_img)
            flow_img = self.transform["flow"](flow_img)

        label_name = row["Estimated Emotion"]
        label_idx = self.label2idx[label_name]

        return mag_img, flow_img, au_labels, label_idx

    @staticmethod
    def collate_fn(batch):
        mag_imgs, flow_imgs, au_labels_list, label_idxs = tuple(zip(*batch))

        mag_imgs = torch.stack(mag_imgs, dim=0)
        flow_imgs = torch.stack(flow_imgs, dim=0)
        label_idxs = torch.tensor(label_idxs, dtype=torch.long)

        # au_labels stays as a list of lists (variable length)
        # e.g., [["au1", "au2"], ["au4", "au6", "au12"], ...]

        return mag_imgs, flow_imgs, list(au_labels_list), label_idxs


def test():
    def create_label_mapping(df: pd.DataFrame, label_col: str = "Estimated Emotion"):
        labels = sorted(df[label_col].unique())
        label2idx = {lbl: i for i, lbl in enumerate(labels)}
        idx2label = {i: lbl for lbl, i in label2idx.items()}

        print(label2idx)
        print(idx2label)
        return label2idx, idx2label

    from torchvision import transforms
    transform = {
        "mag": transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            transforms.RandomGrayscale(p=0.2),
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

    dataset_root = "../../datasets/CASME^3"
    dataset_annotation_path = "../../../datasets/annotation_files/cleaned_CASME^3.xlsx"
    dataset_df = pd.read_excel(dataset_annotation_path)

    label2idx, idx2label = create_label_mapping(dataset_df, label_col="Estimated Emotion")

    dataset = EmotionDataset(
        dataset_df, dataset_root=dataset_root, label2idx=label2idx, transform=transform
    )

    import matplotlib.pyplot as plt
    mag_img, flow_img, au_labels, label_idx = dataset[51]
    print(au_labels)
    plt.imshow(mag_img.numpy().transpose((1, 2, 0)))
    plt.imshow(flow_img.numpy().transpose((1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    test()
