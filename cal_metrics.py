import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score


def confusionMatrix(gt, pred):
    """
    Calculate the F1-score, recall, and support for one class using
    one-vs-rest binary labels.

    Args:
        gt (list or np.ndarray): Binary ground-truth labels.
        pred (list or np.ndarray): Binary predicted labels.

    Returns:
        class_f1 (float): F1-score of the current class.
        class_recall (float): Recall of the current class.
        support (int): Number of ground-truth samples in the current class.
    """
    tn, fp, fn, tp = confusion_matrix(
        gt,
        pred,
        labels=[0, 1]
    ).ravel()

    f1_denominator = 2 * tp + fp + fn
    class_f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    support = tp + fn
    class_recall = (
        tp / support
        if support > 0
        else 0.0
    )

    return class_f1, class_recall, support


def recognition_evaluation(final_gt, final_pred, label_dict):
    """
    Calculate weighted F1, unweighted F1, and unweighted average recall.

    Args:
        final_gt (list or np.ndarray):
            Ground-truth class indices.
        final_pred (list or np.ndarray):
            Predicted class indices.
        label_dict (dict):
            Mapping from class names to class indices, for example:
            {"happiness": 0, "disgust": 1, "surprise": 2}.

    Returns:
        F1 (float): Support-weighted F1-score.
        UF1 (float): Unweighted mean of class-wise F1-scores.
        UAR (float): Unweighted mean of class-wise recalls.
    """
    final_gt = np.asarray(final_gt)
    final_pred = np.asarray(final_pred)

    if final_gt.size == 0 or final_pred.size == 0:
        return np.nan, np.nan, np.nan

    if final_gt.shape[0] != final_pred.shape[0]:
        raise ValueError(
            "final_gt and final_pred must contain the same number of samples."
        )

    class_f1_scores = []
    class_recalls = []
    class_supports = []

    for _, class_index in label_dict.items():
        gt_binary = (final_gt == class_index).astype(int)
        pred_binary = (final_pred == class_index).astype(int)

        class_f1, class_recall, support = confusionMatrix(
            gt_binary,
            pred_binary
        )

        # A class without ground-truth samples has undefined recall and
        # should not affect the macro-averaged metrics.
        if support > 0:
            class_f1_scores.append(class_f1)
            class_recalls.append(class_recall)
            class_supports.append(support)

    if not class_supports:
        return np.nan, np.nan, np.nan

    class_f1_scores = np.asarray(class_f1_scores, dtype=np.float64)
    class_recalls = np.asarray(class_recalls, dtype=np.float64)
    class_supports = np.asarray(class_supports, dtype=np.float64)

    # Support-weighted F1-score
    F1 = np.average(class_f1_scores, weights=class_supports)

    # Unweighted macro F1-score
    UF1 = np.mean(class_f1_scores)

    # Unweighted average recall
    UAR = np.mean(class_recalls)

    return F1, UF1, UAR


def cal_megc2019_cd_metrics(exp_result_path):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    df = pd.read_excel(exp_result_path)
    gt = df["True_Label_Index"].tolist()
    pred = df["Pred_Label_Index"].tolist()
    f1, uf1, uar = recognition_evaluation(gt, pred, label_dict)
    print("megc2019_cd: ")
    print(f"ACC: {accuracy_score(gt, pred)}, F1: {f1}, UF1: {uf1}, UAR: {uar}")

    for dataset in sorted(df["Dataset"].unique()):
        gt = df[df["Dataset"] == dataset]["True_Label_Index"].tolist()
        pred = df[df["Dataset"] == dataset]["Pred_Label_Index"].tolist()
        f1, uf1, uar = recognition_evaluation(gt, pred, label_dict)
        print(dataset)
        print(f"ACC: {accuracy_score(gt, pred)}, F1: {f1}, UF1: {uf1}, UAR: {uar}")


def cal_cross_dataset_metrics(exp_result_path):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    df = pd.read_excel(exp_result_path)
    gt = df["True_Label_Index"].tolist()
    pred = df["Pred_Label_Index"].tolist()
    f1, uf1, uar = recognition_evaluation(gt, pred, label_dict)
    print("smic: ")
    print(f"ACC: {accuracy_score(gt, pred)}, F1: {f1}, UF1: {uf1}, UAR: {uar}")


def cal_casme2_5cls_metrics(exp_result_path):
    label_dict = {'disgust': 0, 'happiness': 1, 'others': 2, 'repression': 3, 'surprise': 4}
    df = pd.read_excel(exp_result_path)
    gt = df["True_Label_Index"].tolist()
    pred = df["Pred_Label_Index"].tolist()
    f1, uf1, uar = recognition_evaluation(gt, pred, label_dict)
    print("CASME2_5_cls: ")
    print(f"ACC: {accuracy_score(gt, pred)}, F1: {f1}, UF1: {uf1}, UAR: {uar}")


def cal_samm_5cls_metrics(exp_result_path):
    label_dict = {'Anger': 0, 'Contempt': 1, 'Happiness': 2, 'Other': 3, 'Surprise': 4}
    df = pd.read_excel(exp_result_path)
    gt = df["True_Label_Index"].tolist()
    pred = df["Pred_Label_Index"].tolist()
    f1, uf1, uar = recognition_evaluation(gt, pred, label_dict)
    print("SAMM dataset")
    print(f"ACC: {accuracy_score(gt, pred)}, F1: {f1}, UF1: {uf1}, UAR: {uar}")


def cal_dfme_metrics(exp_result_path):
    label_dict = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}
    df = pd.read_excel(exp_result_path)
    gt = df["True_Label_Index"].tolist()
    pred = df["Pred_Label_Index"].tolist()
    f1, uf1, uar = recognition_evaluation(gt, pred, label_dict)
    print("dfme test dataset: ")
    print(f"ACC: {accuracy_score(gt, pred)}, F1: {f1}, UF1: {uf1}, UAR: {uar}")


def cal_casme_cube_metrics(benchmark, exp_result_path):
    if benchmark == "casme_cube_4cls":
        label_dict = {'negative': 0, 'others': 1, 'positive': 2, 'surprise': 3}
    elif benchmark == "casme_cube_7cls":
        label_dict = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'others': 4, 'sad': 5, 'surprise': 6}
    else:
        raise ValueError("benchmark should be either casme_cube_4cls or casme_cube_7cls.")
    df = pd.read_excel(exp_result_path)
    gt = df["True_Label_Index"].tolist()
    pred = df["Pred_Label_Index"].tolist()
    f1, uf1, uar = recognition_evaluation(gt, pred, label_dict)
    print(f"{benchmark}:")
    print(f"ACC: {accuracy_score(gt, pred)}, F1: {f1}, UF1: {uf1}, UAR: {uar}")


def main(args):
    if args.benchmark == "megc2019_cd":
        cal_megc2019_cd_metrics(args.exp_result_path)
    elif args.benchmark == "casme2_5cls":
        cal_casme2_5cls_metrics(args.exp_result_path)
    elif args.benchmark == "samm_5cls":
        cal_samm_5cls_metrics(args.exp_result_path)
    elif args.benchmark == "dfme":
        cal_dfme_metrics(args.exp_result_path)
    elif args.benchmark == "casme_cube_7cls" or args.benchmark == "casme_cube_4cls":
        cal_casme_cube_metrics(args.benchmark, args.exp_result_path)
    elif args.benchmark == "cross_dataset":
        cal_cross_dataset_metrics(args.exp_result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate metrics. (Acc, F1-score, UF1, UAR)")
    parser.add_argument(
        "--exp_result_path",
        type=str,
        default=r"",
        required=True,
        help="Path to the Excel file containing prediction results.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Name of benchmark to use.",
    )

    opt = parser.parse_args()

    main(opt)
