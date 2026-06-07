import argparse
import json
import string
from typing import Dict, List, Union

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


CaptionType = Union[str, List[str]]


def load_json(path: str):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str, remove_punctuation: bool = False) -> str:
    """Normalize caption text."""
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())

    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    return text.lower()


def load_panda_gt(gt_path: str, cap_type: str = "short") -> Dict[str, List[str]]:
    """
    Load panda dataset annotations.

    Expected format:
    [
        {
            "video": "xxx.mp4",
            "short": ["A panda is eating bamboo."],
            ...
        }
    ]
    """
    data = load_json(gt_path)
    gt = {}

    for item in data:
        video_id = item["video"]
        captions = item[cap_type]

        if isinstance(captions, str):
            captions = [captions]

        gt[video_id] = captions

    return gt


def load_caption_dict(gt_path: str) -> Dict[str, List[str]]:
    """
    Load general caption annotations.

    Supported formats:
    {
        "video1": "caption",
        "video2": ["caption1", "caption2"]
    }
    """
    data = load_json(gt_path)
    gt = {}

    for key, value in data.items():
        if isinstance(value, str):
            gt[key] = [value]
        elif isinstance(value, list):
            gt[key] = value
        else:
            raise ValueError(f"Unsupported caption format for key: {key}")

    return gt


def load_predictions(pred_path: str) -> Dict[str, str]:
    """
    Load prediction file.

    Expected format:
    {
        "video1": "generated caption",
        "video2": "generated caption"
    }
    """
    data = load_json(pred_path)

    pred = {}
    for key, value in data.items():
        if isinstance(value, str):
            pred[key] = value
        elif isinstance(value, list) and len(value) > 0:
            pred[key] = value[0]
        else:
            raise ValueError(f"Unsupported prediction format for key: {key}")

    return pred


def align_data(
    gt: Dict[str, List[str]],
    pred: Dict[str, str],
    remove_punctuation: bool = False,
):
    """Align ground-truth captions and predictions by the same video ids."""
    keys = sorted([k for k in gt.keys() if k in pred])

    if len(keys) == 0:
        raise ValueError("No matched video ids between ground truth and predictions.")

    gt_aligned = {
        k: [normalize_text(c, remove_punctuation) for c in gt[k]]
        for k in keys
    }

    pred_aligned = {
        k: normalize_text(pred[k], remove_punctuation)
        for k in keys
    }

    return keys, gt_aligned, pred_aligned


def build_coco_format(gt: Dict[str, List[str]], pred: Dict[str, str]):
    """Convert captions to pycocoevalcap format."""
    gts = {}
    res = {}

    for key in gt:
        gts[key] = [{"caption": cap} for cap in gt[key]]
        res[key] = [{"caption": pred[key]}]

    return gts, res


def compute_cider(gts, res) -> float:
    tokenizer = PTBTokenizer()
    cider = Cider()

    gts_tokenized = tokenizer.tokenize(gts)
    res_tokenized = tokenizer.tokenize(res)

    score, _ = cider.compute_score(gts_tokenized, res_tokenized)
    return score


def compute_meteor(gts, res) -> float:
    meteor = Meteor()
    score, _ = meteor.compute_score(gts, res)
    return score


def compute_bleu4(gt: Dict[str, List[str]], pred: Dict[str, str]) -> float:
    smoothing = SmoothingFunction().method4
    total_score = 0.0

    for key in gt:
        references = [cap.split() for cap in gt[key]]
        hypothesis = pred[key].split()

        score = sentence_bleu(
            references,
            hypothesis,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        )
        total_score += score

    return total_score / len(gt)


def compute_rouge_l(gt: Dict[str, List[str]], pred: Dict[str, str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total_score = 0.0

    for key in gt:
        scores = [
            scorer.score(ref, pred[key])["rougeL"].fmeasure
            for ref in gt[key]
        ]

        # For multiple references, use the best-matching reference.
        total_score += max(scores)

    return total_score / len(gt)


def evaluate(gt: Dict[str, List[str]], pred: Dict[str, str]):
    """Evaluate generated captions."""
    gts, res = build_coco_format(gt, pred)

    cider = compute_cider(gts, res)
    bleu4 = compute_bleu4(gt, pred)
    rouge_l = compute_rouge_l(gt, pred)
    meteor = compute_meteor(gts, res)

    return {
        "CIDEr": cider,
        "BLEU-4": bleu4,
        "ROUGE-L": rouge_l,
        "METEOR": meteor,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt", type=str, required=True, help="Path to ground-truth caption JSON file.")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction JSON file.")
    parser.add_argument("--dataset", type=str, default="panda", choices=["panda", "general"])
    parser.add_argument("--cap_type", type=str, default="short", help="Caption type for panda dataset.")
    parser.add_argument(
        "--remove_punctuation",
        action="store_true",
        help="Whether to remove punctuation before evaluation.",
    )

    args = parser.parse_args()

    if args.dataset == "panda":
        gt = load_panda_gt(args.gt, args.cap_type)
    else:
        gt = load_caption_dict(args.gt)

    pred = load_predictions(args.pred)

    keys, gt_aligned, pred_aligned = align_data(
        gt,
        pred,
        remove_punctuation=args.remove_punctuation,
    )

    print(f"Number of evaluated samples: {len(keys)}")

    scores = evaluate(gt_aligned, pred_aligned)

    print("Evaluation results:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")

    print("\nLaTeX/Table format:")
    print(
        f"{scores['CIDEr']:.4f}\t"
        f"{scores['ROUGE-L']:.4f}\t"
        f"{scores['BLEU-4']:.4f}\t"
        f"{scores['METEOR']:.4f}"
    )


if __name__ == "__main__":
    main()