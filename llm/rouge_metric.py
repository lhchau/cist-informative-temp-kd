"""ROUGE-L and exact match against one or more references (Super-NaturalInstructions evaluation)."""
import string

from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_punctuation = set(string.punctuation)


def _normalize(text):
    """Lowercase, remove punctuation and collapse whitespace (SQuAD-style, articles kept)."""
    return " ".join("".join(ch for ch in text.lower() if ch not in _punctuation).split())


def compute_metrics(predictions, references):
    """predictions: list[str]; references: list[list[str]]. Each score is the max over references, in percent."""
    n = min(len(predictions), len(references))
    em = rouge_l = 0.0
    for pred, golds in zip(predictions[:n], references[:n]):
        em += max(float(_normalize(pred) == _normalize(gold)) for gold in golds)
        rouge_l += max(_scorer.score(target=gold, prediction=pred)["rougeL"].fmeasure for gold in golds)
    return {"exact_match": round(100.0 * em / n, 4), "rougeL": round(100.0 * rouge_l / n, 4)}
