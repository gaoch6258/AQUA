import math
import re


FORMAT_REWARD_WEIGHT = 0.1
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

QCLR_PROG_WEIGHT = 0.5
QCLR_SOFT_WEIGHT = 0.3
QCLR_QUERY_PENALTY = 0.05


def _extract_first_number(text: str) -> float | None:
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _tokenize_for_similarity(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower())


def compute_answer_similarity(predicted: str, true_answer: str) -> float:


    if not predicted or true_answer is None:
        return 0.0


    pred_val = _extract_first_number(predicted)
    true_val = _extract_first_number(str(true_answer))
    if pred_val is not None and true_val is not None:
        if true_val == 0:
            return 1.0 if pred_val == 0 else 0.0

        relative_error = abs(pred_val - true_val) / abs(true_val)
        return math.exp(-5 * relative_error)


    pred_lower = predicted.lower().strip()
    true_lower = str(true_answer).lower().strip()

    if not pred_lower or not true_lower:
        return 0.0


    if true_lower in pred_lower or pred_lower in true_lower:
        return 1.0


    true_tokens = set(_tokenize_for_similarity(true_lower))
    pred_tokens = set(_tokenize_for_similarity(pred_lower))
    if true_tokens:
        matches = len(true_tokens & pred_tokens)
        return matches / len(true_tokens)

    return 0.0


def reward_answer_accuracy(completions: list[str], **kwargs) -> list[float]:

    rewards = kwargs.get("answer_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_format_correctness(completions: list[str], **kwargs) -> list[float]:

    answers = kwargs.get("final_answer") or completions
    qa_options = kwargs.get("qa_options") or []
    rewards = []
    for idx, comp in enumerate(answers):
        score = 0.0
        text = "" if comp is None else str(comp).strip()
        options = qa_options[idx] if isinstance(qa_options, list) and idx < len(qa_options) else None
        if options:
            normalized = text.upper()
            option_keys = {str(opt).strip().upper() for opt in options}
            score = 1.0 if (normalized in option_keys and len(normalized) == 1) else 0.0
        else:
            score = 1.0 if _NUMBER_RE.search(text) else 0.0
        rewards.append(score * FORMAT_REWARD_WEIGHT)
    return rewards


def reward_qclr_progressive(completions: list[str], **kwargs) -> list[float]:

    values = kwargs.get("qclr_prog_reward") or []
    if not values:
        return [0.0] * len(completions)
    rewards = []
    for idx in range(len(completions)):
        raw = values[idx] if idx < len(values) else 0.0
        rewards.append(float(raw) * QCLR_PROG_WEIGHT)
    return rewards


def reward_qclr_terminal_soft(completions: list[str], **kwargs) -> list[float]:

    values = kwargs.get("qclr_soft_reward") or []
    if not values:
        return [0.0] * len(completions)
    rewards = []
    for idx in range(len(completions)):
        raw = values[idx] if idx < len(values) else 0.0
        rewards.append(float(raw) * QCLR_SOFT_WEIGHT)
    return rewards


def reward_qclr_query_penalty(completions: list[str], **kwargs) -> list[float]:

    values = kwargs.get("qclr_num_queries") or []
    if not values:
        return [0.0] * len(completions)
    rewards = []
    for idx in range(len(completions)):
        count = values[idx] if idx < len(values) else 0
        rewards.append(-float(count) * QCLR_QUERY_PENALTY)
    return rewards
