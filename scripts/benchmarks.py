#!/usr/bin/env python3
"""Benchmark abstraction layer for AdaThink experiments.

Supports GSM8K, MATH-500, and BBH with unified interfaces for:
- Dataset loading and field mapping
- Gold answer extraction
- Prediction parsing
- Correctness checking
- Prompt construction
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Shared regex patterns
# ---------------------------------------------------------------------------
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer\s*[:：]|the answer is\s*)([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}")
OPTION_RE = re.compile(r"\(?([A-Za-z])\)")
LATEX_FRAC_RE = re.compile(r"\\(?:d?frac)\{([^}]+)\}\{([^}]+)\}")
LATEX_SQRT_RE = re.compile(r"\\sqrt\{([^}]+)\}")

# BBH subtasks that use multiple-choice format
BBH_MC_TASKS = {
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "formal_fallacies", "geometric_shapes",
    "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "navigate", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection",
    "snarks", "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies",
}

BBH_FREEFORM_TASKS = {
    "dyck_languages", "multistep_arithmetic_two",
    "object_counting", "word_sorting",
}

BBH_ALL_TASKS = BBH_MC_TASKS | BBH_FREEFORM_TASKS


# ---------------------------------------------------------------------------
# Number extraction helpers (shared across benchmarks)
# ---------------------------------------------------------------------------

def extract_last_number(text: str) -> Optional[str]:
    if not text:
        return None
    matches = NUM_RE.findall(text)
    return matches[-1] if matches else None


def extract_final_number(text: str) -> Optional[str]:
    if not text:
        return None
    matches = list(FINAL_ANSWER_RE.finditer(text))
    return matches[-1].group(1) if matches else None


def extract_boxed(text: str) -> Optional[str]:
    if not text:
        return None
    matches = list(BOXED_RE.finditer(text))
    return matches[-1].group(1).strip() if matches else None


def has_explicit_final(text: str) -> bool:
    return extract_final_number(text) is not None


def to_float(num_str: Optional[str]) -> Optional[float]:
    if num_str is None:
        return None
    s = num_str.replace(",", "").strip()
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                denom = float(parts[1])
                if denom == 0:
                    return None
                return float(parts[0]) / denom
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# LaTeX normalization for MATH answers
# ---------------------------------------------------------------------------

def normalize_latex(s: str) -> str:
    """Normalize a LaTeX math expression to a canonical string for comparison."""
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\\text{", "").replace("\\mathrm{", "").replace("\\mathbf{", "")
    s = s.replace("\\textbf{", "")
    s = s.replace("\\%", "%")
    s = s.replace("\\$", "$")
    while "  " in s:
        s = s.replace("  ", " ")

    # Handle \dfrac and \frac → a/b
    def frac_repl(m):
        return f"({m.group(1)})/({m.group(2)})"
    s = LATEX_FRAC_RE.sub(frac_repl, s)

    # Handle \sqrt{x} → sqrt(x)
    def sqrt_repl(m):
        return f"sqrt({m.group(1)})"
    s = LATEX_SQRT_RE.sub(sqrt_repl, s)

    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\pi", "pi").replace("\\infty", "inf")

    # Remove remaining single-char LaTeX commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Remove braces
    s = s.replace("{", "").replace("}", "")
    s = s.strip()
    return s


def _strip_variable_assignment(s: str) -> str:
    """Strip 'x = ' or 'y = ' prefix from answers like 'x=5' -> '5'."""
    m = re.match(r"^[a-zA-Z]\s*=\s*(.+)$", s.strip())
    return m.group(1).strip() if m else s.strip()


def math_answers_equiv(pred: str, gold: str) -> bool:
    """Check if two math answers are equivalent (for MATH benchmark)."""
    pred_norm = normalize_latex(pred)
    gold_norm = normalize_latex(gold)

    if pred_norm == gold_norm:
        return True

    # Strip variable assignments: "x = 5" -> "5"
    pred_val = _strip_variable_assignment(pred_norm)
    gold_val = _strip_variable_assignment(gold_norm)
    if pred_val == gold_val:
        return True

    # Try numeric comparison on raw and stripped forms
    for p, g in [(pred_norm, gold_norm), (pred_val, gold_val)]:
        pn = to_float(p)
        gn = to_float(g)
        if pn is not None and gn is not None:
            tol = 1e-4 * max(1.0, abs(gn))
            if abs(pn - gn) <= tol:
                return True

    # Handle fraction strings: "(1)/(4)" should match "0.25"
    for p, g in [(pred_val, gold_val), (pred_norm, gold_norm)]:
        pn = _try_eval_fraction(p)
        gn = _try_eval_fraction(g)
        if pn is not None and gn is not None:
            if abs(pn - gn) <= 1e-4 * max(1.0, abs(gn)):
                return True

    # Normalized string comparison (remove all whitespace and compare)
    pred_clean = re.sub(r"\s+", "", pred_val).lower()
    gold_clean = re.sub(r"\s+", "", gold_val).lower()
    return pred_clean == gold_clean


def _try_eval_fraction(s: str) -> float | None:
    """Try to evaluate simple fraction expressions like '(1)/(4)' or '3/7'."""
    s = s.strip().replace(" ", "")
    # Match patterns like (a)/(b) or a/b
    m = re.match(r"^\(?(-?[\d.]+)\)?\s*/\s*\(?(-?[\d.]+)\)?$", s)
    if m:
        try:
            denom = float(m.group(2))
            if denom == 0:
                return None
            return float(m.group(1)) / denom
        except ValueError:
            return None
    return to_float(s)


# ---------------------------------------------------------------------------
# Benchmark dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSample:
    """Unified sample representation across all benchmarks."""
    question: str
    gold: str
    meta: Dict[str, Any]


@dataclass
class BenchmarkConfig:
    """Configuration for a specific benchmark."""
    name: str
    answer_type: str  # "numeric", "latex", "mcq", "freeform"
    system_prompt: str
    system_prompt_direct: str


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_gsm8k(split: str = "test", **kwargs) -> List[BenchmarkSample]:
    ds = load_dataset("gsm8k", "main", split=split)
    samples = []
    for ex in ds:
        answer_field = ex["answer"]
        if "####" in answer_field:
            after = answer_field.split("####")[-1]
            match = NUM_RE.search(after)
            gold = match.group(0) if match else (extract_last_number(answer_field) or "")
        else:
            gold = extract_last_number(answer_field) or ""
        samples.append(BenchmarkSample(
            question=ex["question"],
            gold=gold,
            meta={"source": "gsm8k"},
        ))
    return samples


def load_math500(split: str = "test", **kwargs) -> List[BenchmarkSample]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    samples = []
    for ex in ds:
        gold = ex.get("answer", "")
        # Try extracting from \boxed{} if answer field itself contains it
        boxed = extract_boxed(gold)
        if boxed:
            gold = boxed
        samples.append(BenchmarkSample(
            question=ex["problem"],
            gold=gold.strip(),
            meta={
                "source": "math500",
                "subject": ex.get("subject", ""),
                "level": ex.get("level", ""),
            },
        ))
    return samples


def load_bbh(split: str = "test", task: str = "all", **kwargs) -> List[BenchmarkSample]:
    """Load BBH (BIG-Bench Hard) dataset.

    Args:
        split: Dataset split.
        task: Specific BBH subtask name or "all" for all tasks.
    """
    tasks = BBH_ALL_TASKS if task == "all" else {task}
    samples = []
    for t in sorted(tasks):
        try:
            ds = load_dataset("lukaemon/bbh", t, split=split)
        except Exception:
            try:
                ds = load_dataset("maveriq/bigbenchhard", t, split="train")
            except Exception:
                print(f"Warning: Could not load BBH task '{t}', skipping.")
                continue
        is_mc = t in BBH_MC_TASKS
        for ex in ds:
            question = ex.get("input", ex.get("question", ""))
            gold = ex.get("target", ex.get("answer", "")).strip()
            # For MC tasks, extract just the option letter from gold
            if is_mc:
                m = OPTION_RE.search(gold)
                if m:
                    gold = m.group(1).upper()
                else:
                    gold = gold.strip("() ").upper()
            samples.append(BenchmarkSample(
                question=question,
                gold=gold,
                meta={
                    "source": "bbh",
                    "task": t,
                    "is_mc": is_mc,
                },
            ))
    return samples


BENCHMARK_LOADERS = {
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "bbh": load_bbh,
}


def load_benchmark(name: str, **kwargs) -> List[BenchmarkSample]:
    loader = BENCHMARK_LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_LOADERS.keys())}")
    return loader(**kwargs)


# ---------------------------------------------------------------------------
# Prediction parsing (benchmark-aware)
# ---------------------------------------------------------------------------

def parse_prediction_gsm8k(
    text: str, strict_final_only: bool = False
) -> Tuple[Optional[str], bool, str]:
    """Parse a GSM8K prediction: extract numeric answer."""
    final = extract_final_number(text)
    if final is not None:
        return final, True, "final_marker"

    boxed = extract_boxed(text)
    if boxed is not None:
        val = extract_last_number(boxed)
        if val is not None:
            return val, False, "boxed"

    if strict_final_only:
        return None, False, "none"

    tail = extract_last_number(text)
    if tail is not None:
        return tail, False, "fallback_last"
    return None, False, "none"


def parse_prediction_math(
    text: str, strict_final_only: bool = False
) -> Tuple[Optional[str], bool, str]:
    """Parse a MATH prediction: extract LaTeX or numeric answer."""
    # For thinking models, focus on content after </think> if present
    think_end = text.rfind("</think>")
    search_text = text[think_end:] if think_end >= 0 else text

    # Priority 1: boxed answer (search full text, then post-think)
    for t in ([search_text, text] if think_end >= 0 else [text]):
        boxed = extract_boxed(t)
        if boxed is not None:
            return boxed, True, "boxed"

    # Priority 2: "Final answer:" or "the answer is" marker
    fa_re = re.compile(
        r"(?:final answer\s*[:：]|the answer is\s*)(.*?)(?:\.|$)",
        re.IGNORECASE | re.MULTILINE,
    )
    for t in ([search_text, text] if think_end >= 0 else [text]):
        matches = list(fa_re.finditer(t))
        if matches:
            ans = matches[-1].group(1).strip()
            if ans:
                return ans, True, "final_marker"

    if strict_final_only:
        return None, False, "none"

    # Fallback: last number from post-think content, then full text
    for t in ([search_text, text] if think_end >= 0 else [text]):
        num = extract_last_number(t)
        if num is not None:
            return num, False, "fallback_last"
    return None, False, "none"


def parse_prediction_bbh(
    text: str, is_mc: bool = True, strict_final_only: bool = False
) -> Tuple[Optional[str], bool, str]:
    """Parse a BBH prediction: extract option letter or free-form text."""
    text_lower = text.lower().strip()

    # Look for explicit answer markers
    for marker in ["the answer is", "final answer:", "answer:"]:
        idx = text_lower.rfind(marker)
        if idx != -1:
            tail = text[idx + len(marker):].strip()
            if is_mc:
                m = OPTION_RE.search(tail)
                if m:
                    return m.group(1).upper(), True, "final_marker"
                first_char = tail.strip("()[] ").upper()
                if len(first_char) >= 1 and first_char[0].isalpha():
                    return first_char[0], True, "final_marker"
            else:
                ans = tail.split("\n")[0].strip().rstrip(".")
                if ans:
                    return ans, True, "final_marker"

    if strict_final_only:
        return None, False, "none"

    # Fallback for MC: find last standalone option letter
    if is_mc:
        mc_matches = list(re.finditer(r"\b([A-Z])\b", text))
        if mc_matches:
            return mc_matches[-1].group(1), False, "fallback_last"

    # Fallback for freeform: last line content
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1].rstrip("."), False, "fallback_last"
    return None, False, "none"


def parse_prediction(
    text: str,
    benchmark: str,
    strict_final_only: bool = False,
    is_mc: bool = True,
) -> Tuple[Optional[str], bool, str]:
    """Unified prediction parser dispatching to benchmark-specific logic."""
    if benchmark in ("gsm8k",):
        return parse_prediction_gsm8k(text, strict_final_only)
    elif benchmark in ("math500", "math"):
        return parse_prediction_math(text, strict_final_only)
    elif benchmark in ("bbh",):
        return parse_prediction_bbh(text, is_mc=is_mc, strict_final_only=strict_final_only)
    else:
        return parse_prediction_gsm8k(text, strict_final_only)


# ---------------------------------------------------------------------------
# Correctness checking (benchmark-aware)
# ---------------------------------------------------------------------------

def is_correct_gsm8k(pred: Optional[str], gold: Optional[str], tol: float = 1e-6) -> bool:
    p = to_float(pred)
    g = to_float(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= tol * max(1.0, abs(g))


def is_correct_math(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    return math_answers_equiv(pred, gold)


def is_correct_bbh(pred: Optional[str], gold: Optional[str], is_mc: bool = True) -> bool:
    if pred is None or gold is None:
        return False
    if is_mc:
        return pred.strip().upper() == gold.strip().upper()
    pred_clean = re.sub(r"\s+", " ", pred.strip().lower())
    gold_clean = re.sub(r"\s+", " ", gold.strip().lower())
    return pred_clean == gold_clean


def is_correct(
    pred: Optional[str],
    gold: Optional[str],
    benchmark: str,
    is_mc: bool = True,
    tol: float = 1e-6,
) -> bool:
    """Unified correctness check dispatching to benchmark-specific logic."""
    if benchmark in ("gsm8k",):
        return is_correct_gsm8k(pred, gold, tol)
    elif benchmark in ("math500", "math"):
        return is_correct_math(pred, gold)
    elif benchmark in ("bbh",):
        return is_correct_bbh(pred, gold, is_mc=is_mc)
    else:
        return is_correct_gsm8k(pred, gold, tol)


# ---------------------------------------------------------------------------
# Prompt construction (benchmark-aware)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "gsm8k": (
        "You are a careful math solver. Solve the problem step by step briefly. "
        "End with a single line: Final answer: <number>."
    ),
    "gsm8k_direct": (
        "You are a careful math solver. "
        "Return only one line in this exact format: Final answer: <number>."
    ),
    "math500": (
        "You are an expert mathematician. Solve the following problem step by step. "
        "Put your final answer inside \\boxed{}."
    ),
    "math500_direct": (
        "You are an expert mathematician. "
        "Return only the final answer inside \\boxed{}."
    ),
    "bbh": (
        "You are a careful reasoner. Think through the problem step by step. "
        "End with a single line: The answer is <answer>."
    ),
    "bbh_direct": (
        "You are a careful reasoner. "
        "Return only one line in this exact format: The answer is <answer>."
    ),
}


def build_prompt(
    question: str,
    benchmark: str,
    tokenizer=None,
    prompt_format: str = "chat",
    direct_answer: bool = False,
    enable_thinking: Optional[bool] = False,
) -> str:
    """Build a prompt for the given benchmark."""
    key = f"{benchmark}_direct" if direct_answer else benchmark
    system_text = SYSTEM_PROMPTS.get(key, SYSTEM_PROMPTS.get(benchmark, SYSTEM_PROMPTS["gsm8k"]))

    if prompt_format == "chat" and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": question},
        ]
        chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            chat_kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(messages, **chat_kwargs)
        except TypeError:
            chat_kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **chat_kwargs)
        except Exception:
            pass

    return f"{system_text}\n\nQuestion: {question}\nSolution:\n"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_benchmark_config(name: str) -> BenchmarkConfig:
    """Get benchmark configuration."""
    configs = {
        "gsm8k": BenchmarkConfig(
            name="gsm8k",
            answer_type="numeric",
            system_prompt=SYSTEM_PROMPTS["gsm8k"],
            system_prompt_direct=SYSTEM_PROMPTS["gsm8k_direct"],
        ),
        "math500": BenchmarkConfig(
            name="math500",
            answer_type="latex",
            system_prompt=SYSTEM_PROMPTS["math500"],
            system_prompt_direct=SYSTEM_PROMPTS["math500_direct"],
        ),
        "bbh": BenchmarkConfig(
            name="bbh",
            answer_type="mcq",
            system_prompt=SYSTEM_PROMPTS["bbh"],
            system_prompt_direct=SYSTEM_PROMPTS["bbh_direct"],
        ),
    }
    if name not in configs:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(configs.keys())}")
    return configs[name]


def default_budgets(benchmark: str, enable_thinking: bool = False) -> List[int]:
    """Reasonable default token budgets per benchmark."""
    if enable_thinking:
        return {
            "gsm8k": [128, 256, 512],
            "math500": [512, 1024, 2048],
            "bbh": [256, 512, 1024],
        }.get(benchmark, [256, 512, 1024])
    return {
        "gsm8k": [64, 128, 256],
        "math500": [256, 512, 1024],
        "bbh": [128, 256, 512],
    }.get(benchmark, [128, 256, 512])
