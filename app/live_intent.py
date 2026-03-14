import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

from app.models import LiveIntentResponse

if TYPE_CHECKING:
    from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_PATH = ROOT_DIR / "prompts" / "live_intent.system.txt"
DEFAULT_PROMPT_TEXT = """
You classify an in-progress user draft into a small fixed enum schema.
Return JSON only.
""".strip()

TASK_TYPES = {
    "explain",
    "analyze",
    "compare",
    "create",
    "persuade",
    "troubleshoot",
    "plan",
}
SOPHISTICATIONS = {"intro", "intermediate", "deep"}
AUDIENCES = {
    "general_public",
    "practitioner",
    "mid_management",
    "senior_management",
    "academic",
}
STAKE_LEVELS = {"low", "medium", "high"}
TIME_HORIZONS = {"immediate", "near_term", "strategic"}
VISIBLE_FIELDS = (
    "task_type",
    "sophistication",
    "audience",
    "stake_level",
    "time_horizon",
)
TASK_HINTS: Dict[str, tuple[str, ...]] = {
    "compare": (
        "compare",
        "versus",
        " vs ",
        "pros and cons",
        "tradeoff",
        "trade-off",
        "对比",
        "比较",
        "区别",
        "优缺点",
        "差异",
    ),
    "persuade": (
        "persuade",
        "convince",
        "memo",
        "pitch",
        "business case",
        "说服",
        "打动",
        "争取",
        "汇报材料",
        "立项",
    ),
    "troubleshoot": (
        "error",
        "failing",
        "failure",
        "bug",
        "debug",
        "fix",
        "broken",
        "issue",
        "incident",
        "502",
        "排查",
        "报错",
        "故障",
        "异常",
        "修复",
        "问题",
        "出错",
    ),
    "plan": (
        "plan",
        "roadmap",
        "strategy",
        "strategic",
        "next steps",
        "rollout",
        "migration plan",
        "规划",
        "计划",
        "路线图",
        "方案",
        "实施",
        "战略",
    ),
    "create": (
        "draft",
        "write",
        "create",
        "design",
        "build",
        "generate",
        "写",
        "起草",
        "设计",
        "生成",
        "做一份",
        "整理一份",
    ),
    "explain": (
        "explain",
        "what is",
        "why",
        "how does",
        "overview",
        "introduction",
        "解释",
        "为什么",
        "怎么",
        "介绍",
        "原理",
        "是什么",
    ),
    "analyze": (
        "analyze",
        "assess",
        "evaluate",
        "recommendation",
        "recommend",
        "implications",
        "decision",
        "分析",
        "评估",
        "判断",
        "建议",
        "决策",
    ),
}
SOPHISTICATION_HINTS: Dict[str, tuple[str, ...]] = {
    "intro": (
        "simple",
        "simply",
        "beginner",
        "basic",
        "introduction",
        "eli5",
        "入门",
        "基础",
        "简单",
        "小白",
        "通俗",
    ),
    "deep": (
        "deep",
        "detailed",
        "rigorous",
        "exhaustive",
        "comprehensive",
        "technical",
        "architecture",
        "benchmark",
        "academic",
        "深入",
        "详细",
        "严谨",
        "全面",
        "系统",
        "技术细节",
    ),
}
AUDIENCE_HINTS: Dict[str, tuple[str, ...]] = {
    "senior_management": (
        "ceo",
        "cto",
        "leadership",
        "executive",
        "senior management",
        "board",
        "exec team",
        "领导",
        "管理层",
        "高层",
        "老板",
    ),
    "mid_management": (
        "manager",
        "managers",
        "director",
        "directors",
        "team lead",
        "stakeholder",
        "经理",
        "主管",
        "负责人",
        "项目负责人",
    ),
    "academic": (
        "academic",
        "literature",
        "journal",
        "paper",
        "citation",
        "citations",
        "research community",
        "学术",
        "论文",
        "文献",
        "研究",
    ),
    "practitioner": (
        "engineer",
        "developer",
        "operator",
        "implementation",
        "architecture",
        "production",
        "deploy",
        "debugging",
        "工程师",
        "开发",
        "运维",
        "研发",
        "程序员",
        "架构师",
    ),
    "general_public": (
        "customer",
        "non-technical",
        "public",
        "general audience",
        "everyone",
        "普通人",
        "大众",
        "非技术",
        "小白用户",
    ),
}
STAKE_HINTS: Dict[str, tuple[str, ...]] = {
    "high": (
        "urgent",
        "asap",
        "critical",
        "board",
        "compliance",
        "risk",
        "production",
        "incident",
        "outage",
        "high stakes",
        "紧急",
        "高风险",
        "事故",
        "线上",
        "生产",
        "宕机",
        "故障",
    ),
    "medium": (
        "decide",
        "decision",
        "recommend",
        "migration",
        "launch",
        "plan",
        "persuade",
        "tradeoff",
        "决定",
        "决策",
        "建议",
        "迁移",
        "上线",
        "规划",
        "说服",
    ),
}
HORIZON_HINTS: Dict[str, tuple[str, ...]] = {
    "immediate": (
        "now",
        "today",
        "asap",
        "immediate",
        "right away",
        "fix",
        "incident",
        "urgent",
        "现在",
        "今天",
        "马上",
        "尽快",
        "立刻",
        "当前",
    ),
    "near_term": (
        "this week",
        "this month",
        "next sprint",
        "rollout",
        "migration",
        "near term",
        "launch",
        "本周",
        "这周",
        "近期",
        "这个月",
        "下周",
        "下个月",
    ),
    "strategic": (
        "next quarter",
        "quarter",
        "next year",
        "yearly",
        "long term",
        "strategic",
        "roadmap",
        "multi-year",
        "下季度",
        "明年",
        "长期",
        "战略",
        "年度",
        "路线图",
    ),
}
QUESTION_MARKS = {"?", "？"}
_CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]"
)
_LATIN_RE = re.compile(r"[A-Za-z]")


def _clean_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text or "").strip().split())
    return compact[: max(32, int(max_chars or 0))]


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
    return {}


def _contains_any(text: str, phrases: Iterable[str]) -> float:
    score = 0.0
    lowered = str(text or "").lower()
    for phrase in phrases:
        token = str(phrase or "").strip().lower()
        if not token:
            continue
        if token in lowered:
            score += 1.0 + (0.15 * min(2, max(0, len(token.split()) - 1)))
    return score


def _pick_label(scores: Dict[str, float], minimum: float) -> Optional[str]:
    best_label = ""
    best_score = 0.0
    for label, score in scores.items():
        if score > best_score:
            best_label = label
            best_score = float(score)
    if not best_label or best_score < minimum:
        return None
    return best_label


def _cjk_char_count(text: str) -> int:
    return len(_CJK_RE.findall(str(text or "")))


def _latin_char_count(text: str) -> int:
    return len(_LATIN_RE.findall(str(text or "")))


def _has_meaningful_cjk(text: str) -> bool:
    return _cjk_char_count(text) >= 2


class LiveIntentClassifier:
    def __init__(
        self,
        *,
        enabled: bool = True,
        min_chars_default: int = 12,
        min_chars_cjk: int = 6,
        min_chars: int | None = None,
        max_input_chars: int = 600,
        model: str = "gpt-4.1",
        prompt_path: str = "",
        confidence_threshold: float = 0.45,
    ) -> None:
        self.enabled = bool(enabled)
        default_chars = (
            int(min_chars)
            if min_chars is not None
            else int(min_chars_default)
        )
        self.min_chars_default = max(1, default_chars)
        self.min_chars_cjk = max(1, int(min_chars_cjk))
        self.max_input_chars = max(64, int(max_input_chars))
        self.model = str(model or "").strip()
        raw_prompt_path = (
            Path(str(prompt_path or "").strip())
            if prompt_path
            else DEFAULT_PROMPT_PATH
        )
        self.prompt_path = (
            raw_prompt_path
            if raw_prompt_path.is_absolute()
            else (ROOT_DIR / raw_prompt_path).resolve()
        )
        self.confidence_threshold = min(0.95, max(0.0, float(confidence_threshold)))
        self._prompt_cache: Optional[str] = None
        self._client: Optional[Any] = None

    def classify(self, text: str) -> LiveIntentResponse:
        if not self.enabled:
            return LiveIntentResponse()
        cleaned = _clean_text(text, self.max_input_chars)
        if self._signal_char_count(cleaned) < self._minimum_signal_chars(cleaned):
            return LiveIntentResponse()

        heuristic = self._heuristic_prediction(cleaned)
        model_prediction = self._model_prediction(cleaned, heuristic)
        merged = self._merge_predictions(heuristic, model_prediction)
        if merged.get("confidence") is None:
            merged["confidence"] = heuristic.get("confidence")
        return LiveIntentResponse(**merged)

    def _signal_char_count(self, text: str) -> int:
        return sum(1 for ch in str(text or "") if not ch.isspace())

    def _minimum_signal_chars(self, text: str) -> int:
        return self.min_chars_cjk if _has_meaningful_cjk(text) else self.min_chars_default

    def _heuristic_prediction(self, text: str) -> Dict[str, Any]:
        lowered = f" {text.lower()} "
        word_count = len([part for part in lowered.split() if part.strip()])
        cjk_count = _cjk_char_count(text)
        length_score = max(word_count, cjk_count)
        cjk_mode = _has_meaningful_cjk(text)

        task_scores = {
            label: _contains_any(lowered, hints)
            for label, hints in TASK_HINTS.items()
        }
        if text and text[-1] in QUESTION_MARKS:
            task_scores["explain"] += 0.35
            task_scores["analyze"] += 0.2
        if word_count >= 10 or cjk_count >= 10:
            task_scores["analyze"] += 0.15
        task_type = _pick_label(task_scores, 0.6) or "analyze"

        sophistication_scores = {
            "intro": _contains_any(lowered, SOPHISTICATION_HINTS["intro"]),
            "deep": _contains_any(lowered, SOPHISTICATION_HINTS["deep"]),
            "intermediate": 0.3 + (0.15 if length_score >= 12 else 0.0),
        }
        if task_type in {"compare", "troubleshoot", "plan", "persuade"}:
            sophistication_scores["intermediate"] += 0.25
        sophistication = _pick_label(sophistication_scores, 0.4) or "intermediate"

        audience_scores = {
            label: _contains_any(lowered, hints)
            for label, hints in AUDIENCE_HINTS.items()
        }
        if task_type == "troubleshoot":
            audience_scores["practitioner"] += 0.8
        if task_type == "explain" and sophistication == "intro":
            audience_scores["general_public"] += 0.55
        audience = _pick_label(audience_scores, 0.5)

        stake_scores = {
            "high": _contains_any(lowered, STAKE_HINTS["high"]),
            "medium": _contains_any(lowered, STAKE_HINTS["medium"]),
            "low": 0.3,
        }
        if task_type in {"persuade", "compare", "plan"}:
            stake_scores["medium"] += 0.35
        if task_type == "troubleshoot":
            stake_scores["high"] += 0.4
        stake_level = _pick_label(stake_scores, 0.35) or "low"

        horizon_scores = {
            label: _contains_any(lowered, hints)
            for label, hints in HORIZON_HINTS.items()
        }
        if task_type == "plan":
            horizon_scores["near_term"] += 0.35
            horizon_scores["strategic"] += 0.25
        time_horizon = _pick_label(horizon_scores, 0.35)

        field_scores = []
        for label, scores, minimum in (
            (task_type, task_scores, 0.6),
            (sophistication, sophistication_scores, 0.4),
            (audience, audience_scores, 0.5),
            (stake_level, stake_scores, 0.35),
            (time_horizon, horizon_scores, 0.35),
        ):
            if not label:
                continue
            field_scores.append(float(scores.get(label, minimum)))
        overall_confidence = min(
            0.9,
            0.38 + (sum(field_scores) / max(1.0, len(field_scores) * 3.8)),
        )

        prediction: Dict[str, Any] = {
            "task_type": task_type,
            "sophistication": sophistication,
            "audience": audience,
            "stake_level": stake_level,
            "time_horizon": time_horizon,
            "confidence": round(overall_confidence, 2),
            "source": "heuristic",
        }
        if overall_confidence < self.confidence_threshold:
            prediction["audience"] = None
            prediction["time_horizon"] = None
        if task_scores.get(task_type, 0.0) < 0.75 and not cjk_mode:
            prediction["task_type"] = None
        if audience and audience_scores.get(audience, 0.0) < 0.85:
            prediction["audience"] = None
        if time_horizon and horizon_scores.get(time_horizon, 0.0) < 0.75:
            prediction["time_horizon"] = None
        if cjk_mode and prediction.get("task_type") is None:
            prediction["task_type"] = task_type
        return prediction

    def _merge_predictions(
        self, heuristic: Dict[str, Any], model_prediction: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not isinstance(model_prediction, dict):
            return heuristic
        merged = dict(heuristic)
        used_model = False
        for field_name, allowed in (
            ("task_type", TASK_TYPES),
            ("sophistication", SOPHISTICATIONS),
            ("audience", AUDIENCES),
            ("stake_level", STAKE_LEVELS),
            ("time_horizon", TIME_HORIZONS),
        ):
            value = model_prediction.get(field_name)
            if isinstance(value, str) and value in allowed:
                merged[field_name] = value
                used_model = True
        model_confidence = model_prediction.get("confidence")
        if isinstance(model_confidence, (float, int)):
            merged["confidence"] = round(
                min(
                    1.0,
                    max(
                        float(heuristic.get("confidence") or 0.0),
                        float(model_confidence),
                    ),
                ),
                2,
            )
        if used_model:
            merged["source"] = "model"
        return merged

    def _model_prediction(
        self, text: str, heuristic: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not self.model or not os.getenv("OPENAI_API_KEY"):
            return None
        client = self._ensure_client()
        if client is None:
            return None
        try:
            rsp = client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self._prompt()},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "draft_text": text,
                                "heuristic_guess": {
                                    key: heuristic.get(key) for key in VISIBLE_FIELDS
                                },
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
            raw = rsp.choices[0].message.content if rsp and rsp.choices else "{}"
            parsed = _parse_json_object(raw or "{}")
            if parsed:
                parsed["source"] = "model"
            return parsed
        except Exception:
            return None

    def _ensure_client(self) -> Optional[Any]:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI

            self._client = OpenAI()
        except Exception:
            self._client = None
        return self._client

    def _prompt(self) -> str:
        if self._prompt_cache is not None:
            return self._prompt_cache
        path = self.prompt_path if self.prompt_path.exists() else DEFAULT_PROMPT_PATH
        if path.exists():
            self._prompt_cache = path.read_text(encoding="utf-8").strip()
        else:
            self._prompt_cache = DEFAULT_PROMPT_TEXT
        return self._prompt_cache
