import argparse
import json
import os
import time
import textwrap
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from openai import OpenAI, RateLimitError

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SOURCE_POLICY_PATH = Path(__file__).resolve().parent / "source_policy.json"
SEARCH_POLICY_PATH = Path(__file__).resolve().parent / "search_policy.json"

DEFAULT_SEARCH_POLICY: Dict[str, Any] = {
    "cache_ttl_seconds": 3600,
    "max_broaden_steps": 2,
    "broaden_k_multipliers": [1, 2, 3],
    "min_new_fact_gain": 1,
    "max_no_gain_retries_per_intent": 2,
    "time_sensitive_terms": ["latest", "today", "current", "update", "new"],
    "allow_rerun_on_search_error": True,
    "intent_collapse_similarity": 0.55,
}


def load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_source_policy() -> Dict[str, List[str]]:
    if not SOURCE_POLICY_PATH.exists():
        raise FileNotFoundError(f"Missing source policy file: {SOURCE_POLICY_PATH}")
    raw = json.loads(SOURCE_POLICY_PATH.read_text(encoding="utf-8"))
    primary_tlds = [str(v).lower() for v in raw.get("primary_tlds", [])]
    primary_domain_suffixes = [
        str(v).lower() for v in raw.get("primary_domain_suffixes", [])
    ]
    secondary_tlds = [str(v).lower() for v in raw.get("secondary_tlds", [])]
    return {
        "primary_tlds": primary_tlds,
        "primary_domain_suffixes": primary_domain_suffixes,
        "secondary_tlds": secondary_tlds,
    }


def load_search_policy() -> Dict[str, Any]:
    policy = dict(DEFAULT_SEARCH_POLICY)
    if not SEARCH_POLICY_PATH.exists():
        return policy
    try:
        raw = json.loads(SEARCH_POLICY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return policy
    if not isinstance(raw, dict):
        return policy

    if isinstance(raw.get("cache_ttl_seconds"), int):
        policy["cache_ttl_seconds"] = max(0, int(raw["cache_ttl_seconds"]))
    if isinstance(raw.get("max_broaden_steps"), int):
        policy["max_broaden_steps"] = max(0, int(raw["max_broaden_steps"]))
    if isinstance(raw.get("min_new_fact_gain"), int):
        policy["min_new_fact_gain"] = max(0, int(raw["min_new_fact_gain"]))
    if isinstance(raw.get("max_no_gain_retries_per_intent"), int):
        policy["max_no_gain_retries_per_intent"] = max(
            0, int(raw["max_no_gain_retries_per_intent"])
        )
    if isinstance(raw.get("allow_rerun_on_search_error"), bool):
        policy["allow_rerun_on_search_error"] = raw["allow_rerun_on_search_error"]
    if isinstance(raw.get("intent_collapse_similarity"), (int, float)):
        v = float(raw["intent_collapse_similarity"])
        policy["intent_collapse_similarity"] = min(0.99, max(0.4, v))
    if isinstance(raw.get("broaden_k_multipliers"), list):
        vals = [int(v) for v in raw["broaden_k_multipliers"] if isinstance(v, int) and v > 0]
        if vals:
            policy["broaden_k_multipliers"] = vals
    if isinstance(raw.get("time_sensitive_terms"), list):
        terms = [str(v).strip().lower() for v in raw["time_sensitive_terms"] if str(v).strip()]
        if terms:
            policy["time_sensitive_terms"] = terms
    return policy


def matches_domain_rule(domain: str, rule: str) -> bool:
    """
    Domain matching with label-boundary safety.
    Supported rule formats:
    - "example.com" => exact domain only
    - "*.example.com" => example.com and all subdomains
    """
    d = (domain or "").strip(".").lower()
    r = (rule or "").strip(".").lower()
    if not d or not r:
        return False
    if r.startswith("*."):
        base = r[2:]
        return d == base or d.endswith("." + base)
    return d == r


def slugify_for_filename(text: str, max_len: int = 80) -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "-" for ch in text.strip())
    compact = "-".join(part for part in raw.split("-") if part)
    if not compact:
        compact = "research-task"
    return compact[:max_len].strip("-") or "research-task"


def is_valid_absolute_http_url(url: str) -> bool:
    candidate = (url or "").strip()
    if not candidate:
        return False
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    return True


def as_clean_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


SYSTEM_DECOMPOSE = load_prompt("decompose.system.txt")
SYSTEM_QUERY_GEN = load_prompt("query_gen.system.txt")
SYSTEM_SYNTHESIZE = load_prompt("synthesize.system.txt")
SYSTEM_SUFFICIENCY = load_prompt("sufficiency.system.txt")
SYSTEM_REPORT = load_prompt("report.system.txt")


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


@dataclass
class EvaluatedResult:
    title: str
    snippet: str
    url: str
    quality_tier: str
    quality_score: int


@dataclass
class SubQuestionFinding:
    sub_question: str
    summaries: List[str] = field(default_factory=list)
    facts: List[Dict[str, Any]] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)


class UsageTracker:
    def __init__(
        self,
        enabled: bool,
        cost_enabled: bool,
        pricing_source: str,
        pricing_models: Dict[str, Dict[str, float]],
        pricing_default: Dict[str, float],
    ) -> None:
        self.enabled = enabled
        self.cost_enabled = cost_enabled
        self.pricing_source = pricing_source
        self.pricing_models = pricing_models
        self.pricing_default = pricing_default
        self.events: List[Dict[str, Any]] = []

    def record(
        self,
        stage: str,
        provider: str,
        model: str,
        usage: Any,
        attempt: int,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        input_tokens, output_tokens, total_tokens, usage_missing = self._extract_usage(
            usage
        )
        cost = self._estimate_cost(model, input_tokens, output_tokens)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "provider": provider,
            "model": model,
            "attempt": attempt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(cost, 8),
            "metadata": metadata or {},
        }
        if usage_missing:
            event["metadata"]["usage_missing"] = True
        if self.cost_enabled and self._model_missing_in_pricing(model):
            event["metadata"]["pricing_fallback"] = True
        self.events.append(event)

    def _extract_usage(self, usage: Any) -> tuple[int, int, int, bool]:
        if usage is None:
            return 0, 0, 0, True

        def pick_int(obj: Any, keys: List[str]) -> int:
            for key in keys:
                val = None
                if isinstance(obj, dict):
                    val = obj.get(key)
                else:
                    val = getattr(obj, key, None)
                if isinstance(val, int):
                    return val
            return 0

        input_tokens = pick_int(usage, ["prompt_tokens", "input_tokens"])
        output_tokens = pick_int(usage, ["completion_tokens", "output_tokens"])
        total_tokens = pick_int(usage, ["total_tokens"])
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens, False

    def _model_missing_in_pricing(self, model: str) -> bool:
        if not self.cost_enabled:
            return False
        return model not in self.pricing_models

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if not self.cost_enabled:
            return 0.0
        rates = self.pricing_models.get(model, self.pricing_default)
        in_rate = float(rates.get("input_per_1m", 0.0))
        out_rate = float(rates.get("output_per_1m", 0.0))
        return (input_tokens / 1_000_000.0) * in_rate + (
            output_tokens / 1_000_000.0
        ) * out_rate

    def to_dict(self) -> Dict[str, Any]:
        by_stage: Dict[str, Dict[str, float]] = {}
        by_model: Dict[str, Dict[str, float]] = {}
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "calls": len(self.events),
        }
        for e in self.events:
            stage = e["stage"]
            model = e["model"]
            for bucket, key in ((by_stage, stage), (by_model, model)):
                if key not in bucket:
                    bucket[key] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost_usd": 0.0,
                        "calls": 0,
                    }
                bucket[key]["input_tokens"] += e["input_tokens"]
                bucket[key]["output_tokens"] += e["output_tokens"]
                bucket[key]["total_tokens"] += e["total_tokens"]
                bucket[key]["estimated_cost_usd"] += e["estimated_cost_usd"]
                bucket[key]["calls"] += 1
            totals["input_tokens"] += e["input_tokens"]
            totals["output_tokens"] += e["output_tokens"]
            totals["total_tokens"] += e["total_tokens"]
            totals["estimated_cost_usd"] += e["estimated_cost_usd"]

        totals["estimated_cost_usd"] = round(totals["estimated_cost_usd"], 8)
        for bucket in (by_stage, by_model):
            for key in bucket:
                bucket[key]["estimated_cost_usd"] = round(
                    bucket[key]["estimated_cost_usd"], 8
                )

        return {
            "enabled": self.enabled,
            "cost_enabled": self.cost_enabled,
            "pricing_source": self.pricing_source,
            "events": self.events,
            "by_stage": by_stage,
            "by_model": by_model,
            "total": totals,
        }

    def load_from_dict(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        events = data.get("events", [])
        if isinstance(events, list):
            self.events = [e for e in events if isinstance(e, dict)]


def load_pricing_config(path_str: str) -> tuple[Dict[str, Dict[str, float]], Dict[str, float], str]:
    path = Path(path_str)
    if not path.exists():
        return {}, {"input_per_1m": 0.0, "output_per_1m": 0.0}, f"{path} (missing)"
    raw = json.loads(path.read_text(encoding="utf-8"))
    models_raw = raw.get("models", {}) if isinstance(raw, dict) else {}
    default_raw = raw.get("default", {}) if isinstance(raw, dict) else {}
    models: Dict[str, Dict[str, float]] = {}
    if isinstance(models_raw, dict):
        for model, rates in models_raw.items():
            if not isinstance(rates, dict):
                continue
            models[str(model)] = {
                "input_per_1m": float(rates.get("input_per_1m", 0.0)),
                "output_per_1m": float(rates.get("output_per_1m", 0.0)),
            }
    default = {
        "input_per_1m": float(default_raw.get("input_per_1m", 0.0))
        if isinstance(default_raw, dict)
        else 0.0,
        "output_per_1m": float(default_raw.get("output_per_1m", 0.0))
        if isinstance(default_raw, dict)
        else 0.0,
    }
    return models, default, str(path)


class LLM:
    def __init__(self, model: str, usage_tracker: UsageTracker | None = None) -> None:
        self.client = OpenAI()
        self.model = model
        self.max_retries = 3
        self.usage_tracker = usage_tracker

    def json(
        self,
        system_prompt: str,
        user_prompt: str,
        stage: str = "unknown",
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        rsp = self._chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            stage=stage,
            metadata=metadata,
        )
        text = rsp.choices[0].message.content or "{}"
        return json.loads(text)

    def text(
        self,
        system_prompt: str,
        user_prompt: str,
        stage: str = "unknown",
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        rsp = self._chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stage=stage,
            metadata=metadata,
        )
        return rsp.choices[0].message.content or ""

    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        stage: str,
        metadata: Dict[str, Any] | None,
        **kwargs: Any,
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                rsp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
                if self.usage_tracker:
                    self.usage_tracker.record(
                        stage=stage,
                        provider="chat.completions",
                        model=self.model,
                        usage=getattr(rsp, "usage", None),
                        attempt=attempt + 1,
                        metadata=metadata,
                    )
                return rsp
            except RateLimitError as exc:
                last_exc = exc
                # Exponential backoff for transient rate-limit pressure.
                if attempt < self.max_retries - 1:
                    time.sleep(1.2 * (2**attempt))
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("Unexpected failure in chat completion.")


class WebSearch:
    def __init__(self, model: str, usage_tracker: UsageTracker | None = None) -> None:
        self.client = OpenAI()
        self.model = model
        self.usage_tracker = usage_tracker
        # Default to the current Responses API web search tool first,
        # with a compatibility fallback.
        tool_types = os.getenv(
            "OPENAI_WEB_SEARCH_TOOL_TYPES", "web_search_preview,web_search"
        )
        self.tool_types = [t.strip() for t in tool_types.split(",") if t.strip()]
        self.last_error = ""

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        prompt = textwrap.dedent(
            f"""
            Search the web for the query below and return STRICT JSON:
            {{
              "results": [
                {{"title": "...", "url": "https://...", "snippet": "..."}}
              ]
            }}
            Rules:
            - Return at most {k} results.
            - Include only results with valid absolute URLs.
            - Keep each snippet to 1-2 sentences.

            Query: {query}
            """
        ).strip()
        self.last_error = ""
        errors: List[str] = []
        for tool_type in self.tool_types:
            try:
                rsp = self.client.responses.create(
                    model=self.model,
                    tools=[{"type": tool_type}],
                    tool_choice={"type": tool_type},
                    input=prompt,
                )
                if self.usage_tracker:
                    self.usage_tracker.record(
                        stage="web_search",
                        provider="responses",
                        model=self.model,
                        usage=getattr(rsp, "usage", None),
                        attempt=1,
                        metadata={"query": query, "tool_type": tool_type},
                    )
                data = self._parse_results_json(rsp.output_text)
                items = data.get("results", [])[:k]
                return [
                    SearchResult(
                        title=i.get("title", ""),
                        snippet=i.get("snippet", ""),
                        url=i.get("url", "").strip(),
                    )
                    for i in items
                    if isinstance(i, dict)
                    and is_valid_absolute_http_url(i.get("url", ""))
                ]
            except Exception as exc:
                errors.append(f"{tool_type}: {exc}")
        self.last_error = " | ".join(errors) if errors else "unknown search error"
        # Fail-soft per query; caller decides whether failure rate is fatal.
        return []

    def _parse_results_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise ValueError("Failed to parse OpenAI web search response as JSON.")


class DeepResearchAgent:
    def __init__(
        self,
        model: str,
        report_model: str,
        max_depth: int,
        max_rounds: int,
        results_per_query: int,
        trace_file: str = "",
        state_file: str = "",
        resume_from: str = "",
        token_breakdown: bool = True,
        usage_file: str = "",
        pricing_file: str = "pricing.json",
        cost_estimate_enabled: bool = True,
        verbose: bool = True,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        run_id: str = "",
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> None:
        pricing_error = ""
        if cost_estimate_enabled:
            try:
                pricing_models, pricing_default, pricing_source = load_pricing_config(
                    pricing_file
                )
            except Exception as exc:
                pricing_models, pricing_default, pricing_source = (
                    {},
                    {"input_per_1m": 0.0, "output_per_1m": 0.0},
                    f"{pricing_file} (invalid)",
                )
                pricing_error = str(exc)
        else:
            pricing_models, pricing_default, pricing_source = (
                {},
                {"input_per_1m": 0.0, "output_per_1m": 0.0},
                f"{pricing_file} (skipped: cost estimation disabled)",
            )
        self.usage_tracker = UsageTracker(
            enabled=token_breakdown,
            cost_enabled=cost_estimate_enabled,
            pricing_source=pricing_source,
            pricing_models=pricing_models,
            pricing_default=pricing_default,
        )
        self.llm = LLM(model=model, usage_tracker=self.usage_tracker)
        self.decompose_llm = LLM(model=report_model, usage_tracker=self.usage_tracker)
        self.report_llm = LLM(model=report_model, usage_tracker=self.usage_tracker)
        self.search = WebSearch(model=model, usage_tracker=self.usage_tracker)
        self.max_depth = max(1, int(max_depth))
        self.max_rounds = max_rounds
        self.results_per_query = results_per_query
        self.trace_file = trace_file
        self.state_file = state_file
        self.resume_from = resume_from
        self.token_breakdown = token_breakdown
        self.usage_file = usage_file
        self.pricing_file = pricing_file
        self.cost_estimate_enabled = cost_estimate_enabled
        self.pricing_error = pricing_error
        self.verbose = verbose
        self.event_callback = event_callback
        self.run_id = run_id
        self.should_abort = should_abort
        self.source_policy = load_source_policy()
        self.search_policy = load_search_policy()
        self.query_memory: Dict[str, Dict[str, Any]] = {}

    def run(self, task: str) -> str:
        self._log("Starting deep research run.")
        self._abort_if_requested("run_start")
        self._emit("run_started", {"task": task})
        if self.pricing_error:
            self._log(
                "Pricing config invalid; cost estimation falling back to zeros. "
                f"error={self.pricing_error}"
            )

        started_at = self._now_iso()
        state_path = self._resolve_state_path(task, started_at)
        query_history: List[Dict[str, Any]] = []
        completed_query_steps: set[str] = set()
        recheck_queries: List[str] = []
        trace: Dict[str, Any] = {}
        findings: Dict[str, SubQuestionFinding] = {}
        sub_questions: List[str] = []
        success_criteria: List[str] = []
        extra_questions: List[str] = []
        extra_queries: List[str] = []
        unresolved_questions: List[str] = []
        question_depths: Dict[str, int] = {}
        question_parents: Dict[str, str] = {}
        question_node_ids: Dict[str, str] = {}
        decomposed_questions: set[str] = set()
        resolved_questions: set[str] = set()
        final_suff: Dict[str, Any] = {}
        total_search_calls = 0
        failed_search_calls = 0
        queries_with_evidence = 0
        total_selected_results = 0

        if self.resume_from:
            resume_state = self._load_state(Path(self.resume_from))
            restored_task = str(resume_state.get("task", task))
            if restored_task != task:
                self._log(
                    "Provided task differs from resume checkpoint; continuing with checkpoint task."
                )
            task = restored_task
            started_at = str(resume_state.get("started_at", started_at))
            state_path = Path(self.resume_from)
            trace = resume_state.get("trace", {})
            if not isinstance(trace, dict):
                trace = {}
            sub_questions = self._normalize_llm_list(
                resume_state.get("sub_questions", []), "sub_questions"
            )
            success_criteria = self._normalize_llm_list(
                resume_state.get("success_criteria", []), "success_criteria"
            )
            findings = self._deserialize_findings(resume_state.get("findings", {}))
            for sq in sub_questions:
                findings.setdefault(sq, SubQuestionFinding(sub_question=sq))
            extra_questions = self._normalize_llm_list(
                resume_state.get("extra_questions", []), "extra_questions"
            )
            extra_queries = self._normalize_llm_list(
                resume_state.get("extra_queries", []), "extra_queries"
            )
            recheck_queries = self._normalize_llm_list(
                resume_state.get("recheck_queries", []), "recheck_queries"
            )
            unresolved_questions = self._normalize_llm_list(
                resume_state.get("unresolved_questions", []), "unresolved_questions"
            )
            if not unresolved_questions:
                unresolved_questions = self._dedupe_preserve_order(
                    sub_questions + extra_questions
                )
            raw_depths = resume_state.get("question_depths", {})
            if isinstance(raw_depths, dict):
                for q, depth in raw_depths.items():
                    if isinstance(q, str) and isinstance(depth, int):
                        question_depths[q] = max(1, depth)
            for q in self._dedupe_preserve_order(
                sub_questions + unresolved_questions + extra_questions
            ):
                question_depths.setdefault(q, 1)
            raw_parents = resume_state.get("question_parents", {})
            if isinstance(raw_parents, dict):
                for q, parent in raw_parents.items():
                    if isinstance(q, str) and isinstance(parent, str):
                        question_parents[q] = parent
            raw_node_ids = resume_state.get("question_node_ids", {})
            if isinstance(raw_node_ids, dict):
                for q, nid in raw_node_ids.items():
                    if isinstance(q, str) and isinstance(nid, str):
                        question_node_ids[q] = nid
            decomposed_questions = set(
                self._normalize_llm_list(
                    resume_state.get("decomposed_questions", []),
                    "decomposed_questions",
                )
            )
            resolved_questions = set(
                self._normalize_llm_list(
                    resume_state.get("resolved_questions", []), "resolved_questions"
                )
            )
            final_suff = (
                resume_state.get("final_sufficiency", {})
                if isinstance(resume_state.get("final_sufficiency", {}), dict)
                else {}
            )
            stats = resume_state.get("search_stats", {})
            total_search_calls = int(stats.get("total_calls", 0))
            failed_search_calls = int(stats.get("failed_calls", 0))
            queries_with_evidence = int(stats.get("queries_with_evidence", 0))
            total_selected_results = int(stats.get("total_selected_results", 0))
            query_history = (
                resume_state.get("query_history", [])
                if isinstance(resume_state.get("query_history", []), list)
                else []
            )
            completed_query_steps = set(
                self._normalize_llm_list(
                    resume_state.get("completed_query_steps", []),
                    "completed_query_steps",
                )
            )
            self.usage_tracker.load_from_dict(resume_state.get("token_usage", {}))
            qm = resume_state.get("query_memory", {})
            self.query_memory = qm if isinstance(qm, dict) else {}
            start_round = max(1, int(resume_state.get("next_round", 1)))
            self._log(f"Resuming from {state_path} at round {start_round}.")
        else:
            trace = {
                "task": task,
                "model": self.llm.model,
                "decompose_model": self.decompose_llm.model,
                "report_model": self.report_llm.model,
                "token_breakdown_enabled": self.token_breakdown,
                "cost_estimate_enabled": self.cost_estimate_enabled,
                "pricing_source": self.usage_tracker.pricing_source,
                "started_at": started_at,
                "max_rounds": self.max_rounds,
                "max_depth": self.max_depth,
                "execution_mode": "lazy_first_dfs",
                "results_per_query": self.results_per_query,
                "rounds": [],
                "source_policy_file": str(SOURCE_POLICY_PATH),
                "search_policy_file": str(SEARCH_POLICY_PATH),
            }
            self._log("Initializing root task node (no upfront decomposition).")
            root_question = task.strip() or "Research task"
            sub_questions = [root_question]
            success_criteria = []
            self._emit(
                "plan_created",
                {
                    "sub_questions": sub_questions,
                    "success_criteria": success_criteria,
                },
            )
            trace["plan"] = {
                "sub_questions": sub_questions,
                "success_criteria": success_criteria,
                "planning_mode": "root_only",
            }
            findings = {sq: SubQuestionFinding(sub_question=sq) for sq in sub_questions}
            unresolved_questions = list(sub_questions)
            for sq in sub_questions:
                question_depths[sq] = 1
                question_parents[sq] = ""
            start_round = 1
            self.query_memory = {}

        parent_child_order: Dict[str, Dict[str, int]] = {}
        root_order: Dict[str, int] = {}

        for q, nid in list(question_node_ids.items()):
            if not isinstance(nid, str):
                continue
            parts = nid.split(".")
            if not parts or not all(p.isdigit() for p in parts):
                continue
            parent = question_parents.get(q, "")
            if parent:
                parent_child_order.setdefault(parent, {})[q] = int(parts[-1])
            else:
                root_order[q] = int(parts[0])

        def get_or_assign_node_id(question: str, parent: str) -> str:
            q = question.strip()
            p = parent.strip()
            if not q:
                return ""
            existing = question_node_ids.get(q, "")
            if isinstance(existing, str) and existing.strip():
                return existing

            if p:
                parent_id = question_node_ids.get(p, "")
                if not parent_id:
                    parent_id = get_or_assign_node_id(p, question_parents.get(p, ""))
                order_map = parent_child_order.setdefault(p, {})
                child_idx = order_map.get(q)
                if child_idx is None:
                    child_idx = len(order_map) + 1
                    order_map[q] = child_idx
                node_id = f"{parent_id}.{child_idx}" if parent_id else str(child_idx)
            else:
                root_idx = root_order.get(q)
                if root_idx is None:
                    root_idx = len(root_order) + 1
                    root_order[q] = root_idx
                node_id = str(root_idx)

            question_node_ids[q] = node_id
            return node_id

        unresolved_questions = self._dedupe_preserve_order(
            [q for q in unresolved_questions if q not in resolved_questions]
        )
        for q in self._dedupe_preserve_order(
            sub_questions + extra_questions + unresolved_questions
        ):
            get_or_assign_node_id(q, question_parents.get(q, ""))

        def save_checkpoint(
            status: str,
            next_round: int,
            error: str = "",
        ) -> None:
            self._save_state(
                state_path=state_path,
                status=status,
                task=task,
                started_at=started_at,
                next_round=next_round,
                sub_questions=sub_questions,
                success_criteria=success_criteria,
                findings=findings,
                extra_questions=extra_questions,
                extra_queries=extra_queries,
                recheck_queries=recheck_queries,
                unresolved_questions=unresolved_questions,
                question_depths=question_depths,
                question_parents=question_parents,
                question_node_ids=question_node_ids,
                decomposed_questions=decomposed_questions,
                resolved_questions=resolved_questions,
                final_suff=final_suff,
                total_search_calls=total_search_calls,
                failed_search_calls=failed_search_calls,
                queries_with_evidence=queries_with_evidence,
                total_selected_results=total_selected_results,
                query_history=query_history,
                completed_query_steps=completed_query_steps,
                trace=trace,
                error=error,
            )

        save_checkpoint(status="running", next_round=start_round)
        current_round = start_round

        try:
            for round_i in range(start_round, self.max_rounds + 1):
                self._abort_if_requested("round_start")
                current_round = round_i
                frontier_questions = self._dedupe_preserve_order(
                    unresolved_questions + extra_questions
                )
                if not frontier_questions:
                    self._log("No unresolved questions in frontier; stopping early.")
                    break

                self._log(
                    f"Round {round_i}/{self.max_rounds}: processing {len(frontier_questions)} frontier questions."
                )
                self._emit(
                    "round_started",
                    {
                        "round": round_i,
                        "max_rounds": self.max_rounds,
                        "frontier_count": len(frontier_questions),
                    },
                )
                round_trace: Dict[str, Any] = {
                    "round": round_i,
                    "frontier_questions": frontier_questions,
                    "questions": [],
                }
                trace["rounds"].append(round_trace)
                round_new_fact_gain = 0
                round_extra_queries_by_question = self._assign_follow_up_queries(
                    frontier_questions, self._dedupe_preserve_order(extra_queries)
                )
                round_recheck_intents = {
                    self._normalize_query_intent(q)
                    for q in self._normalize_llm_list(
                        recheck_queries, "recheck_queries_for_recheck"
                    )
                }
                next_unresolved: List[str] = []
                processed_questions_in_pass: set[str] = set()
                active_questions_in_branch: set[str] = set()

                def process_question_dfs(
                    sq: str, depth: int, parent: str, ancestry: List[str]
                ) -> bool:
                    nonlocal round_new_fact_gain, total_search_calls, failed_search_calls
                    nonlocal queries_with_evidence, total_selected_results
                    self._abort_if_requested("sub_question_start")
                    if sq in processed_questions_in_pass:
                        return sq in resolved_questions
                    if sq in resolved_questions:
                        processed_questions_in_pass.add(sq)
                        return True
                    if sq in active_questions_in_branch:
                        self._emit(
                            "node_unresolved",
                            {
                                "round": round_i,
                                "sub_question": sq,
                                "depth": depth,
                                "reason": "active_branch_cycle",
                            },
                        )
                        next_unresolved.append(sq)
                        processed_questions_in_pass.add(sq)
                        return False
                    if sq in ancestry:
                        self._emit(
                            "node_unresolved",
                            {
                                "round": round_i,
                                "sub_question": sq,
                                "depth": depth,
                                "reason": "cycle_detected",
                            },
                        )
                        next_unresolved.append(sq)
                        processed_questions_in_pass.add(sq)
                        return False

                    active_questions_in_branch.add(sq)
                    question_depths[sq] = max(1, depth)
                    if sq not in question_parents or not question_parents.get(sq, ""):
                        question_parents[sq] = parent
                    node_id = get_or_assign_node_id(sq, question_parents.get(sq, ""))
                    findings.setdefault(sq, SubQuestionFinding(sub_question=sq))
                    self._log(f"Question [{node_id}] (depth {depth}): {sq}")
                    self._emit(
                        "sub_question_started",
                        {
                            "round": round_i,
                            "sub_question": sq,
                            "depth": depth,
                            "parent": parent,
                            "node_id": node_id,
                        },
                    )

                    question_trace: Dict[str, Any] = {
                        "node_id": node_id,
                        "sub_question": sq,
                        "depth": depth,
                        "parent": parent,
                        "status": "running",
                        "query_steps": [],
                    }
                    round_trace["questions"].append(question_trace)
                    query_prompt = f"""Original task:
{task}

Sub-question:
{sq}

Question depth:
{depth}

Known success criteria:
{json.dumps(success_criteria, ensure_ascii=False)}
"""
                    qobj = self.llm.json(
                        SYSTEM_QUERY_GEN,
                        query_prompt,
                        stage="query_gen",
                        metadata={
                            "round": round_i,
                            "depth": depth,
                            "sub_question": self._trim_text(sq, 120),
                        },
                    )
                    generated_queries = self._normalize_llm_list(
                        qobj.get("queries"), "queries"
                    )
                    queries = self._dedupe_preserve_order(
                        generated_queries + round_extra_queries_by_question.get(sq, [])
                    )
                    self._log(f"Generated {len(queries)} search queries.")
                    self._emit(
                        "queries_generated",
                        {
                            "round": round_i,
                            "sub_question": sq,
                            "depth": depth,
                            "queries": queries,
                            "count": len(queries),
                        },
                    )

                    question_has_evidence = False
                    for query in queries:
                        self._abort_if_requested("query_loop")
                        raw_intent_key = self._normalize_query_intent(query)
                        intent_key, collapse_score = self._resolve_execution_intent(
                            raw_intent_key
                        )
                        if intent_key != raw_intent_key:
                            self._log(
                                "Query intent mapped to existing cache key. "
                                f"raw='{self._trim_text(raw_intent_key, 120)}' "
                                f"mapped='{self._trim_text(intent_key, 120)}' "
                                f"similarity={collapse_score:.2f}"
                            )
                        step_key = self._query_step_key(round_i, sq, query)
                        if step_key in completed_query_steps:
                            self._log(f"Skipping already completed query: {query}")
                            continue

                        explicit_recheck = (
                            raw_intent_key in round_recheck_intents
                            or intent_key in round_recheck_intents
                        )
                        should_run, decision = self._decide_query_execution(
                            intent_key=intent_key,
                            explicit_recheck=explicit_recheck,
                        )
                        planned_k = self.results_per_query
                        if should_run:
                            planned_k, _ = self._effective_results_per_query(intent_key)
                        else:
                            cached = self.query_memory.get(intent_key, {})
                            planned_k = int(
                                cached.get("effective_k", self.results_per_query)
                            )
                        diagnostics = self._log_query_diagnostics(
                            query=query,
                            intent_key=raw_intent_key,
                            decision=decision,
                            effective_k=planned_k,
                            explicit_recheck=explicit_recheck,
                        )
                        self._emit(
                            "query_diagnostic",
                            {
                                "round": round_i,
                                "sub_question": sq,
                                "depth": depth,
                                "query": query,
                                "raw_intent_key": raw_intent_key,
                                "execution_intent_key": intent_key,
                                "intent_mapped": bool(intent_key != raw_intent_key),
                                "intent_map_similarity": float(collapse_score),
                                **diagnostics,
                            },
                        )
                        if not should_run:
                            cache_entry = self.query_memory.get(intent_key, {})
                            step_data = {
                                "node_id": node_id,
                                "round": round_i,
                                "sub_question": sq,
                                "depth": depth,
                                "query": query,
                                "status": "cached" if decision == "cache_hit" else "skipped",
                                "cache_hit": decision == "cache_hit",
                                "rerun_reason": decision,
                                "broadening_step": int(cache_entry.get("attempts", 0)),
                                "effective_results_per_query": int(
                                    cache_entry.get("effective_k", self.results_per_query)
                                ),
                                "search_error": cache_entry.get("search_error"),
                                "selected_results_count": int(
                                    cache_entry.get("selected_results_count", 0)
                                ),
                                "primary_count": int(cache_entry.get("primary_count", 0)),
                                "blocked_by_diminishing_returns": decision
                                == "blocked_diminishing_returns",
                            }
                            question_trace["query_steps"].append(step_data)
                            query_history.append(step_data)
                            completed_query_steps.add(step_key)
                            if decision == "cache_hit":
                                self._emit(
                                    "query_skipped_cached",
                                    {
                                        "round": round_i,
                                        "sub_question": sq,
                                        "depth": depth,
                                        "query": query,
                                        "intent_key": intent_key,
                                    },
                                )
                            else:
                                self._emit(
                                    "query_blocked_diminishing_returns",
                                    {
                                        "round": round_i,
                                        "sub_question": sq,
                                        "depth": depth,
                                        "query": query,
                                        "intent_key": intent_key,
                                    },
                                )
                            save_checkpoint(status="running", next_round=round_i)
                            continue

                        rerun_reason = None if decision == "new_query" else decision
                        if rerun_reason:
                            self._emit(
                                "query_rerun_allowed",
                                {
                                    "round": round_i,
                                    "sub_question": sq,
                                    "depth": depth,
                                    "query": query,
                                    "intent_key": intent_key,
                                    "reason": rerun_reason,
                                },
                            )

                        effective_k, broaden_step = self._effective_results_per_query(
                            intent_key
                        )
                        if effective_k > self.results_per_query:
                            self._emit(
                                "query_broadened",
                                {
                                    "round": round_i,
                                    "sub_question": sq,
                                    "depth": depth,
                                    "query": query,
                                    "intent_key": intent_key,
                                    "effective_results_per_query": effective_k,
                                    "broadening_step": broaden_step,
                                },
                            )

                        self._log(f"Searching: {query}")
                        self._emit(
                            "query_started",
                            {
                                "round": round_i,
                                "sub_question": sq,
                                "depth": depth,
                                "query": query,
                            },
                        )
                        total_search_calls += 1
                        raw_results = self.search.search(query, effective_k)
                        self._abort_if_requested("after_search")
                        search_error = self.search.last_error
                        if search_error:
                            failed_search_calls += 1
                            self._log(
                                f"Search failed for this query; continuing. error={search_error}"
                            )
                        selected_results = self._select_high_quality_results(
                            raw_results, max(effective_k, self.results_per_query)
                        )
                        primary_count = sum(
                            1
                            for r in selected_results
                            if r.quality_tier == "primary_or_official"
                        )
                        self._log(
                            "Selected "
                            f"{len(selected_results)} results "
                            f"({primary_count} primary/official)."
                        )
                        self._emit(
                            "search_completed",
                            {
                                "round": round_i,
                                "sub_question": sq,
                                "depth": depth,
                                "query": query,
                                "intent_key": intent_key,
                                "search_error": search_error,
                                "raw_results_count": len(raw_results),
                                "selected_results_count": len(selected_results),
                                "primary_count": primary_count,
                                "selected_sources": [
                                    {
                                        "title": r.title,
                                        "url": r.url,
                                    }
                                    for r in selected_results
                                ],
                                "effective_results_per_query": effective_k,
                                "broadening_step": broaden_step,
                            },
                        )
                        total_selected_results += len(selected_results)
                        if selected_results:
                            question_has_evidence = True
                            queries_with_evidence += 1

                        if not selected_results:
                            limitation = (
                                "No usable evidence retrieved for this query; "
                                "synthesis skipped due to insufficient grounding."
                            )
                            self._log(limitation)
                            findings[sq].uncertainties.append(limitation)
                            step_data = {
                                "node_id": node_id,
                                "round": round_i,
                                "sub_question": sq,
                                "depth": depth,
                                "query": query,
                                "search_error": search_error,
                                "raw_results": [r.__dict__ for r in raw_results],
                                "selected_results": [],
                                "synthesis_skipped": True,
                                "limitation": limitation,
                            }
                            question_trace["query_steps"].append(step_data)
                            query_history.append(step_data)
                            completed_query_steps.add(step_key)
                            self._update_query_memory(
                                intent_key=intent_key,
                                query=query,
                                round_i=round_i,
                                effective_k=effective_k,
                                selected_results=selected_results,
                                search_error=search_error,
                                rerun_reason=rerun_reason,
                                new_fact_gain=0,
                            )
                            save_checkpoint(status="running", next_round=round_i)
                            continue

                        synth_prompt = self._format_synthesis_prompt(
                            sq, query, selected_results
                        )
                        sobj = self.llm.json(
                            SYSTEM_SYNTHESIZE,
                            synth_prompt,
                            stage="synthesize",
                            metadata={
                                "round": round_i,
                                "depth": depth,
                                "sub_question": self._trim_text(sq, 120),
                                "query": self._trim_text(query, 140),
                            },
                        )
                        self._abort_if_requested("after_synthesis")
                        self._log("Synthesis completed for this query.")
                        self._emit(
                            "synthesis_completed",
                            {
                                "round": round_i,
                                "sub_question": sq,
                                "depth": depth,
                                "query": query,
                                "summary": self._trim_text(
                                    str(sobj.get("summary", "")), 300
                                ),
                            },
                        )
                        facts_before = len(findings[sq].facts)
                        findings[sq].summaries.append(str(sobj.get("summary", "")))
                        findings[sq].facts.extend(
                            self._normalize_synth_facts(sobj.get("facts", []))
                        )
                        findings[sq].uncertainties.extend(
                            self._normalize_llm_list(
                                sobj.get("uncertainties"), "synthesis_uncertainties"
                            )
                        )
                        new_fact_gain = max(0, len(findings[sq].facts) - facts_before)
                        round_new_fact_gain += new_fact_gain
                        step_data = {
                            "node_id": node_id,
                            "round": round_i,
                            "sub_question": sq,
                            "depth": depth,
                            "query": query,
                            "intent_key": intent_key,
                            "search_error": search_error,
                            "raw_results": [r.__dict__ for r in raw_results],
                            "selected_results": [
                                {
                                    "title": r.title,
                                    "snippet": r.snippet,
                                    "url": r.url,
                                    "quality_tier": r.quality_tier,
                                    "quality_score": r.quality_score,
                                }
                                for r in selected_results
                            ],
                            "synthesis": sobj,
                            "cache_hit": False,
                            "rerun_reason": rerun_reason,
                            "broadening_step": broaden_step,
                            "effective_results_per_query": effective_k,
                            "new_fact_gain": new_fact_gain,
                            "blocked_by_diminishing_returns": False,
                        }
                        question_trace["query_steps"].append(step_data)
                        query_history.append(step_data)
                        completed_query_steps.add(step_key)
                        self._update_query_memory(
                            intent_key=intent_key,
                            query=query,
                            round_i=round_i,
                            effective_k=effective_k,
                            selected_results=selected_results,
                            search_error=search_error,
                            rerun_reason=rerun_reason,
                            new_fact_gain=new_fact_gain,
                        )
                        save_checkpoint(status="running", next_round=round_i)

                    try:
                        node_suff: Dict[str, Any] = {
                            "is_sufficient": False,
                            "reasoning": "No node-level sufficiency run.",
                            "gaps": [],
                        }
                        if question_has_evidence:
                            self._emit(
                                "node_sufficiency_started",
                                {
                                    "round": round_i,
                                    "sub_question": sq,
                                    "depth": depth,
                                },
                            )
                            node_suff = self.llm.json(
                                SYSTEM_SUFFICIENCY,
                                self._format_node_sufficiency_prompt(
                                    task=task,
                                    sub_question=sq,
                                    success_criteria=success_criteria,
                                    finding=findings[sq],
                                    depth=depth,
                                ),
                                stage="node_sufficiency",
                                metadata={
                                    "round": round_i,
                                    "depth": depth,
                                    "sub_question": self._trim_text(sq, 120),
                                },
                            )
                        else:
                            findings[sq].uncertainties.append(
                                "Insufficient direct evidence for this question in this pass."
                            )
                        node_is_sufficient = bool(node_suff.get("is_sufficient", False))
                        node_reasoning = str(node_suff.get("reasoning", "")).strip()
                        node_gaps = self._normalize_llm_list(node_suff.get("gaps"), "node_gaps")
                        question_trace["node_sufficiency"] = {
                            "is_sufficient": node_is_sufficient,
                            "reasoning": node_reasoning,
                            "gaps": node_gaps,
                        }
                        self._emit(
                            "node_sufficiency_completed",
                            {
                                "round": round_i,
                                "sub_question": sq,
                                "depth": depth,
                                "is_sufficient": node_is_sufficient,
                                "reasoning": self._trim_text(node_reasoning, 300),
                                "gaps": node_gaps,
                            },
                        )

                        if node_is_sufficient:
                            resolved_questions.add(sq)
                            self._emit(
                                "node_completed",
                                {
                                    "round": round_i,
                                    "sub_question": sq,
                                    "depth": depth,
                                },
                            )
                            question_trace["status"] = "solved"
                            save_checkpoint(status="running", next_round=round_i)
                            return True

                        if depth < self.max_depth:
                            self._emit(
                                "node_decomposition_started",
                                {
                                    "round": round_i,
                                    "sub_question": sq,
                                    "depth": depth,
                                },
                            )
                            question_trace["status"] = "decomposing"
                            children = self._decompose_sub_question(
                                task=task,
                                sub_question=sq,
                                success_criteria=success_criteria,
                                node_gaps=node_gaps,
                            )
                            children = self._dedupe_preserve_order(children)
                            if children:
                                child_node_ids: List[str] = []
                                for child in children:
                                    question_depths[child] = min(self.max_depth, depth + 1)
                                    if child not in question_parents or not question_parents.get(child, ""):
                                        question_parents[child] = sq
                                    child_node_ids.append(get_or_assign_node_id(child, sq))
                                decomposed_questions.add(sq)
                                self._emit(
                                    "node_decomposed",
                                    {
                                        "round": round_i,
                                        "sub_question": sq,
                                        "depth": depth,
                                        "children": children,
                                        "child_node_ids": child_node_ids,
                                    },
                                )
                                question_trace["children"] = children
                                question_trace["child_node_ids"] = child_node_ids
                                child_all_solved = True
                                for child in children:
                                    findings.setdefault(
                                        child, SubQuestionFinding(sub_question=child)
                                    )
                                    child_solved = process_question_dfs(
                                        child, min(self.max_depth, depth + 1), sq, ancestry + [sq]
                                    )
                                    child_all_solved = child_all_solved and child_solved
                                if child_all_solved:
                                    self._emit(
                                        "node_completed",
                                        {
                                            "round": round_i,
                                            "sub_question": sq,
                                            "depth": depth,
                                            "derived_from_children": True,
                                        },
                                    )
                                    resolved_questions.add(sq)
                                    question_trace["status"] = "solved_via_children"
                                    save_checkpoint(status="running", next_round=round_i)
                                    return True
                                self._emit(
                                    "node_unresolved",
                                    {
                                        "round": round_i,
                                        "sub_question": sq,
                                        "depth": depth,
                                        "reason": "children_insufficient",
                                    },
                                )
                                question_trace["status"] = "unresolved"
                                question_trace["unresolved_reason"] = "children_insufficient"
                                next_unresolved.append(sq)
                                save_checkpoint(status="running", next_round=round_i)
                                return False

                        reason = "depth_limit_reached" if depth >= self.max_depth else "decompose_failed"
                        self._emit(
                            "node_unresolved",
                            {
                                "round": round_i,
                                "sub_question": sq,
                                "depth": depth,
                                "reason": reason,
                            },
                        )
                        question_trace["status"] = "unresolved"
                        question_trace["unresolved_reason"] = reason
                        next_unresolved.append(sq)
                        save_checkpoint(status="running", next_round=round_i)
                        return False
                    finally:
                        active_questions_in_branch.discard(sq)
                        processed_questions_in_pass.add(sq)

                for sq in frontier_questions:
                    if sq in resolved_questions:
                        continue
                    root_depth = max(1, int(question_depths.get(sq, 1)))
                    root_parent = question_parents.get(sq, "")
                    process_question_dfs(sq, root_depth, root_parent, [])

                unresolved_questions = self._dedupe_preserve_order(
                    [q for q in next_unresolved if q not in resolved_questions]
                )

                root_resolved = bool(
                    sub_questions and sub_questions[0] in resolved_questions
                )
                if not unresolved_questions or root_resolved:
                    suff = {
                        "is_sufficient": True,
                        "reasoning": "All unresolved questions are closed or root question is resolved.",
                        "gaps": [],
                        "follow_up_questions": [],
                        "follow_up_queries": [],
                        "recheck_queries": [],
                    }
                else:
                    self._emit(
                        "sufficiency_started",
                        {
                            "round": round_i,
                        },
                    )
                    suff = self.llm.json(
                        SYSTEM_SUFFICIENCY,
                        self._format_sufficiency_prompt(
                            task=task,
                            success_criteria=success_criteria,
                            findings=findings,
                            round_i=round_i,
                        ),
                        stage="sufficiency",
                        metadata={"round": round_i},
                    )
                    self._abort_if_requested("after_sufficiency")
                final_suff = suff
                round_trace["sufficiency"] = suff
                round_trace["round_new_fact_gain"] = round_new_fact_gain
                round_trace["frontier_remaining"] = unresolved_questions
                self._emit(
                    "sufficiency_completed",
                    {
                        "round": round_i,
                        "is_sufficient": bool(suff.get("is_sufficient", False)),
                        "reasoning": self._trim_text(str(suff.get("reasoning", "")), 300),
                        "gaps": self._normalize_llm_list(suff.get("gaps"), "gaps"),
                    },
                )
                if suff.get("is_sufficient", False):
                    self._log("Sufficiency check passed. Stopping iterative search.")
                    save_checkpoint(status="running", next_round=round_i)
                    break

                self._log("Sufficiency check failed. Preparing focused follow-up pass.")
                extra_questions = self._normalize_llm_list(
                    suff.get("follow_up_questions"), "follow_up_questions"
                )
                extra_queries = self._dedupe_preserve_order(
                    self._normalize_llm_list(
                        suff.get("follow_up_queries"), "follow_up_queries"
                    )
                )
                recheck_queries = self._dedupe_preserve_order(
                    self._normalize_llm_list(
                        suff.get("recheck_queries"), "recheck_queries"
                    )
                )
                for q in extra_questions:
                    question_depths.setdefault(q, 1)
                    question_parents.setdefault(q, "")
                    findings.setdefault(q, SubQuestionFinding(sub_question=q))
                unresolved_questions = self._dedupe_preserve_order(
                    unresolved_questions + extra_questions
                )
                if (
                    round_new_fact_gain == 0
                    and not extra_questions
                    and not extra_queries
                    and not recheck_queries
                ):
                    self._log(
                        "Stopping early due to no evidence gain and no targeted follow-up actions."
                    )
                    break
                save_checkpoint(status="running", next_round=round_i + 1)

            trace["search_stats"] = {
                "total_calls": total_search_calls,
                "failed_calls": failed_search_calls,
                "queries_with_evidence": queries_with_evidence,
                "total_selected_results": total_selected_results,
            }
            trace["query_memory"] = self._query_memory_summary()
            trace["question_tree"] = {
                "depths": question_depths,
                "parents": question_parents,
                "node_ids": question_node_ids,
                "resolved_questions": sorted(resolved_questions),
                "decomposed_questions": sorted(decomposed_questions),
                "unresolved_questions": unresolved_questions,
            }
            evidence_status, evidence_note = self._assess_evidence_strength(
                total_search_calls=total_search_calls,
                failed_search_calls=failed_search_calls,
                queries_with_evidence=queries_with_evidence,
                total_selected_results=total_selected_results,
            )
            trace["evidence_assessment"] = {
                "status": evidence_status,
                "note": evidence_note,
            }
            if evidence_status != "adequate":
                self._log(f"Evidence limitation: {evidence_note}")

            self._log("Generating final Pyramid Principle report.")
            self._emit(
                "report_generation_started",
                {
                    "mode": "final",
                },
            )
            report_prompt = self._format_report_prompt(
                task=task,
                success_criteria=success_criteria,
                findings=findings,
                evidence_status=evidence_status,
                evidence_note=evidence_note,
                search_stats=trace["search_stats"],
            )
            report = self.report_llm.text(
                SYSTEM_REPORT,
                report_prompt,
                stage="report",
                metadata={"task": self._trim_text(task, 120)},
            )
            self._abort_if_requested("after_report")
            trace["finished_at"] = self._now_iso()
            trace["final_sufficiency"] = final_suff
            trace["report"] = report
            trace["token_usage"] = self.usage_tracker.to_dict()
            self._emit(
                "run_completed",
                {
                    "report_chars": len(report),
                    "evidence_status": evidence_status,
                    "token_usage": self.usage_tracker.to_dict().get("total", {}),
                },
            )
            trace_path = self._write_trace(trace)
            print(f"[trace] saved: {trace_path}")
            self._print_token_summary(self.usage_tracker.to_dict())
            if self.usage_file:
                usage_path = self._write_usage_report(self.usage_tracker.to_dict())
                print(f"[usage] saved: {usage_path}")
            save_checkpoint(status="completed", next_round=self.max_rounds + 1)
            self._log("Run completed.")
            return report
        except Exception as exc:
            if str(exc) == "Run aborted by user":
                self._emit(
                    "run_aborted",
                    {"round": current_round, "error": str(exc)},
                )
            else:
                self._emit(
                    "run_failed",
                    {"round": current_round, "error": str(exc)},
                )
            trace["query_memory"] = self._query_memory_summary()
            trace["token_usage"] = self.usage_tracker.to_dict()
            save_checkpoint(status="failed", next_round=current_round, error=str(exc))
            raise

    def _format_node_sufficiency_prompt(
        self,
        task: str,
        sub_question: str,
        success_criteria: List[str],
        finding: SubQuestionFinding,
        depth: int,
    ) -> str:
        compact = self._compact_findings(
            findings={sub_question: finding},
            summaries_limit=3,
            facts_limit=10,
            uncertainties_limit=6,
            char_budget=9000,
        )
        return textwrap.dedent(
            f"""
            Original task:
            {task}

            Focused sub-question:
            {sub_question}

            Current depth:
            {depth}

            Success criteria:
            {json.dumps(success_criteria, ensure_ascii=False)}

            Evidence for this sub-question:
            {json.dumps(compact, ensure_ascii=False)}

            Evaluate only whether this specific sub-question is sufficiently answered.
            """
        ).strip()

    def _decompose_sub_question(
        self,
        task: str,
        sub_question: str,
        success_criteria: List[str],
        node_gaps: List[str],
    ) -> List[str]:
        prompt = textwrap.dedent(
            f"""
            Original task:
            {task}

            Parent question to decompose:
            {sub_question}

            Known unresolved gaps:
            {json.dumps(node_gaps, ensure_ascii=False)}

            Success criteria:
            {json.dumps(success_criteria, ensure_ascii=False)}

            Return only 2 to 4 child sub-questions focused on unresolved evidence gaps.
            """
        ).strip()
        try:
            plan = self.decompose_llm.json(
                SYSTEM_DECOMPOSE,
                prompt,
                stage="decompose_child",
                metadata={"sub_question": self._trim_text(sub_question, 120)},
            )
        except Exception as exc:
            self._log(f"Child decomposition failed: {exc}")
            return []
        children = self._normalize_llm_list(
            plan.get("sub_questions"), "child_sub_questions"
        )
        filtered = [q for q in children if q.strip() and q.strip() != sub_question.strip()]
        return self._enforce_mece_children(
            task=task,
            parent_question=sub_question,
            success_criteria=success_criteria,
            node_gaps=node_gaps,
            candidates=filtered,
        )

    def _enforce_mece_children(
        self,
        task: str,
        parent_question: str,
        success_criteria: List[str],
        node_gaps: List[str],
        candidates: List[str],
    ) -> List[str]:
        cleaned = self._dedupe_preserve_order(
            [q.strip() for q in candidates if q.strip() and q.strip() != parent_question.strip()]
        )
        if len(cleaned) <= 1:
            return cleaned
        # If overlap is high, ask once for a stricter MECE rewrite.
        overlap_pairs = self._find_overlapping_question_pairs(cleaned, threshold=0.55)
        if overlap_pairs:
            self._log(
                "Child decomposition overlap detected; running MECE rewrite. "
                f"parent='{self._trim_text(parent_question, 120)}' "
                f"overlap_pairs={len(overlap_pairs)}"
            )
            rewrite_prompt = textwrap.dedent(
                f"""
                Original task:
                {task}

                Parent question:
                {parent_question}

                Unresolved gaps:
                {json.dumps(node_gaps, ensure_ascii=False)}

                Success criteria:
                {json.dumps(success_criteria, ensure_ascii=False)}

                Candidate child questions (overlapping):
                {json.dumps(cleaned, ensure_ascii=False)}

                Rewrite into 2 to 4 child questions that are MECE:
                - Mutually Exclusive: no overlap/paraphrase.
                - Collectively Exhaustive: covers unresolved scope.
                - Specific and researchable.
                Return strict JSON only.
                """
            ).strip()
            try:
                rewritten = self.decompose_llm.json(
                    SYSTEM_DECOMPOSE,
                    rewrite_prompt,
                    stage="decompose_child_mece_rewrite",
                    metadata={"sub_question": self._trim_text(parent_question, 120)},
                )
                cleaned = self._normalize_llm_list(
                    rewritten.get("sub_questions"), "child_sub_questions_mece_rewrite"
                )
            except Exception as exc:
                self._log(f"MECE rewrite failed; keeping original children. error={exc}")
        # Final overlap cleanup to avoid near-duplicate siblings.
        return self._drop_near_duplicate_questions(cleaned, threshold=0.72)

    def _find_overlapping_question_pairs(
        self, questions: List[str], threshold: float
    ) -> List[tuple[str, str, float]]:
        out: List[tuple[str, str, float]] = []
        for i in range(len(questions)):
            qi = questions[i]
            ti = self._tokenize(qi)
            if not ti:
                continue
            for j in range(i + 1, len(questions)):
                qj = questions[j]
                tj = self._tokenize(qj)
                if not tj:
                    continue
                score = len(ti & tj) / float(len(ti | tj))
                if score >= threshold:
                    out.append((qi, qj, score))
        return out

    def _drop_near_duplicate_questions(
        self, questions: List[str], threshold: float
    ) -> List[str]:
        kept: List[str] = []
        for q in questions:
            qn = self._normalize_query_intent(q)
            is_dup = False
            for k in kept:
                kn = self._normalize_query_intent(k)
                if self._query_similarity(qn, kn) >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(q)
        return kept

    def _format_synthesis_prompt(
        self, sub_question: str, query: str, results: List[EvaluatedResult]
    ) -> str:
        lines = [f"Sub-question: {sub_question}", f"Search query: {query}", "Results:"]
        for i, r in enumerate(results, start=1):
            lines.append(
                f"{i}. title={r.title}\n   url={r.url}\n   quality={r.quality_tier}\n   snippet={r.snippet[:700]}"
            )
        return "\n".join(lines)

    def _format_sufficiency_prompt(
        self,
        task: str,
        success_criteria: List[str],
        findings: Dict[str, SubQuestionFinding],
        round_i: int,
    ) -> str:
        compact = self._compact_findings(
            findings=findings,
            summaries_limit=2,
            facts_limit=8,
            uncertainties_limit=5,
            char_budget=12000,
        )
        return textwrap.dedent(
            f"""
            Original task:
            {task}

            Current round:
            {round_i}

            Success criteria:
            {json.dumps(success_criteria, ensure_ascii=False)}

            Current findings:
            {json.dumps(compact, ensure_ascii=False)}
            """
        ).strip()

    def _format_report_prompt(
        self,
        task: str,
        success_criteria: List[str],
        findings: Dict[str, SubQuestionFinding],
        evidence_status: str,
        evidence_note: str,
        search_stats: Dict[str, int],
    ) -> str:
        compact = self._compact_findings(
            findings=findings,
            summaries_limit=3,
            facts_limit=12,
            uncertainties_limit=6,
            char_budget=18000,
        )
        return textwrap.dedent(
            f"""
            Create the final report for this research task.

            Task:
            {task}

            Success criteria:
            {json.dumps(success_criteria, ensure_ascii=False)}

            Evidence quality:
            status={evidence_status}
            note={evidence_note}
            stats={json.dumps(search_stats, ensure_ascii=False)}

            Reporting rule:
            If evidence status is not "adequate", explicitly state this limitation in the
            Governing Thought and Risks/Unknowns sections, and avoid definitive claims.

            Evidence base:
            {json.dumps(compact, ensure_ascii=False)}
            """
        ).strip()

    def _compact_findings(
        self,
        findings: Dict[str, SubQuestionFinding],
        summaries_limit: int,
        facts_limit: int,
        uncertainties_limit: int,
        char_budget: int,
    ) -> Dict[str, Dict[str, Any]]:
        compact: Dict[str, Dict[str, Any]] = {}
        for k, v in findings.items():
            compact[k] = {
                "summaries": [self._trim_text(s, 220) for s in v.summaries[-summaries_limit:]],
                "facts": self._compact_facts(v.facts[-facts_limit:]),
                "uncertainties": [
                    self._trim_text(u, 200) for u in v.uncertainties[-uncertainties_limit:]
                ],
            }

        payload = json.dumps(compact, ensure_ascii=False)
        while len(payload) > char_budget and compact:
            changed = False
            for key in list(compact.keys()):
                entry = compact[key]
                if len(entry["facts"]) > 1:
                    entry["facts"] = entry["facts"][:-1]
                    changed = True
                    break
                if len(entry["summaries"]) > 1:
                    entry["summaries"] = entry["summaries"][:-1]
                    changed = True
                    break
                if len(entry["uncertainties"]) > 1:
                    entry["uncertainties"] = entry["uncertainties"][:-1]
                    changed = True
                    break
            if not changed:
                largest_key = max(
                    compact.keys(),
                    key=lambda kk: len(json.dumps(compact[kk], ensure_ascii=False)),
                )
                compact.pop(largest_key, None)
            payload = json.dumps(compact, ensure_ascii=False)
        return compact

    def _compact_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            claim = self._trim_text(str(fact.get("claim", "")), 240)
            srcs = fact.get("sources", [])
            if not isinstance(srcs, list):
                srcs = []
            clean_sources = [
                s.strip()
                for s in srcs
                if isinstance(s, str) and is_valid_absolute_http_url(s)
            ][:3]
            out.append({"claim": claim, "sources": clean_sources})
        return out

    def _trim_text(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _normalize_synth_facts(self, value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            self._log("LLM field 'synthesis_facts' is non-list; coercing to empty list.")
            return []
        out: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            srcs_raw = item.get("sources", [])
            if not isinstance(srcs_raw, list):
                srcs_raw = []
            sources = [
                s.strip()
                for s in srcs_raw
                if isinstance(s, str) and is_valid_absolute_http_url(s)
            ]
            out.append({"claim": claim, "sources": sources})
        if len(out) != len(value):
            self._log("LLM field 'synthesis_facts' contained invalid entries; cleaned.")
        return out

    def _assess_evidence_strength(
        self,
        total_search_calls: int,
        failed_search_calls: int,
        queries_with_evidence: int,
        total_selected_results: int,
    ) -> tuple[str, str]:
        if total_search_calls == 0:
            return "none", "No search queries were executed."
        if queries_with_evidence == 0 or total_selected_results == 0:
            return (
                "none",
                "No usable evidence was retrieved. Findings are highly limited and may be unreliable.",
            )
        if failed_search_calls == total_search_calls:
            return (
                "none",
                "All search calls failed technically. Findings are highly limited and may be unreliable.",
            )
        if queries_with_evidence < max(2, total_search_calls // 3):
            return (
                "meager",
                "Only a small fraction of queries returned usable evidence; confidence is limited.",
            )
        if total_selected_results < max(5, total_search_calls):
            return (
                "meager",
                "Evidence volume is low relative to search attempts; conclusions should be cautious.",
            )
        return "adequate", "Evidence coverage is sufficient for qualified conclusions."

    def _dedupe_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            key = item.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _is_cjk_char(self, ch: str) -> bool:
        if not isinstance(ch, str) or len(ch) != 1:
            return False
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # CJK Extension A
            or 0x3040 <= code <= 0x309F  # Hiragana
            or 0x30A0 <= code <= 0x30FF  # Katakana
            or 0xAC00 <= code <= 0xD7AF  # Hangul
        )

    def _contains_cjk(self, text: str) -> bool:
        return any(self._is_cjk_char(ch) for ch in str(text or ""))

    def _normalize_query_intent(self, query: str) -> str:
        tokens: List[str] = []
        buf: List[str] = []
        mode = ""  # "latin", "cjk", ""

        def flush() -> None:
            nonlocal buf, mode
            if buf:
                tok = "".join(buf).strip()
                if tok:
                    tokens.append(tok)
            buf = []
            mode = ""

        for ch in str(query or ""):
            if self._is_cjk_char(ch):
                if mode != "cjk":
                    flush()
                    mode = "cjk"
                buf.append(ch)
            elif ch.isalnum():
                lower = ch.lower()
                if mode != "latin":
                    flush()
                    mode = "latin"
                buf.append(lower)
            else:
                flush()
        flush()
        return " ".join(tokens)

    def _query_similarity(self, intent_a: str, intent_b: str) -> float:
        ta = set(intent_a.split())
        tb = set(intent_b.split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / float(len(ta | tb))

    def _nearest_prior_intent(self, intent_key: str) -> tuple[str, Dict[str, Any], float]:
        best_key = ""
        best_entry: Dict[str, Any] = {}
        best_score = 0.0
        for k, entry in self.query_memory.items():
            score = self._query_similarity(intent_key, str(k))
            if score > best_score:
                best_key = str(k)
                best_entry = entry if isinstance(entry, dict) else {}
                best_score = score
        return best_key, best_entry, best_score

    def _resolve_execution_intent(self, raw_intent_key: str) -> tuple[str, float]:
        if raw_intent_key in self.query_memory:
            return raw_intent_key, 1.0
        nearest_key, _, score = self._nearest_prior_intent(raw_intent_key)
        collapse_threshold = float(
            self.search_policy.get("intent_collapse_similarity", 0.55)
        )
        if nearest_key and score >= collapse_threshold:
            return nearest_key, score
        return raw_intent_key, 0.0

    def _log_query_diagnostics(
        self,
        query: str,
        intent_key: str,
        decision: str,
        effective_k: int,
        explicit_recheck: bool,
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "classification": "first_intent",
            "prior_query": "",
            "similarity": 0.0,
            "new_tokens": [],
            "dropped_tokens": [],
            "is_broadened": bool(effective_k > self.results_per_query),
            "base_k": int(self.results_per_query),
            "effective_k": int(effective_k),
            "decision": decision,
            "explicit_recheck": bool(explicit_recheck),
        }
        collapse_threshold = float(
            self.search_policy.get("intent_collapse_similarity", 0.55)
        )
        current_tokens = set(intent_key.split())
        if intent_key in self.query_memory:
            prior = self.query_memory.get(intent_key, {})
            diagnostics["classification"] = "exact_redundant"
            diagnostics["prior_query"] = str(prior.get("query", ""))
            diagnostics["similarity"] = 1.0
            self._log(
                "Query intent: redundant exact-match to prior intent. "
                f"query='{self._trim_text(query, 140)}' "
                f"prior_query='{self._trim_text(str(prior.get('query', '')), 140)}' "
                f"attempts={int(prior.get('attempts', 0))} "
                f"decision={decision} explicit_recheck={explicit_recheck}"
            )
        else:
            nearest_key, nearest_entry, score = self._nearest_prior_intent(intent_key)
            if nearest_key and score >= collapse_threshold:
                nearest_tokens = set(nearest_key.split())
                new_tokens = sorted(current_tokens - nearest_tokens)
                dropped_tokens = sorted(nearest_tokens - current_tokens)
                diagnostics["classification"] = "near_duplicate_or_paraphrase"
                diagnostics["prior_query"] = str(nearest_entry.get("query", ""))
                diagnostics["similarity"] = float(score)
                diagnostics["new_tokens"] = new_tokens[:10]
                diagnostics["dropped_tokens"] = dropped_tokens[:10]
                self._log(
                    "Query intent: near-duplicate/paraphrase of prior intent. "
                    f"query='{self._trim_text(query, 140)}' "
                    f"prior_query='{self._trim_text(str(nearest_entry.get('query', '')), 140)}' "
                    f"similarity={score:.2f} "
                    f"new_tokens={new_tokens[:10]} dropped_tokens={dropped_tokens[:10]}"
                )
            elif nearest_key:
                nearest_tokens = set(nearest_key.split())
                new_tokens = sorted(current_tokens - nearest_tokens)
                diagnostics["classification"] = "novel_vs_prior"
                diagnostics["prior_query"] = str(nearest_entry.get("query", ""))
                diagnostics["similarity"] = float(score)
                diagnostics["new_tokens"] = new_tokens[:10]
                self._log(
                    "Query intent: novel relative to nearest prior intent. "
                    f"query='{self._trim_text(query, 140)}' "
                    f"nearest_prior='{self._trim_text(str(nearest_entry.get('query', '')), 140)}' "
                    f"similarity={score:.2f} new_tokens={new_tokens[:10]}"
                )
            else:
                self._log(
                    "Query intent: first intent in this run. "
                    f"query='{self._trim_text(query, 140)}'"
                )

        if effective_k > self.results_per_query:
            self._log(
                "Execution plan: broadened result set requested. "
                f"base_k={self.results_per_query} effective_k={effective_k} decision={decision}"
            )
        else:
            self._log(
                "Execution plan: standard result set. "
                f"k={effective_k} decision={decision}"
            )
        return diagnostics

    def _is_time_sensitive(self, query: str) -> bool:
        normalized = self._normalize_query_intent(query)
        terms = self.search_policy.get("time_sensitive_terms", [])
        if not normalized:
            return False
        padded = f" {normalized} "
        for term in terms:
            if not isinstance(term, str):
                continue
            term_norm = self._normalize_query_intent(term)
            if not term_norm:
                continue
            if f" {term_norm} " in padded:
                return True
        return False

    def _decide_query_execution(
        self, intent_key: str, explicit_recheck: bool
    ) -> tuple[bool, str]:
        entry = self.query_memory.get(intent_key)
        if not entry:
            return True, "new_query"

        max_no_gain = int(self.search_policy.get("max_no_gain_retries_per_intent", 2))
        if int(entry.get("no_gain_streak", 0)) >= max_no_gain:
            return False, "blocked_diminishing_returns"

        if explicit_recheck:
            return True, "explicit_recheck"

        if self._is_time_sensitive(entry.get("query", intent_key)):
            return True, "time_sensitive"

        if bool(entry.get("search_error")) and bool(
            self.search_policy.get("allow_rerun_on_search_error", True)
        ):
            return True, "previous_search_error"

        ttl = int(self.search_policy.get("cache_ttl_seconds", 3600))
        last_ts = float(entry.get("last_ts", 0.0))
        if ttl > 0 and (time.time() - last_ts) > ttl:
            return True, "cache_expired"

        return False, "cache_hit"

    def _effective_results_per_query(self, intent_key: str) -> tuple[int, int]:
        entry = self.query_memory.get(intent_key, {})
        attempts = int(entry.get("attempts", 0))
        multipliers = self.search_policy.get("broaden_k_multipliers", [1, 2, 3])
        if not isinstance(multipliers, list) or not multipliers:
            multipliers = [1, 2, 3]
        max_steps = int(self.search_policy.get("max_broaden_steps", 2))
        step = min(attempts, max_steps, len(multipliers) - 1)
        mult = max(1, int(multipliers[step]))
        return self.results_per_query * mult, step

    def _update_query_memory(
        self,
        intent_key: str,
        query: str,
        round_i: int,
        effective_k: int,
        selected_results: List[EvaluatedResult],
        search_error: str,
        rerun_reason: Optional[str],
        new_fact_gain: int,
    ) -> None:
        entry = self.query_memory.get(intent_key, {})
        attempts = int(entry.get("attempts", 0)) + 1
        no_gain_streak = int(entry.get("no_gain_streak", 0))
        min_gain = int(self.search_policy.get("min_new_fact_gain", 1))
        if new_fact_gain < min_gain:
            no_gain_streak += 1
        else:
            no_gain_streak = 0
        self.query_memory[intent_key] = {
            "query": query,
            "last_round": round_i,
            "last_ts": time.time(),
            "attempts": attempts,
            "effective_k": effective_k,
            "selected_results_count": len(selected_results),
            "primary_count": sum(
                1 for r in selected_results if r.quality_tier == "primary_or_official"
            ),
            "search_error": search_error,
            "rerun_reason": rerun_reason or "",
            "new_fact_gain": new_fact_gain,
            "no_gain_streak": no_gain_streak,
        }

    def _query_memory_summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.query_memory.items():
            out[k] = {
                "query": v.get("query", ""),
                "attempts": int(v.get("attempts", 0)),
                "last_round": int(v.get("last_round", 0)),
                "last_ts": float(v.get("last_ts", 0.0)),
                "effective_k": int(v.get("effective_k", self.results_per_query)),
                "selected_results_count": int(v.get("selected_results_count", 0)),
                "primary_count": int(v.get("primary_count", 0)),
                "search_error": v.get("search_error", ""),
                "rerun_reason": v.get("rerun_reason", ""),
                "new_fact_gain": int(v.get("new_fact_gain", 0)),
                "no_gain_streak": int(v.get("no_gain_streak", 0)),
            }
        return out

    def _normalize_llm_list(self, value: Any, field_name: str) -> List[str]:
        normalized = as_clean_str_list(value)
        if not isinstance(value, list):
            self._log(
                f"LLM field '{field_name}' is non-list; coercing to empty list."
            )
        elif len(normalized) != len(value):
            self._log(
                f"LLM field '{field_name}' contained invalid/duplicate entries; cleaned."
            )
        return normalized

    def _assign_follow_up_queries(
        self, questions: List[str], follow_up_queries: List[str]
    ) -> Dict[str, List[str]]:
        assignments: Dict[str, List[str]] = {q: [] for q in questions}
        if not questions:
            return assignments
        for query in self._dedupe_preserve_order(follow_up_queries):
            best_q = self._best_match_question(query, questions)
            assignments[best_q].append(query)
        return assignments

    def _best_match_question(self, query: str, questions: List[str]) -> str:
        query_tokens = self._tokenize(query)
        best_question = questions[0]
        best_score = -1
        for question in questions:
            q_tokens = self._tokenize(question)
            score = len(query_tokens & q_tokens)
            if score > best_score:
                best_score = score
                best_question = question
        return best_question

    def _tokenize(self, text: str) -> set[str]:
        out: set[str] = set()
        for tok in self._normalize_query_intent(text).split():
            if self._contains_cjk(tok):
                out.add(tok)
                cjk_chars = [ch for ch in tok if self._is_cjk_char(ch)]
                if len(cjk_chars) >= 2:
                    for i in range(len(cjk_chars) - 1):
                        out.add("".join(cjk_chars[i : i + 2]))
                continue
            if len(tok) > 2:
                out.add(tok)
        return out

    def _query_step_key(self, round_i: int, sub_question: str, query: str) -> str:
        return f"{round_i}|{sub_question.strip()}|{query.strip().lower()}"

    def _serialize_findings(
        self, findings: Dict[str, SubQuestionFinding]
    ) -> Dict[str, Any]:
        return {
            k: {
                "sub_question": v.sub_question,
                "summaries": v.summaries,
                "facts": v.facts,
                "uncertainties": v.uncertainties,
            }
            for k, v in findings.items()
        }

    def _deserialize_findings(self, data: Any) -> Dict[str, SubQuestionFinding]:
        out: Dict[str, SubQuestionFinding] = {}
        if not isinstance(data, dict):
            return out
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            sq = str(v.get("sub_question", k))
            facts_raw = v.get("facts", [])
            facts: List[Dict[str, Any]] = []
            if isinstance(facts_raw, list):
                for item in facts_raw:
                    if isinstance(item, dict):
                        facts.append(item)
            out[str(k)] = SubQuestionFinding(
                sub_question=sq,
                summaries=as_clean_str_list(v.get("summaries", [])),
                facts=facts,
                uncertainties=as_clean_str_list(v.get("uncertainties", [])),
            )
        return out

    def _resolve_state_path(self, task: str, started_at: str) -> Path:
        if self.resume_from:
            return Path(self.resume_from)
        if self.state_file:
            return Path(self.state_file)
        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        ts = started_at.replace(":", "").replace("-", "").split(".")[0].replace("+00", "Z")
        slug = slugify_for_filename(task, max_len=50)
        return runs_dir / f"state_{slug}_{ts}.json"

    def _load_state(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Resume state file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid state file format: {path}")
        return data

    def _save_state(
        self,
        state_path: Path,
        status: str,
        task: str,
        started_at: str,
        next_round: int,
        sub_questions: List[str],
        success_criteria: List[str],
        findings: Dict[str, SubQuestionFinding],
        extra_questions: List[str],
        extra_queries: List[str],
        recheck_queries: List[str],
        unresolved_questions: List[str],
        question_depths: Dict[str, int],
        question_parents: Dict[str, str],
        question_node_ids: Dict[str, str],
        decomposed_questions: set[str],
        resolved_questions: set[str],
        final_suff: Dict[str, Any],
        total_search_calls: int,
        failed_search_calls: int,
        queries_with_evidence: int,
        total_selected_results: int,
        query_history: List[Dict[str, Any]],
        completed_query_steps: set[str],
        trace: Dict[str, Any],
        error: str = "",
    ) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "status": status,
            "task": task,
            "model": self.llm.model,
            "decompose_model": self.decompose_llm.model,
            "report_model": self.report_llm.model,
            "started_at": started_at,
            "updated_at": self._now_iso(),
            "next_round": next_round,
            "sub_questions": sub_questions,
            "success_criteria": success_criteria,
            "findings": self._serialize_findings(findings),
            "extra_questions": extra_questions,
            "extra_queries": extra_queries,
            "recheck_queries": recheck_queries,
            "unresolved_questions": unresolved_questions,
            "question_depths": question_depths,
            "question_parents": question_parents,
            "question_node_ids": question_node_ids,
            "decomposed_questions": sorted(decomposed_questions),
            "resolved_questions": sorted(resolved_questions),
            "final_sufficiency": final_suff,
            "search_stats": {
                "total_calls": total_search_calls,
                "failed_calls": failed_search_calls,
                "queries_with_evidence": queries_with_evidence,
                "total_selected_results": total_selected_results,
            },
            "query_history": query_history,
            "completed_query_steps": sorted(completed_query_steps),
            "query_memory": self._query_memory_summary(),
            "token_usage": self.usage_tracker.to_dict(),
            "trace": trace,
            "error": error,
        }
        tmp = state_path.with_suffix(state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(state_path)

    def _write_trace(self, trace: Dict[str, Any]) -> str:
        if self.trace_file:
            path = Path(self.trace_file)
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            runs_dir = Path("runs")
            runs_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = runs_dir / f"research_trace_{ts}.json"
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _write_usage_report(self, usage: Dict[str, Any]) -> str:
        path = Path(self.usage_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _print_token_summary(self, usage: Dict[str, Any]) -> None:
        if not self.token_breakdown:
            return
        total = usage.get("total", {})
        by_stage = usage.get("by_stage", {})
        print("[usage] token breakdown")
        print(
            "[usage] total "
            f"in={total.get('input_tokens', 0)} "
            f"out={total.get('output_tokens', 0)} "
            f"all={total.get('total_tokens', 0)} "
            f"calls={total.get('calls', 0)} "
            f"cost_usd={total.get('estimated_cost_usd', 0.0)}"
        )
        top = sorted(
            [
                (stage, stats.get("total_tokens", 0), stats.get("estimated_cost_usd", 0.0))
                for stage, stats in by_stage.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:6]
        for stage, tok, cost in top:
            print(f"[usage] stage={stage} tokens={tok} cost_usd={cost}")

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[progress] {message}")

    def _abort_if_requested(self, context: str) -> None:
        if self.should_abort and self.should_abort():
            self._log(f"Abort requested ({context}).")
            raise RuntimeError("Run aborted by user")

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.event_callback:
            return
        event = {
            "run_id": self.run_id,
            "timestamp": self._now_iso(),
            "event_type": event_type,
            "payload": payload,
        }
        try:
            self.event_callback(event)
        except Exception:
            # Observability must not break core execution flow.
            pass

    def _select_high_quality_results(
        self, results: List[SearchResult], max_results: int
    ) -> List[EvaluatedResult]:
        evaluated = [self._evaluate_source_quality(r) for r in results]
        evaluated.sort(key=lambda x: x.quality_score, reverse=True)
        return evaluated[:max_results]

    def _evaluate_source_quality(self, result: SearchResult) -> EvaluatedResult:
        domain = (urlparse(result.url).hostname or "").lower()
        score = 0
        tier = "other"

        primary_tlds = tuple(self.source_policy.get("primary_tlds", []))
        primary_domain_suffixes = tuple(
            self.source_policy.get("primary_domain_suffixes", [])
        )
        secondary_tlds = tuple(self.source_policy.get("secondary_tlds", []))

        if primary_tlds and domain.endswith(primary_tlds):
            score += 100
            tier = "primary_or_official"
        elif primary_domain_suffixes and any(
            matches_domain_rule(domain, tok) for tok in primary_domain_suffixes
        ):
            score += 100
            tier = "primary_or_official"
        elif secondary_tlds and domain.endswith(secondary_tlds):
            score += 70 if domain.endswith(".org") else 45
            tier = "secondary_reputable"
        else:
            score += 30

        snippet_lower = result.snippet.lower()
        title_lower = result.title.lower()
        if "press release" in snippet_lower or "sponsored" in snippet_lower:
            score -= 15
        if "wikipedia" in domain:
            score -= 10
        if "research" in title_lower or "report" in title_lower or "paper" in title_lower:
            score += 8
        if "blog" in domain:
            score -= 10

        return EvaluatedResult(
            title=result.title,
            snippet=result.snippet,
            url=result.url,
            quality_tier=tier,
            quality_score=score,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Research Agent")
    parser.add_argument("task", help="Research task prompt")
    parser.add_argument(
        "--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1"), help="Research model"
    )
    parser.add_argument(
        "--report-model",
        default=os.getenv("OPENAI_REPORT_MODEL", "gpt-5.2"),
        help="Final report model (also used for decomposition)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Max recursive decomposition depth per question node",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=1, help="Max iterative research rounds"
    )
    parser.add_argument(
        "--results-per-query", type=int, default=3, help="Baseline search results per query"
    )
    parser.add_argument(
        "--trace-file",
        default="",
        help="Optional path for execution trace JSON output",
    )
    parser.add_argument(
        "--state-file",
        default="",
        help="Optional path for incremental run state checkpoint JSON",
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help="Resume from a previously saved run state checkpoint JSON",
    )
    parser.add_argument(
        "--report-file",
        default="",
        help="Optional path for final report output (markdown text)",
    )
    parser.add_argument(
        "--usage-file",
        default="",
        help="Optional path for standalone token usage report JSON",
    )
    parser.add_argument(
        "--pricing-file",
        default=os.getenv("OPENAI_PRICING_FILE", "pricing.json"),
        help="Pricing config JSON used for cost estimation",
    )
    parser.add_argument(
        "--no-token-breakdown",
        action="store_true",
        help="Disable token usage tracking and summary output",
    )
    parser.add_argument(
        "--no-cost-estimate",
        action="store_true",
        help="Disable USD cost estimation while keeping token tracking",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress updates during execution (default: enabled)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress updates",
    )
    args = parser.parse_args()

    agent = DeepResearchAgent(
        model=args.model,
        report_model=args.report_model,
        max_depth=args.max_depth,
        max_rounds=args.max_rounds,
        results_per_query=args.results_per_query,
        trace_file=args.trace_file,
        state_file=args.state_file,
        resume_from=args.resume_from,
        token_breakdown=not args.no_token_breakdown,
        usage_file=args.usage_file,
        pricing_file=args.pricing_file,
        cost_estimate_enabled=not args.no_cost_estimate,
        verbose=(args.verbose and not args.quiet),
    )
    report = agent.run(args.task)
    if args.report_file:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = slugify_for_filename(args.task)
        report_path = reports_dir / f"report_{slug}_{ts}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[report] saved: {report_path}")
    print(report)


if __name__ == "__main__":
    main()
