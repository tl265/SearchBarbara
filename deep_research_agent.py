import argparse
import json
import os
import time
import textwrap
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from openai import OpenAI, RateLimitError

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SOURCE_POLICY_PATH = Path(__file__).resolve().parent / "source_policy.json"


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


class LLM:
    def __init__(self, model: str) -> None:
        self.client = OpenAI()
        self.model = model
        self.max_retries = 3

    def json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        rsp = self._chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        text = rsp.choices[0].message.content or "{}"
        return json.loads(text)

    def text(self, system_prompt: str, user_prompt: str) -> str:
        rsp = self._chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return rsp.choices[0].message.content or ""

    def _chat_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
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
    def __init__(self, model: str) -> None:
        self.client = OpenAI()
        self.model = model
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
        max_rounds: int,
        results_per_query: int,
        trace_file: str = "",
        state_file: str = "",
        resume_from: str = "",
        verbose: bool = True,
    ) -> None:
        self.llm = LLM(model=model)
        self.report_llm = LLM(model=report_model)
        self.search = WebSearch(model=model)
        self.max_rounds = max_rounds
        self.results_per_query = results_per_query
        self.trace_file = trace_file
        self.state_file = state_file
        self.resume_from = resume_from
        self.verbose = verbose
        self.source_policy = load_source_policy()

    def run(self, task: str) -> str:
        self._log("Starting deep research run.")
        started_at = self._now_iso()
        state_path = self._resolve_state_path(task, started_at)
        query_history: List[Dict[str, Any]] = []
        completed_query_steps: set[str] = set()

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
            start_round = max(1, int(resume_state.get("next_round", 1)))
            self._log(
                f"Resuming from {state_path} at round {start_round}."
            )
        else:
            trace = {
                "task": task,
                "model": self.llm.model,
                "report_model": self.report_llm.model,
                "started_at": started_at,
                "max_rounds": self.max_rounds,
                "results_per_query": self.results_per_query,
                "rounds": [],
                "source_policy_file": str(SOURCE_POLICY_PATH),
            }
            self._log("Decomposing task into sub-questions and success criteria.")
            plan = self.llm.json(SYSTEM_DECOMPOSE, f"Task:\n{task}")
            sub_questions = self._normalize_llm_list(
                plan.get("sub_questions"), "sub_questions"
            )
            success_criteria = self._normalize_llm_list(
                plan.get("success_criteria"), "success_criteria"
            )
            self._log(
                f"Plan ready: {len(sub_questions)} sub-questions, {len(success_criteria)} success criteria."
            )
            trace["plan"] = {
                "sub_questions": sub_questions,
                "success_criteria": success_criteria,
            }
            findings = {sq: SubQuestionFinding(sub_question=sq) for sq in sub_questions}
            extra_questions = []
            extra_queries = []
            final_suff = {}
            total_search_calls = 0
            failed_search_calls = 0
            queries_with_evidence = 0
            total_selected_results = 0
            start_round = 1

        self._save_state(
            state_path=state_path,
            status="running",
            task=task,
            started_at=started_at,
            next_round=start_round,
            sub_questions=sub_questions,
            success_criteria=success_criteria,
            findings=findings,
            extra_questions=extra_questions,
            extra_queries=extra_queries,
            final_suff=final_suff,
            total_search_calls=total_search_calls,
            failed_search_calls=failed_search_calls,
            queries_with_evidence=queries_with_evidence,
            total_selected_results=total_selected_results,
            query_history=query_history,
            completed_query_steps=completed_query_steps,
            trace=trace,
        )

        current_round = start_round
        try:
            for round_i in range(start_round, self.max_rounds + 1):
                current_round = round_i
                self._log(f"Round {round_i}/{self.max_rounds}: gathering evidence.")
                round_trace: Dict[str, Any] = {
                    "round": round_i,
                    "questions": [],
                }
                all_questions = self._dedupe_preserve_order(
                    sub_questions + extra_questions
                )
                round_extra_queries_by_question = self._assign_follow_up_queries(
                    all_questions, extra_queries
                )
                for sq in all_questions:
                    self._log(f"Sub-question: {sq}")
                    question_trace: Dict[str, Any] = {
                        "sub_question": sq,
                        "query_steps": [],
                    }
                    query_prompt = f"""Original task:
{task}

Sub-question:
{sq}

Known success criteria:
{json.dumps(success_criteria, ensure_ascii=True)}
"""
                    qobj = self.llm.json(SYSTEM_QUERY_GEN, query_prompt)
                    generated_queries = self._normalize_llm_list(
                        qobj.get("queries"), "queries"
                    )
                    queries = self._dedupe_preserve_order(
                        generated_queries + round_extra_queries_by_question.get(sq, [])
                    )
                    self._log(f"Generated {len(queries)} search queries.")
                    for query in queries:
                        step_key = self._query_step_key(round_i, sq, query)
                        if step_key in completed_query_steps:
                            self._log(f"Skipping already completed query: {query}")
                            continue

                        self._log(f"Searching: {query}")
                        total_search_calls += 1
                        raw_results = self.search.search(query, self.results_per_query)
                        search_error = self.search.last_error
                        if search_error:
                            failed_search_calls += 1
                            self._log(
                                f"Search failed for this query; continuing. error={search_error}"
                            )
                        selected_results = self._select_high_quality_results(
                            raw_results, self.results_per_query
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
                        total_selected_results += len(selected_results)
                        if selected_results:
                            queries_with_evidence += 1
                        findings.setdefault(sq, SubQuestionFinding(sub_question=sq))

                        if not selected_results:
                            limitation = (
                                "No usable evidence retrieved for this query; "
                                "synthesis skipped due to insufficient grounding."
                            )
                            self._log(limitation)
                            findings[sq].uncertainties.append(limitation)
                            step_data = {
                                "round": round_i,
                                "sub_question": sq,
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
                            self._save_state(
                                state_path=state_path,
                                status="running",
                                task=task,
                                started_at=started_at,
                                next_round=round_i,
                                sub_questions=sub_questions,
                                success_criteria=success_criteria,
                                findings=findings,
                                extra_questions=extra_questions,
                                extra_queries=extra_queries,
                                final_suff=final_suff,
                                total_search_calls=total_search_calls,
                                failed_search_calls=failed_search_calls,
                                queries_with_evidence=queries_with_evidence,
                                total_selected_results=total_selected_results,
                                query_history=query_history,
                                completed_query_steps=completed_query_steps,
                                trace=trace,
                            )
                            continue

                        synth_prompt = self._format_synthesis_prompt(
                            sq, query, selected_results
                        )
                        sobj = self.llm.json(SYSTEM_SYNTHESIZE, synth_prompt)
                        self._log("Synthesis completed for this query.")
                        findings[sq].summaries.append(sobj.get("summary", ""))
                        findings[sq].facts.extend(sobj.get("facts", []))
                        findings[sq].uncertainties.extend(sobj.get("uncertainties", []))
                        step_data = {
                            "round": round_i,
                            "sub_question": sq,
                            "query": query,
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
                        }
                        question_trace["query_steps"].append(step_data)
                        query_history.append(step_data)
                        completed_query_steps.add(step_key)
                        self._save_state(
                            state_path=state_path,
                            status="running",
                            task=task,
                            started_at=started_at,
                            next_round=round_i,
                            sub_questions=sub_questions,
                            success_criteria=success_criteria,
                            findings=findings,
                            extra_questions=extra_questions,
                            extra_queries=extra_queries,
                            final_suff=final_suff,
                            total_search_calls=total_search_calls,
                            failed_search_calls=failed_search_calls,
                            queries_with_evidence=queries_with_evidence,
                            total_selected_results=total_selected_results,
                            query_history=query_history,
                            completed_query_steps=completed_query_steps,
                            trace=trace,
                        )

                    round_trace["questions"].append(question_trace)

                suff = self.llm.json(
                    SYSTEM_SUFFICIENCY,
                    self._format_sufficiency_prompt(
                        task=task,
                        success_criteria=success_criteria,
                        findings=findings,
                        round_i=round_i,
                    ),
                )
                final_suff = suff
                round_trace["sufficiency"] = suff
                trace["rounds"].append(round_trace)
                if suff.get("is_sufficient", False):
                    self._log("Sufficiency check passed. Stopping iterative search.")
                    self._save_state(
                        state_path=state_path,
                        status="running",
                        task=task,
                        started_at=started_at,
                        next_round=round_i,
                        sub_questions=sub_questions,
                        success_criteria=success_criteria,
                        findings=findings,
                        extra_questions=extra_questions,
                        extra_queries=extra_queries,
                        final_suff=final_suff,
                        total_search_calls=total_search_calls,
                        failed_search_calls=failed_search_calls,
                        queries_with_evidence=queries_with_evidence,
                        total_selected_results=total_selected_results,
                        query_history=query_history,
                        completed_query_steps=completed_query_steps,
                        trace=trace,
                    )
                    break
                self._log("Sufficiency check failed. Preparing follow-up search cycle.")
                extra_questions = self._normalize_llm_list(
                    suff.get("follow_up_questions"), "follow_up_questions"
                )
                extra_queries = self._normalize_llm_list(
                    suff.get("follow_up_queries"), "follow_up_queries"
                )
                self._save_state(
                    state_path=state_path,
                    status="running",
                    task=task,
                    started_at=started_at,
                    next_round=round_i + 1,
                    sub_questions=sub_questions,
                    success_criteria=success_criteria,
                    findings=findings,
                    extra_questions=extra_questions,
                    extra_queries=extra_queries,
                    final_suff=final_suff,
                    total_search_calls=total_search_calls,
                    failed_search_calls=failed_search_calls,
                    queries_with_evidence=queries_with_evidence,
                    total_selected_results=total_selected_results,
                    query_history=query_history,
                    completed_query_steps=completed_query_steps,
                    trace=trace,
                )

            trace["search_stats"] = {
                "total_calls": total_search_calls,
                "failed_calls": failed_search_calls,
                "queries_with_evidence": queries_with_evidence,
                "total_selected_results": total_selected_results,
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
            report_prompt = self._format_report_prompt(
                task=task,
                success_criteria=success_criteria,
                findings=findings,
                evidence_status=evidence_status,
                evidence_note=evidence_note,
                search_stats=trace["search_stats"],
            )
            report = self.report_llm.text(SYSTEM_REPORT, report_prompt)
            trace["finished_at"] = self._now_iso()
            trace["final_sufficiency"] = final_suff
            trace["report"] = report
            trace_path = self._write_trace(trace)
            print(f"[trace] saved: {trace_path}")
            self._save_state(
                state_path=state_path,
                status="completed",
                task=task,
                started_at=started_at,
                next_round=self.max_rounds + 1,
                sub_questions=sub_questions,
                success_criteria=success_criteria,
                findings=findings,
                extra_questions=extra_questions,
                extra_queries=extra_queries,
                final_suff=final_suff,
                total_search_calls=total_search_calls,
                failed_search_calls=failed_search_calls,
                queries_with_evidence=queries_with_evidence,
                total_selected_results=total_selected_results,
                query_history=query_history,
                completed_query_steps=completed_query_steps,
                trace=trace,
            )
            self._log("Run completed.")
            return report
        except Exception as exc:
            self._save_state(
                state_path=state_path,
                status="failed",
                task=task,
                started_at=started_at,
                next_round=current_round,
                sub_questions=sub_questions,
                success_criteria=success_criteria,
                findings=findings,
                extra_questions=extra_questions,
                extra_queries=extra_queries,
                final_suff=final_suff,
                total_search_calls=total_search_calls,
                failed_search_calls=failed_search_calls,
                queries_with_evidence=queries_with_evidence,
                total_selected_results=total_selected_results,
                query_history=query_history,
                completed_query_steps=completed_query_steps,
                trace=trace,
                error=str(exc),
            )
            raise

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
            {json.dumps(success_criteria, ensure_ascii=True)}

            Current findings:
            {json.dumps(compact, ensure_ascii=True)}
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
            {json.dumps(success_criteria, ensure_ascii=True)}

            Evidence quality:
            status={evidence_status}
            note={evidence_note}
            stats={json.dumps(search_stats, ensure_ascii=True)}

            Reporting rule:
            If evidence status is not "adequate", explicitly state this limitation in the
            Governing Thought and Risks/Unknowns sections, and avoid definitive claims.

            Evidence base:
            {json.dumps(compact, ensure_ascii=True)}
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

        payload = json.dumps(compact, ensure_ascii=True)
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
                    key=lambda kk: len(json.dumps(compact[kk], ensure_ascii=True)),
                )
                compact.pop(largest_key, None)
            payload = json.dumps(compact, ensure_ascii=True)
        return compact

    def _compact_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for fact in facts:
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
        return {
            t
            for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
            if len(t) > 2
        }

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
            out[str(k)] = SubQuestionFinding(
                sub_question=sq,
                summaries=as_clean_str_list(v.get("summaries", [])),
                facts=v.get("facts", []) if isinstance(v.get("facts", []), list) else [],
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
            "version": 1,
            "status": status,
            "task": task,
            "model": self.llm.model,
            "started_at": started_at,
            "updated_at": self._now_iso(),
            "next_round": next_round,
            "sub_questions": sub_questions,
            "success_criteria": success_criteria,
            "findings": self._serialize_findings(findings),
            "extra_questions": extra_questions,
            "extra_queries": extra_queries,
            "final_sufficiency": final_suff,
            "search_stats": {
                "total_calls": total_search_calls,
                "failed_calls": failed_search_calls,
                "queries_with_evidence": queries_with_evidence,
                "total_selected_results": total_selected_results,
            },
            "query_history": query_history,
            "completed_query_steps": sorted(completed_query_steps),
            "trace": trace,
            "error": error,
        }
        tmp = state_path.with_suffix(state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
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
        path.write_text(json.dumps(trace, ensure_ascii=True, indent=2))
        return str(path)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[progress] {message}")

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
        help="Final report model",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=4, help="Max iterative research rounds"
    )
    parser.add_argument(
        "--results-per-query", type=int, default=5, help="Search results per query"
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
        max_rounds=args.max_rounds,
        results_per_query=args.results_per_query,
        trace_file=args.trace_file,
        state_file=args.state_file,
        resume_from=args.resume_from,
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
