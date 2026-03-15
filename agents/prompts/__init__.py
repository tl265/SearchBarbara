from pathlib import Path


PROMPTS_ROOT = Path(__file__).resolve().parent

_LEGACY_PROMPT_MAP = {
    "live_intent.system.txt": "live_intent/system.txt",
    "decompose.system.txt": "deep_research/decompose.system.txt",
    "query_gen.system.txt": "deep_research/query_gen.system.txt",
    "synthesize.system.txt": "deep_research/synthesize.system.txt",
    "sufficiency_node.system.txt": "deep_research/sufficiency_node.system.txt",
    "sufficiency_pass.system.txt": "deep_research/sufficiency_pass.system.txt",
    "sufficiency.system.txt": "deep_research/sufficiency.system.txt",
    "report.system.txt": "deep_research/report.system.txt",
    "report.universal.system.txt": "deep_research/report.universal.system.txt",
    "report.background.executive.txt": "deep_research/report.background.executive.txt",
    "report.background.business_head_execution.txt": "deep_research/report.background.business_head_execution.txt",
    "context_split.system.txt": "context_preprocess/context_split.system.txt",
    "context_preprocess_common.system.txt": "context_preprocess/common.system.txt",
    "context_preprocess_per_file.system.txt": "context_preprocess/per_file.system.txt",
    "context_preprocess_aggregate.system.txt": "context_preprocess/aggregate.system.txt",
}


def prompt_root() -> Path:
    return PROMPTS_ROOT


def resolve_prompt_path(spec: str) -> Path:
    raw = str(spec or "").strip()
    if not raw:
        raise FileNotFoundError("Prompt path is empty.")
    direct = Path(raw)
    if direct.exists():
        return direct
    if raw.startswith("prompts/"):
        raw = raw[len("prompts/") :]
    if raw.startswith("agents/prompts/"):
        raw = raw[len("agents/prompts/") :]
    rel = _LEGACY_PROMPT_MAP.get(raw, raw)
    candidate = PROMPTS_ROOT / rel
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Missing prompt file: {spec}")


def load_prompt_text(spec: str) -> str:
    return resolve_prompt_path(spec).read_text(encoding="utf-8").strip()
