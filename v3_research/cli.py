"""End-to-end intent → model CLI (spaCy-only finalist).

Pipeline:
  prompt --(lexical scorer, exp02)------> ranked task_types
         --(config-vocab matcher, exp25)-> inferred domain(s) / specialty(ies)
         --(DB lookup)--------------------> best-matching models, domain-aware

The task scorer is the pre-committed finalist: lemma/IDF cosine of the prompt
against per-task documents (clean DB + HF supervision). Domain/specialty come from
exp25's VERIFIED config-vocabulary PhraseMatcher (domain F1=0.840). When a domain
is inferred, DB models tagged with that domain are preferred over task-only matches.
No prompt text, no config.py contents, no regex, no external models. Model loads
REPO-LOCAL via spacy_env.

Usage:
  uv run --no-sync python src/research/cli.py "transcribe my voicemails to text"
  uv run --no-sync python src/research/cli.py         # then type at the prompt
"""

import json
import sys
from time import perf_counter
from pathlib import Path
from typing import TypedDict

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from lib import db, intent  # noqa: E402

_TOP_TASKS = 3
_TOP_MODELS = 5


class Result(TypedDict):
    """One prompt's inference: tasks, inferred domain/specialty, and models."""

    prompt: str
    task_types: list[str]
    top_task: str | None
    domains: list[str]
    specialties: list[str]
    abstained: bool
    models: list[dict[str, object]]
    profile: dict[str, float]


_inferer = None


def _get_inferer() -> intent.IntentInferer:
    """Build the intent inferer once (task scorer + config matchers)."""
    global _inferer
    if _inferer is None:
        _inferer = intent.IntentInferer()
    return _inferer


def _domain_match(domains_json: str | None, want: set[str]) -> bool:
    """True if a model's ``domains`` JSON-list cell overlaps ``want``."""
    if not want or not domains_json:
        return False
    try:
        vals = json.loads(domains_json)
    except (ValueError, TypeError):
        return False
    return isinstance(vals, list) and bool(want & set(vals))


def top_models(
    task: str, domains: set[str] | None = None, n: int = _TOP_MODELS
) -> list[dict[str, object]]:
    """Top-n DB models for a task_type, domain-aware and deterministic.

    Models whose ``domains`` column overlaps ``domains`` rank first; the rest fall
    back to task-only matches. Within each group, order by downloads desc, likes
    desc, id asc. Read-only. Returns id/description/likes/downloads/pipeline_tag/
    proprietary/domains plus a ``domain_match`` flag.
    """
    want = domains or set()
    con = db.connect()
    try:
        rows = con.execute(
            "SELECT id, pipeline_tag, short_description, likes, downloads, "
            "proprietary, domains FROM models WHERE task_type LIKE ? "
            "ORDER BY COALESCE(downloads,0) DESC, COALESCE(likes,0) DESC, id ASC",
            (f'%"{task}"%',),
        ).fetchall()
    finally:
        con.close()
    # Stable partition: domain matches first, task-only after. Both halves keep the
    # SQL order (downloads, likes, id), so the result stays deterministic.
    matched: list[dict[str, object]] = []
    rest: list[dict[str, object]] = []
    for r in rows:
        d = dict(r)
        d["domain_match"] = _domain_match(d.get("domains"), want)
        (matched if d["domain_match"] else rest).append(d)
    return (matched + rest)[:n]


def infer(prompt: str) -> Result:
    """Infer task_type ranking, domain/specialty, and best models for a prompt.

    Records per-stage wall-clock timing (ms) under ``profile`` — the inference
    stages only (model/task-doc build is a one-time startup cost, reported
    separately by main()).
    """
    inf = _get_inferer()
    prof: dict[str, float] = {}

    t = perf_counter()
    ranked = inf.rank_tasks(prompt)
    prof["task_rank_ms"] = (perf_counter() - t) * 1000.0
    tasks = ranked[:_TOP_TASKS]
    top = tasks[0] if tasks else None

    t = perf_counter()
    domains = inf.domains(prompt)
    specialties = inf.specialties(prompt)
    prof["domain_specialty_ms"] = (perf_counter() - t) * 1000.0

    t = perf_counter()
    models = top_models(top, domains) if top else []
    prof["model_lookup_ms"] = (perf_counter() - t) * 1000.0

    prof["total_ms"] = sum(prof.values())
    return {
        "prompt": prompt,
        "task_types": tasks,
        "top_task": top,
        "domains": sorted(domains),
        "specialties": sorted(specialties),
        "abstained": not tasks,
        "models": models,
        "profile": prof,
    }


def _print(res: Result) -> None:
    print(f'\nprompt             : "{res["prompt"]}"')
    print(f"inferred domain    : {res['domains'] or '(none)'}")
    print(f"inferred specialties: {res['specialties'] or '(none)'}")
    if res["abstained"]:
        print("inferred task      : (abstained — no confident task match)")
        return
    print(f"top task           : {res['top_task']}")
    print(f"  alt tasks        : {res['task_types'][1:]}")
    hdr = f"--- top {len(res['models'])} models for {res['top_task']}"
    if res["domains"]:
        hdr += f" (domain-aware: {res['domains']})"
    print(hdr + " ---")
    for i, m in enumerate(res["models"], 1):
        tag = " [PROPRIETARY]" if m.get("proprietary") else ""
        dm = " [DOMAIN-MATCH]" if m.get("domain_match") else ""
        url = "" if m.get("proprietary") else f"  https://huggingface.co/{m['id']}"
        print(f"  {i}. {m['id']}{tag}{dm}{url}")
        print(
            f"       likes/dl={m.get('likes')}/{m.get('downloads')}"
            f"  pipeline={m.get('pipeline_tag')}"
        )
        raw = m.get("short_description")
        desc = raw.strip() if isinstance(raw, str) else ""
        if desc:
            print(f"       {desc[:160]}")
    p = res["profile"]
    print(
        f"--- profiling (ms) ---\n"
        f"  task_rank={p['task_rank_ms']:.2f}  domain/specialty="
        f"{p['domain_specialty_ms']:.2f}  model_lookup={p['model_lookup_ms']:.2f}"
        f"  TOTAL={p['total_ms']:.2f}"
    )
    print()


def main() -> None:
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        try:
            prompt = input("prompt> ").strip()
        except EOFError:
            prompt = ""
    if not prompt:
        print("no prompt given")
        return
    _print(infer(prompt))


if __name__ == "__main__":
    main()
