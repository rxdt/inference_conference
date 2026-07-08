"""Reusable intent inference: task_type ranking + domain/specialty tagging.

Wraps the two VERIFIED spaCy mechanisms so the CLI (and tests) share one path:

- ``task_type`` via exp02's ``LexScorer`` (lemma/IDF cosine over per-task docs).
- ``domain`` / ``specialty`` via exp25's config-vocabulary ``PhraseMatcher``
  (F1: domain 0.840, specialty 0.639; adversarially reviewed non-circular).

The matcher-building logic is IMPORTED from exp25 — never duplicated here. config
is touched only through exp25's runtime import; no config/prompt text is emitted.

spaCy + builtins only. No regex, no external models. Model loads REPO-LOCAL.
"""

from experiments.exp02_lexoverlap import lexoverlap as lx
from experiments.exp25_domain import run as exp25
from lib import spacy_env
from spacy.language import Language


class IntentInferer:
    """Infer task_type ranking + domain/specialty labels for a free-text prompt.

    Built once (loads the model, task-docs, and both config matchers), then reused
    across prompts. Deterministic: every underlying mechanism is order-stable.
    """

    def __init__(self, nlp: Language | None = None) -> None:
        self._nlp = nlp or spacy_env.load_nlp()
        # use_cache=True: load the persisted task-doc lemmatization when unchanged,
        # so the CLI starts in seconds instead of re-lemmatizing 56 docs (~90s).
        self._scorer = lx.LexScorer(self._nlp, lx.build_task_docs(), use_cache=True)
        self._dom_pm, self._dom_id2label, _ = exp25._build_matcher(
            self._nlp, "DOMAINS"
        )
        self._spec_pm, self._spec_id2label, _ = exp25._build_matcher(
            self._nlp, "SPECIALTIES"
        )

    def rank_tasks(self, prompt: str) -> list[str]:
        """Task_types ranked best-first (empty list = abstain)."""
        return self._scorer.rank(prompt)

    def domains(self, prompt: str) -> set[str]:
        """Config-vocab domain labels present in the prompt (may be empty)."""
        return exp25._predict(self._nlp, self._dom_pm, self._dom_id2label, prompt)

    def specialties(self, prompt: str) -> set[str]:
        """Config-vocab specialty labels present in the prompt (may be empty)."""
        return exp25._predict(self._nlp, self._spec_pm, self._spec_id2label, prompt)
