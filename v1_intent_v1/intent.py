"""Public intent-extraction API.

This module exposes two callable extractors. Each takes one free-text prompt and
returns a ranking of Hugging Face "task types" (for example ``text-generation``
or ``text-ranking``), best match first. Both wrap the best honest, audited
methods developed in the experiment harnesses (see RESULT.md / EXPERIMENTS.md);
the winning scorers are reused read-only rather than reimplemented.

Everything runs locally with spaCy (the ``en_core_web_md`` pipeline) plus plain
Python. There is no machine-learning training, no large-language-model call, no
remote inference, and no regular expression anywhere. Results are deterministic:
ties are broken alphabetically by task name, not by Python's hash ordering.

    from intent import extract_intent, extract_intent_by_matching_to_db_metadata

    extract_intent("a math tester for children in school")["task_type"][0]
    # -> 'text-generation'

Two complementary strategies are provided:

* ``extract_intent`` matches the prompt against the curated taxonomy in
  ``config.py``. Best honest accuracy: 42.69% top-1 / 44.17% in-list on the
  2162-prompt evaluation set.
* ``extract_intent_by_matching_to_db_metadata`` matches the prompt against the
  metadata of ~13k real models in the bundled database. Best honest accuracy:
  22.85% top-1 / 23.54% in-list. It needs the model-vector cache that
  ``build_corpus.py`` writes to ``.cache_db/`` on first use.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy
import spacy
from spacy.language import Language
from spacy.tokens import Doc

import buckets
import harness_db as database_matcher  # approach #2: matching against DB metadata
import harness_ensemble as taxonomy_ensemble  # approach #1: matching against config
import harness_gate as bucket_gate  # config-derived coarse-bucket signal


@lru_cache(maxsize=1)
def load_spacy_pipeline() -> Language:
    """Load (once) the spaCy pipeline used for taxonomy matching. Singleton pattern + the result is cached so spaCy model loads only on the first call.

    ```exclude=["ner", "parser"]```: The named-entity recogniser and the dependency parser are excluded because
    the shipped problem-#1 signals (lemma tf-idf, lexical-idf, and the soft
    bucket bonus) only ever read lemmas and part-of-speech tags. Dropping the
    parser is ~20% faster and produces byte-identical predictions
    (verified: still 42.69% / 44.17%).

    Returns:
        The loaded spaCy ``Language`` pipeline.

    """
    return spacy.load("en_core_web_md", exclude=["ner", "parser"])


def build_result(ranked_tasks: list[tuple[str, float]]) -> dict[str, Any]:
    """Package a ranked list of tasks into the public return shape.

    Args:
        ranked_tasks: ``(task_type, score)`` pairs already sorted best-first.

    Returns:
        A dictionary with ``"task_type"`` (the ranked task names) and
        ``"scores"`` (the task-name to score mapping).

    """
    return {
        "task_type": [entry[0] for entry in ranked_tasks],
        "scores": dict(ranked_tasks),
    }


def max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Scale a score mapping so the largest value becomes 1.0.

    Max-normalisation puts differently-scaled signals on a common 0..1 range so
    they can be added together fairly.

    Args:
        scores: Mapping from task name to a raw score.

    Returns:
        The scores divided by their maximum, or all zeros when every score is
        zero (or the mapping is empty).

    """
    highest_score = max(scores.values()) if scores else 0.0
    if highest_score <= 0.0:
        return dict.fromkeys(scores, 0.0)
    return {task_type: value / highest_score for task_type, value in scores.items()}


def database_query_lemmas(query_document: Doc) -> set[str]:
    """Return the content lemmas of a prompt for database-metadata matching.

    This mirrors the lemma rule used by ``harness_db`` when it indexed the model
    descriptions, so the prompt side and the model side are compared with exactly
    the same definition of a "content word". Keeping the two identical is what
    makes the lexical overlap meaningful; do not change one without the other.

    Args:
        query_document: A spaCy document for the user prompt.

    Returns:
        Lowercased lemmas of alphabetic, non-stop-word, vector-bearing tokens
        longer than two characters.

    """
    return {
        token.lemma_.lower()
        for token in query_document
        if token.is_alpha and not token.is_stop and len(token) > 2 and token.has_vector
    }


# --------------------------------------------------------------------------
# Approach #1 — match the prompt against the curated taxonomy in config.py.
# --------------------------------------------------------------------------
def extract_intent(prompt: str) -> dict[str, Any]:
    """Rank the task types for a prompt by matching it against ``config.py``.

    This is the project's best honest method: a max-normalised score-sum of two
    config-derived lemma tf-idf rankers plus a soft coarse-bucket affinity bonus
    (the exact signal set is ``harness_ensemble.BEST_SIGNALS``).

    Args:
        prompt: The free-text user request.

    Returns:
        ``{"task_type": [ranked task names], "scores": {task name: score}}``.

    """
    query_document = load_spacy_pipeline()(prompt)

    # Compute each winning signal's ranking over all task types for this prompt.
    signal_rankings = {
        signal_name: taxonomy_ensemble.SIGNALS[signal_name](query_document)
        for signal_name in taxonomy_ensemble.BEST_SIGNALS
    }

    # Fuse the signals. The score-sum operator reads each signal's value by task
    # (so signal order does not matter) and the final ranking is sorted by
    # (-score, task name); ties therefore break alphabetically and stably rather
    # than by Python's hash-randomised set iteration.
    ranked_tasks = taxonomy_ensemble.fuse(
        signal_rankings,
        taxonomy_ensemble.BEST_SIGNALS,
        taxonomy_ensemble.BEST_WEIGHTS,
        taxonomy_ensemble.BEST_OP,
    )
    return build_result(ranked_tasks)


# --------------------------------------------------------------------------
# Approach #2 — match the prompt against the bundled model database.
# --------------------------------------------------------------------------
def extract_intent_by_matching_to_db_metadata(
    prompt: str, neighbor_count: int = 50
) -> dict[str, Any]:
    """Rank task types for a prompt by matching it against database model metadata.

    This is the single-prompt form of ``harness_db.score_fusion_softbucket`` (the
    best honest problem-#2 method). Three signals are combined:

    1. A nearest-neighbour vote: the prompt's vector is compared by cosine
       similarity to every model's vector, and the closest models "vote" for
       their own task type.
    2. A lexical overlap score: distinctive, inverse-document-frequency-weighted
       lemmas the prompt shares with each task's model descriptions.
    3. A soft coarse-bucket affinity bonus derived only from ``config.py``, which
       nudges (but never forces) the ranking toward the predicted bucket.

    The three are max-normalised and summed. Prompt-independent data (the model
    vectors and the per-task lemma weights) is reused from ``harness_db``'s caches.

    Args:
        prompt: The free-text user request.
        neighbor_count: How many nearest model vectors vote (the "k" of k-NN).

    Returns:
        ``{"task_type": [ranked task names], "scores": {task name: score}}``.

    """
    all_task_types = sorted(database_matcher.CANON)
    query_document = database_matcher.nlp()(prompt)

    # Signal 1 -- nearest-neighbour vote over the model vectors. Both the prompt
    # vector and the cached model vectors are unit length, so the dot product is
    # cosine similarity.
    query_vector = database_matcher.l2norm(
        database_matcher.task_vec(query_document)[None, :]
    )[0]
    model_vectors, model_task_types = database_matcher.corpus()
    similarities = query_vector @ model_vectors.T
    effective_neighbor_count = min(neighbor_count, similarities.size)
    neighbor_votes: dict[str, float] = {}
    nearest_model_indices = numpy.argpartition(
        -similarities, effective_neighbor_count - 1
    )[:effective_neighbor_count]
    for model_index in nearest_model_indices:
        voted_task = model_task_types[model_index]
        neighbor_votes[voted_task] = neighbor_votes.get(voted_task, 0.0) + float(
            similarities[model_index]
        )

    # Signal 2 -- lexical overlap with each task's distinctive metadata lemmas.
    task_lemma_weights = database_matcher.task_term_idf()
    query_lemmas = database_query_lemmas(query_document)
    lexical_scores = {
        task_type: sum(
            weight
            for lemma, weight in task_lemma_weights[task_type].items()
            if lemma in query_lemmas
        )
        for task_type in all_task_types
        if task_type in task_lemma_weights
    }

    # Signal 3 -- soft, config-derived coarse-bucket affinity for this prompt.
    bucket_affinity = max_normalize(bucket_gate.gate_scores(bucket_gate.nlp()(prompt)))

    # Fuse the three signals: max-normalise each, then add (the bucket bonus is
    # down-weighted so it only nudges).
    normalized_neighbor_votes = max_normalize(neighbor_votes)
    normalized_lexical_scores = max_normalize(lexical_scores)
    fused_scores = {
        task_type: normalized_neighbor_votes.get(task_type, 0.0)
        + normalized_lexical_scores.get(task_type, 0.0)
        + database_matcher.SOFT_BONUS_W
        * bucket_affinity.get(buckets.TASK_TO_BUCKET[task_type], 0.0)
        for task_type in all_task_types
    }

    # Sort best-first; break ties by (-score, task name) so equal or all-zero
    # scores resolve alphabetically and stably across runs.
    ranked_tasks = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
    return build_result(ranked_tasks)


if __name__ == "__main__":
    # Quick self-evaluation: run both extractors over the golden prompts and
    # print top-1 and in-list accuracy. This reproduces the headline numbers.
    from prompts import PROMPT_MATRIX

    def measure_accuracy(
        extractor: Callable[[str], dict[str, Any]],
    ) -> tuple[float, float]:
        """Return (top-1, in-list) accuracy of an extractor over the golden set.

        Args:
            extractor: A function mapping a prompt to a result dictionary.

        Returns:
            The fraction predicted exactly right, and the fraction whose top
            prediction appears anywhere in the expected task-type list.

        """
        correct_count = 0
        within_expected_count = 0
        for prompt_entry in PROMPT_MATRIX:
            ranked_task_types = extractor(prompt_entry["prompt"])["task_type"]
            expected_task_types = prompt_entry["expected_task_type"]
            if ranked_task_types[0] == expected_task_types[0]:
                correct_count += 1
            if ranked_task_types[0] in expected_task_types:
                within_expected_count += 1
        total_prompts = len(PROMPT_MATRIX)
        return correct_count / total_prompts, within_expected_count / total_prompts

    taxonomy_top1, taxonomy_in_list = measure_accuracy(extract_intent)
    print(
        f"extract_intent (approach #1):                    "
        f"top1={taxonomy_top1:.2%} in-list={taxonomy_in_list:.2%}"
    )
    database_top1, database_in_list = measure_accuracy(
        extract_intent_by_matching_to_db_metadata
    )
    print(
        f"extract_intent_by_matching_to_db_metadata (#2): "
        f"top1={database_top1:.2%} in-list={database_in_list:.2%}"
    )
