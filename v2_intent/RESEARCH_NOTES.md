# ⛔ SUPERSEDED — V1 ARCHIVE — DO NOT USE FOR REVIEW ⛔
> # ⚠️ OLD (V1) NOTES. Framing here (`prompt → HF task_type`) was CORRECTED to
> # semantic retrieval. Read **`RESEARCH_NOTES_V2.md`** (+ `_V3`) and **`README.md`**.
> # Kept as history only. **Claims here may be stale or misleading — do not cite.**

---

# RESEARCH NOTES — spaCy-only intent extraction  *(V1, archived)*

Problem framing: `free-text prompt → HF task_type (+ domain/specialty) → best DB model`.
Rules: spaCy `en_core_web_md` + Python builtins only. Forbidden in any solution:
regex, non-spaCy fuzzymatch, external/LLM/embedding APIs, runtime downloads, remote
inference. spaCy-native `Matcher` FUZZY is allowed. Cues may come only from DB/HF text,
never from config/prompts.

All sources below were fetched and checked on 2026-07-01. Verification status is per source.

---

## 1. spaCy docs (spaCy 3.x, API current for 3.8)

### 1a. Rule-based matching — VERIFIED (https://spacy.io/usage/rule-based-matching)
- (a) Reference for `Matcher`, `PhraseMatcher`, `DependencyMatcher`, `EntityRuler`.
- (b) Concrete mechanisms:
  - `Matcher(vocab)`; `.add(id, patterns, *, on_match=None)`; `matcher(doc)` → `(match_id, start, end)`.
    Token attrs: `LOWER, LEMMA, POS, TAG, DEP, SHAPE, MORPH`; ops `IN, NOT_IN, REGEX, FUZZY`,
    comparisons, quantifiers `!,?,+,*,{n},{n,m}`.
  - **FUZZY is spaCy-native (v3.5+)**: "The FUZZY attribute allows fuzzy matches for any
    attribute string value." Default Levenshtein distance 2 (≤30% of pattern length);
    `FUZZY1`..`FUZZY9` set explicit edit distance. In-bounds (native).
  - `PhraseMatcher(vocab, *, attr=None)`; `.add(id, [Doc,...])`. `attr="LOWER"` = case-insensitive;
    `attr="LEMMA"` matches on lemmas. Fast for large alias/gazetteer tables.
  - `DependencyMatcher(vocab)` (v3.0+, needs parser). Patterns = list of dicts with
    `LEFT_ID, REL_OP, RIGHT_ID, RIGHT_ATTRS`. Rel ops include `>` (head→dependent),
    `<` (dependent→head), `>>`/`<<` (chains), sibling/precedence ops. Encodes SVO / verb→object.
  - `EntityRuler` pipe: `.add_patterns([{ "label":..., "pattern":... }])`; token or phrase patterns;
    persists to JSONL. Placed before/after NER; `overwrite_ents` controls override.
- (c) Mapping: PhraseMatcher = alias tables built from DB model names/tags. DependencyMatcher =
  head-verb + object-noun extraction ("classify sentiment", "translate to French") → task cue.
  EntityRuler = tag DB-derived domain/specialty terms. Matcher FUZZY = tolerate typos/morph variants
  in cue words without regex.
- (d) In-bounds: YES — all first-party spaCy, no external deps.

### 1b. Linguistic features — VERIFIED (https://spacy.io/usage/linguistic-features)
- (a) Core annotation API surface.
- (b) `token.pos_` (UPOS), `token.tag_` (fine), `token.morph`; `token.dep_`, `token.head`,
  `token.children`/`lefts`/`rights`, `token.subtree`, `ROOT` dep; `doc.noun_chunks` (Spans) with
  `chunk.root` (head noun); `token.lemma_`; `doc.ents` / `ent.label_`; `token.vector`,
  `token.has_vector`, `doc.similarity()`, `token.similarity()`.
- (c) Mapping: `noun_chunks` + `ROOT` verb give the object-noun / head-verb of a prompt — the
  primary syntactic intent signal. Lemmas give normalized cue tokens for overlap scoring. Vectors
  give a semantic fallback for paraphrase.
- (d) In-bounds: YES.
- **CORRECTION (verified separately, see refs):** the auto-summary claimed `md` lacks word vectors.
  That is false for `md`. Only `en_core_web_sm` ships without vectors. **`en_core_web_md` ships
  ~685k keys → 20k unique 300-dim static GloVe-style vectors** (many keys share vectors). `lg` has
  ~500k+ unique. So static-vector cosine IS available under our rules, just coarser than `lg`.
  `md` also has a tok2vec (context tensors) used by the parser/tagger — distinct from the static
  table exposed via `token.vector`.

### 1c. API index — VERIFIED (https://spacy.io/api)
- Classes confirmed to exist as named in the brief: `Matcher`, `PhraseMatcher`, `DependencyMatcher`,
  `EntityRuler`, `TextCategorizer`, `Vectors`, `Tok2Vec`.
- `TextCategorizer` (`textcat` / `textcat_multilabel`) is a trainable in-pipeline classifier; can be
  trained offline on DB task labels and run at inference with zero network. `Vectors` is the static
  table object (`nlp.vocab.vectors`) supporting `most_similar`, key→row lookup. All in-bounds.

---

## 2. HF task taxonomy — VERIFIED (https://huggingface.co/tasks)
- (a) The `pipeline_tag` label space grouped by modality.
- (b)/(c) This is our **target label set** for `task_type`. NLP tasks (most relevant): `feature-extraction,
  fill-mask, question-answering, sentence-similarity, summarization, table-question-answering,
  text-classification, text-generation, text-ranking, token-classification, translation,
  zero-shot-classification`. Other modalities present (must be handled/routed even if out of NLP scope):
  CV (`image-classification, object-detection, image-segmentation, image-to-text, text-to-image, depth-estimation,
  image-to-image, image-to-video, text-to-video, mask-generation, keypoint-detection, zero-shot-image-classification,
  zero-shot-object-detection, unconditional-image-generation, video-classification, video-to-video, text-to-3d, image-to-3d`),
  Audio (`audio-classification, audio-to-audio, automatic-speech-recognition, text-to-speech`),
  Multimodal (`any-to-any, audio-text-to-text, document-question-answering, visual-document-retrieval,
  image-text-to-text, image-text-to-image, image-text-to-video, video-text-to-text, visual-question-answering`),
  Tabular (`tabular-classification, tabular-regression`), and `reinforcement-learning`.
- Implication: the label space is a **fixed, closed vocabulary of dashed strings**. This is exactly what a
  PhraseMatcher alias table / TextCategorizer label set should be built against. The hyphenated labels
  are compositional (verb-to-noun, modality-to-modality) — the "to"/verb structure is itself a syntactic cue.
- (d) In-bounds as a *label source* (HF text is allowed): YES.

---

## 3. Papers

### 3.1 arxiv 2602.12005 — VERIFIED
- (a) "LaCy: What Small Language Models Can and Should Learn is Not Just a Question of Loss"
  (Ujváry, Béthune, Ablin, Monteiro, Cuturi, Kirchhof). NOTE: title/topic differ from the brief's claim
  ("spaCy grammar clarity-OOS gate"). Actual topic: SLM pretraining delegation via a `<CALL>` token.
- (b) Technique of interest to us: combines loss signal with **grammatical info from a spaCy parser** to
  decide when a prediction is "factually or semantically invalid" and should be delegated. The reusable
  idea = use spaCy grammatical structure as a *confidence / validity gate* on a primary decision.
- (c) Mapping: analog of a **clarity/OOS gate** — use parse-derived features (no ROOT verb, no object
  noun_chunk, fragment, imperative-less) to flag low-confidence/underspecified prompts for clarification
  or OOS routing, instead of forcing a task label.
- (d) In-bounds: the *spaCy-grammar-as-gate* pattern is in-bounds. The paper's own SLM/`<CALL>` machinery
  (LLM delegation) is NOT — out of scope for us. Borrow the pattern, not the pipeline.

### 3.2 arxiv 2605.24518 — VERIFIED
- (a) "Grammatically-Guided Sparse Attention for Efficient and Interpretable Transformers" (Spandan Pratyush).
  Matches brief's "soft-POS weighting" claim.
- (b) Technique: constrain/bias attention by **POS-tag grammatical roles**; hard mask (strict) vs soft mask
  (bias toward grammatical interactions). Reusable idea = **weight tokens by POS role** rather than treat all
  tokens equally.
- (c) Mapping: when scoring lexical/vector overlap between a prompt and DB/HF cues, **up-weight content POS**
  (VERB, NOUN, PROPN) and down-weight function words — a soft-POS weighting on our overlap/centroid signals.
- (d) In-bounds: YES as a *weighting scheme* over spaCy POS tags. The transformer attention machinery itself
  is irrelevant; only the POS-weighting heuristic transfers.

### 3.3 arxiv 2601.00506 — VERIFIED
- (a) "Rule-Based Approaches to Atomic Sentence Extraction" (Kamana, Subramanian, Ghosh, Saha). Matches
  brief's "rule-based clause splitting" claim.
- (b) Technique: **spaCy dependency rules** split complex sentences (relative/adverbial clauses, coordination,
  passive) into atomic single-idea sentences. Reported ROUGE-1 F1 0.6714 / ROUGE-L 0.650 on WikiSplit gold.
  Caveat noted by authors: sensitive to syntactic complexity.
- (c) Mapping: pre-split multi-intent prompts ("summarize this and translate it to Spanish") into atomic
  clauses via DependencyMatcher/`token.subtree`, then run task detection per clause → supports multi-label /
  multi-task prompts and cleaner single-verb extraction.
- (d) In-bounds: YES — pure spaCy dependency rules.

### 3.4 arxiv 2404.07220 — VERIFIED
- (a) "Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers"
  (Sawarkar, Mangal, Solanki).
- (b) Technique: blend **dense vector indexes + sparse encoder indexes** with hybrid query strategies; beats
  single-retriever on NQ/TREC-COVID/SQuAD.
- (c) Mapping: our intent→DB-model matching stage should be **hybrid**: sparse lexical (lemma/alias overlap)
  + dense (static-vector centroid cosine), not either alone.
- (d) In-bounds: the *hybrid concept* is in-bounds if implemented with spaCy static vectors + builtin
  lexical scoring. Their specific encoders (Elastic sparse encoder, transformer dense) are NOT usable —
  reimplement the fusion idea with in-bounds parts.

### 3.5 arxiv 2507.22289 — VERIFIED
- (a) "Intent Recognition and Out-of-Scope Detection using LLMs in Multi-party Conversations"
  (Castillo-López, de Chalendar, Semmar).
- (b) Technique: hybrid BERT + LLM (zero/few-shot) for intent recognition **and explicit OOS detection**;
  passing BERT outputs into the LLM improves results.
- (c) Mapping: reinforces that **OOS/clarity detection is a first-class output**, not an afterthought — every
  prompt either maps to an in-vocab task or is flagged OOS/ambiguous. Validates a dedicated gate stage.
- (d) In-bounds: the *architecture* (BERT+LLM) is out-of-bounds. Only the **problem formulation** (intent +
  OOS as joint outputs, with a confidence threshold) transfers.

---

## 4. Hybrid lexical+semantic retrieval — VERIFIED
(https://machinelearningmastery.com/implementing-hybrid-semantic-lexical-search-in-rag/)
- (a) Tutorial on combining lexical + semantic search for RAG.
- (b) Lexical = BM25 (rank_bm25, tokenized docs); semantic = dense sentence-transformer embeddings + cosine;
  fusion = **Reciprocal Rank Fusion (RRF)**: `RRF_score(doc) = Σ 1/(k + rank_i)` over each ranker, `k≈60`,
  sort desc. RRF avoids scale-mismatch of raw-score blending and needs no tuned weight.
- (c) Mapping: rank DB models against a prompt with (i) lexical ranker = lemma/alias overlap (builtin), (ii)
  semantic ranker = static-vector centroid cosine, then **fuse by RRF** for the final "best DB model" pick.
- (d) In-bounds: RRF is pure arithmetic (in-bounds). BM25 via `rank_bm25` and sentence-transformers are
  external libs → NOT allowed. Substitute: builtin IDF/overlap ranker + spaCy-vector cosine ranker, fused by RRF.

---

## Candidate signal families to test (deduped, ranked)

Ranking heuristic: expected signal strength × in-bounds simplicity × cue-provenance cleanliness.
All cue provenance is DB and/or HF text ONLY — never config/prompts.

1. **Head-verb + object-noun extraction** (primary intent core)
   - Signal: prompt ROOT verb + governing `noun_chunks` head → maps to task label (e.g. translate→translation,
     summarize→summarization, classify→text-classification).
   - spaCy API: `doc.noun_chunks`, `chunk.root`, `token.dep_=='ROOT'`, `token.head`, `token.lemma_`.
   - Cue provenance: verb/noun cue lexicon derived from HF task-label strings + DB pipeline_tag/task fields.
   - Risk: imperative-less or nominal prompts ("a French translation of…") have weak/no ROOT verb.

2. **DependencyMatcher SVO / verb→object patterns**
   - Signal: structured (verb, dobj/pobj) tuples give higher-precision task cues than bag-of-lemmas.
   - spaCy API: `DependencyMatcher` with `>` (head→child) rel ops on `dobj`/`pobj`.
   - Cue provenance: patterns seeded from HF labels + DB task text.
   - Risk: pattern authoring is brittle to phrasing; recall gap without many patterns.

3. **PhraseMatcher alias table from DB/HF text**
   - Signal: direct surface/lemma match of prompt spans against model names, tags, task labels.
   - spaCy API: `PhraseMatcher(attr="LEMMA")` (or LOWER); tables built from DB model cards + HF label vocab.
   - Cue provenance: DB text + HF label set — clean.
   - Risk: label strings are hyphenated/multiword; needs tokenization normalization; common-noun cues collide
     (note: repo history shows an NER variant rejected because cues were common nouns — same failure mode).

4. **Hybrid lexical+semantic ranking with RRF** (intent→model matching stage)
   - Signal: fuse (a) IDF-weighted lemma overlap ranker and (b) static-vector centroid cosine ranker via
     `RRF = Σ 1/(k+rank)`, k≈60.
   - spaCy API: `token.lemma_`, `token.vector`/`Doc.vector`, cosine (builtin/numpy); RRF in pure Python.
   - Cue provenance: DB model text (names, tags, descriptions) as the corpus; HF labels as anchors.
   - Risk: `md` has only 20k unique vectors → coarse semantics, many OOV rows share vectors; scale/quality
     below `lg`.

5. **Static-vector centroid cosine** (standalone semantic fallback)
   - Signal: cosine(prompt content-token centroid, per-task/per-model centroid built from DB/HF text).
   - spaCy API: `token.vector`, `token.has_vector`, manual centroid; skip no-vector tokens.
   - Cue provenance: DB/HF text centroids.
   - Risk: centroid averaging dilutes; `md` coverage limits; needs POS filtering (see #7) to avoid function-word noise.

6. **IDF lemma-overlap score** (sparse lexical, standalone)
   - Signal: IDF-weighted overlap of prompt lemmas vs each model/task's lemma bag.
   - spaCy API: `token.lemma_`; IDF computed offline over DB corpus (pure Python).
   - Cue provenance: DB corpus lemma frequencies.
   - Risk: pure lexical → misses paraphrase; must pair with #5 (that is #4).

7. **Soft-POS content weighting** (a multiplier on #4/#5/#6, from arxiv 2605.24518)
   - Signal: weight VERB/NOUN/PROPN high, function words ~0 in overlap/centroid computations.
   - spaCy API: `token.pos_` gate on all lexical/vector aggregations.
   - Cue provenance: none needed (structural).
   - Risk: over-suppressing prepositions can drop "to"/"from" cues that disambiguate translation/direction.

8. **TextCategorizer trained on DB task labels**
   - Signal: in-pipeline `textcat_multilabel` predicting HF task_type from prompt text.
   - spaCy API: `TextCategorizer`; train offline on (DB/HF-derived text → task label) pairs; infer with no network.
   - Cue provenance: DB task-labeled text + HF label set as the training target.
   - Risk: needs a labeled training set; overfits to DB phrasing; repo history shows a task-classifier ensemble
     already REJECTED (large mem gap, hurt home metric) — re-test only with cleaner labels/features.

9. **Clarity / OOS gate** (from arxiv 2602.12005 + 2507.22289)
   - Signal: parse-derived confidence — flag prompts with no ROOT verb, no object noun_chunk, fragment, or
     low top-1 vs top-2 task margin → ask-to-clarify / route OOS instead of forcing a label.
   - spaCy API: `token.dep_`, `doc.noun_chunks`, sentence completeness heuristics; threshold on #1/#4 scores.
   - Cue provenance: structural + score margins (no external cues).
   - Risk: threshold tuning; false-OOS on terse-but-valid prompts.

10. **Atomic clause splitting for multi-intent prompts** (from arxiv 2601.00506)
    - Signal: split conjoined/subordinated prompts into atomic clauses, run #1–#4 per clause → multi-task output.
    - spaCy API: `DependencyMatcher` / `token.subtree` / conj+advcl traversal.
    - Cue provenance: structural.
    - Risk: paper shows sensitivity to syntactic complexity (ROUGE-L ~0.65); adds latency; only worth it if
      multi-intent prompts are common in the eval set.

11. **Matcher FUZZY cue matching** (typo/morph tolerance layer)
    - Signal: fuzzy-match cue words ("summarise"/"summarize", typos) against DB/HF cue lexicon.
    - spaCy API: `Matcher` with `{"LOWER": {"FUZZY": ...}}` / `FUZZY1..9`.
    - Cue provenance: DB/HF cue lexicon.
    - Risk: repo history shows a fuzzy variant REJECTED — fired on ~96% noise, hurt metric, ~480x slower.
      Only reintroduce as a *narrow, low-recall* fallback with tight edit distance, if at all.

---

### Cross-cutting notes
- Every family's cues are sourced from DB and/or HF text only. config.py / prompts.py were NOT read and are
  not referenced (per hard constraints).
- The one factual conflict encountered was the auto-summary claim that `md` lacks word vectors; corrected in §1b
  and verified via spaCy model docs / HF model card.
