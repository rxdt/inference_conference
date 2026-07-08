"""Coarse routing buckets for the 56 canonical task types.

These 10 buckets are a hierarchical routing layer (predict bucket from the
prompt, then disambiguate the task WITHIN the bucket). They are a semantic
grouping of the task LABEL SPACE (prior knowledge) — NOT the config-derived
`capability_families` (which is a deterministic task->family lookup carrying no
independent signal) and NOT fit to any prompt.

Source: product UI bucket list:
  Text generation / chat | Text embeddings / reranking / classification |
  Encoder-decoder generation | Vision understanding |
  Vision-language / multimodal | Image generation / diffusion |
  Video generation | Speech / audio | Tabular / classical ML | Custom / unknown

The only scoreboard remains task_type[0]; buckets are an internal device for
candidate-narrowing, never an end in themselves.
"""
from __future__ import annotations

import config as cfg

BUCKETS: dict[str, set[str]] = {
    "text-generation-chat": {
        "text-generation", "question-answering", "multiple-choice",
        "table-question-answering",
    },
    "text-embedding-rerank-classify": {
        "feature-extraction", "sentence-similarity", "text-ranking",
        "text-retrieval", "text-classification", "token-classification",
        "zero-shot-classification", "fill-mask",
    },
    "encoder-decoder-generation": {
        "text2text-generation", "summarization", "translation",
    },
    "vision-understanding": {
        "image-classification", "image-segmentation", "object-detection",
        "depth-estimation", "keypoint-detection", "mask-generation",
        "image-feature-extraction", "zero-shot-image-classification",
        "zero-shot-object-detection", "image-to-text", "video-classification",
    },
    "vision-language-multimodal": {
        "image-text-to-text", "visual-question-answering", "video-text-to-text",
        "audio-text-to-text", "any-to-any", "multimodal-chat-completion",
        "visual-document-retrieval", "document-question-answering",
    },
    "image-generation-diffusion": {
        "text-to-image", "image-to-image", "image-text-to-image",
        "unconditional-image-generation", "image-to-3d", "text-to-3d",
    },
    "video-generation": {
        "text-to-video", "image-to-video", "image-text-to-video", "video-to-video",
    },
    "speech-audio": {
        "automatic-speech-recognition", "text-to-speech", "audio-classification",
        "audio-to-audio", "text-to-audio", "voice-activity-detection",
    },
    "tabular-classical-ml": {
        "tabular-classification", "tabular-regression",
        "time-series-forecasting", "graph-ml",
    },
    "custom-unknown": {
        "reinforcement-learning", "robotics",
    },
}

# task -> bucket reverse index
TASK_TO_BUCKET: dict[str, str] = {
    task: bucket for bucket, tasks in BUCKETS.items() for task in tasks
}

# Ambiguous assignments worth revisiting if they cause errors:
#   image-to-text       -> vision-understanding (captioning/OCR; could be VL)
#   document-question-answering -> vision-language (doc images + Q; could be text)
#   video-classification-> vision-understanding (video analysis, not generation)
#   image-to-3d/text-to-3d -> image-generation (asset synthesis)

_CANON = set(cfg.TASK_TAXONOMY["canonical"])
assert set(TASK_TO_BUCKET) == _CANON, (
    f"bucket coverage mismatch; missing={_CANON - set(TASK_TO_BUCKET)}, "
    f"extra={set(TASK_TO_BUCKET) - _CANON}"
)
