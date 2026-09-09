"""Data-layer package for conversation processing, splitting, and datasets."""

from src.data.splitter import (
    ConversationSplitter,
    SplitValidationError,
    load_all_conversation_ids,
    load_golden_conversation_ids,
    load_split_manifest,
    save_split_manifest,
    split_conversations,
    stream_split_conversations,
    validate_splits,
)

__all__ = [
    "ConversationSplitter",
    "SplitValidationError",
    "load_all_conversation_ids",
    "load_golden_conversation_ids",
    "load_split_manifest",
    "save_split_manifest",
    "split_conversations",
    "stream_split_conversations",
    "validate_splits",
]
