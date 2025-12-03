"""Tests for palet8_agents.models.context module."""

import pytest
from palet8_agents.models.context import ContextCompleteness


class TestContextCompleteness:
    """Tests for ContextCompleteness dataclass."""

    def test_init(self):
        """Test initialization."""
        completeness = ContextCompleteness(
            score=0.75,
            is_sufficient=True,
            missing_fields=["background"],
            clarifying_questions=["What background would you prefer?"],
        )
        assert completeness.score == 0.75
        assert completeness.is_sufficient is True
        assert completeness.missing_fields == ["background"]
        assert completeness.clarifying_questions == ["What background would you prefer?"]
        assert completeness.metadata == {}

    def test_init_with_metadata(self):
        """Test initialization with metadata."""
        completeness = ContextCompleteness(
            score=0.5,
            is_sufficient=False,
            missing_fields=["subject", "style"],
            clarifying_questions=["What is the main subject?", "What style?"],
            metadata={"required_missing": ["subject"], "important_missing": ["style"]},
        )
        assert completeness.metadata["required_missing"] == ["subject"]
        assert completeness.metadata["important_missing"] == ["style"]

    def test_init_sufficient(self):
        """Test initialization when context is sufficient."""
        completeness = ContextCompleteness(
            score=0.9,
            is_sufficient=True,
            missing_fields=[],
            clarifying_questions=[],
        )
        assert completeness.is_sufficient is True
        assert completeness.missing_fields == []

    def test_init_insufficient(self):
        """Test initialization when context is insufficient."""
        completeness = ContextCompleteness(
            score=0.3,
            is_sufficient=False,
            missing_fields=["subject", "style", "colors"],
            clarifying_questions=[
                "What would you like the image to show?",
                "What style are you looking for?",
                "Do you have color preferences?",
            ],
        )
        assert completeness.is_sufficient is False
        assert len(completeness.missing_fields) == 3
        assert len(completeness.clarifying_questions) == 3

    def test_to_dict(self):
        """Test to_dict serialization."""
        completeness = ContextCompleteness(
            score=0.65,
            is_sufficient=True,
            missing_fields=["mood"],
            clarifying_questions=["What mood should the image convey?"],
            metadata={"check": "passed"},
        )
        data = completeness.to_dict()

        assert data["score"] == 0.65
        assert data["is_sufficient"] is True
        assert data["missing_fields"] == ["mood"]
        assert data["clarifying_questions"] == ["What mood should the image convey?"]
        assert data["metadata"] == {"check": "passed"}

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "score": 0.8,
            "is_sufficient": True,
            "missing_fields": [],
            "clarifying_questions": [],
            "metadata": {"source": "rag"},
        }
        completeness = ContextCompleteness.from_dict(data)

        assert completeness.score == 0.8
        assert completeness.is_sufficient is True
        assert completeness.missing_fields == []
        assert completeness.metadata["source"] == "rag"

    def test_from_dict_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {
            "score": 0.4,
            "is_sufficient": False,
            "missing_fields": ["subject"],
            "clarifying_questions": ["What subject?"],
        }
        completeness = ContextCompleteness.from_dict(data)

        assert completeness.score == 0.4
        assert completeness.metadata == {}

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = ContextCompleteness(
            score=0.72,
            is_sufficient=True,
            missing_fields=["composition"],
            clarifying_questions=["How should elements be arranged?"],
            metadata={"rag_items": 3},
        )

        data = original.to_dict()
        restored = ContextCompleteness.from_dict(data)

        assert restored.score == original.score
        assert restored.is_sufficient == original.is_sufficient
        assert restored.missing_fields == original.missing_fields
        assert restored.clarifying_questions == original.clarifying_questions
        assert restored.metadata == original.metadata

    def test_score_bounds(self):
        """Test score at boundaries."""
        # Minimum score
        min_completeness = ContextCompleteness(
            score=0.0,
            is_sufficient=False,
            missing_fields=["subject"],
            clarifying_questions=["What?"],
        )
        assert min_completeness.score == 0.0

        # Maximum score
        max_completeness = ContextCompleteness(
            score=1.0,
            is_sufficient=True,
            missing_fields=[],
            clarifying_questions=[],
        )
        assert max_completeness.score == 1.0
