"""Tests for palet8_agents.models.requirements module."""

import pytest
from palet8_agents.models.requirements import RequirementsStatus


class TestRequirementsStatus:
    """Tests for RequirementsStatus class."""

    def test_init_defaults(self):
        """Test default initialization."""
        status = RequirementsStatus()
        assert status.subject is None
        assert status.style is None
        assert status.colors == []
        assert status.mood is None
        assert status.composition is None
        assert status.include_elements == []
        assert status.avoid_elements == []
        assert status.reference_image is None
        assert status.additional_notes is None

    def test_completeness_score_empty(self):
        """Test completeness score with no fields set."""
        status = RequirementsStatus()
        assert status.completeness_score == 0.0

    def test_completeness_score_subject_only(self):
        """Test completeness score with only subject set."""
        status = RequirementsStatus()
        status.subject = "A cute cat"
        assert status.completeness_score == 0.5

    def test_completeness_score_full(self):
        """Test completeness score with all weighted fields set."""
        status = RequirementsStatus()
        status.subject = "A cute cat"
        status.style = "cartoon"
        status.colors = ["blue", "white"]
        status.mood = "playful"
        assert status.completeness_score == 1.0

    def test_completeness_score_capped(self):
        """Test completeness score is capped at 1.0."""
        status = RequirementsStatus()
        status.subject = "A cute cat"
        status.style = "cartoon"
        status.colors = ["blue", "white"]
        status.mood = "playful"
        # Score should not exceed 1.0 even with all fields
        assert status.completeness_score <= 1.0

    def test_is_complete_false_when_no_subject(self):
        """Test is_complete is False when subject is missing."""
        status = RequirementsStatus()
        status.style = "cartoon"
        status.colors = ["blue"]
        assert status.is_complete is False

    def test_is_complete_true_with_subject(self):
        """Test is_complete is True when subject is set."""
        status = RequirementsStatus()
        status.subject = "A cute cat"
        assert status.is_complete is True

    def test_missing_fields(self):
        """Test missing_fields property."""
        status = RequirementsStatus()
        missing = status.missing_fields
        assert "subject" in missing
        assert "style" in missing
        assert "colors" in missing

    def test_missing_fields_with_some_set(self):
        """Test missing_fields with some fields populated."""
        status = RequirementsStatus()
        status.subject = "A cat"
        status.style = "cartoon"
        missing = status.missing_fields
        assert "subject" not in missing
        assert "style" not in missing
        assert "colors" in missing

    def test_to_dict(self):
        """Test to_dict serialization."""
        status = RequirementsStatus()
        status.subject = "A cat"
        status.style = "cartoon"
        status.colors = ["blue", "white"]

        data = status.to_dict()

        assert data["subject"] == "A cat"
        assert data["style"] == "cartoon"
        assert data["colors"] == ["blue", "white"]
        assert data["mood"] is None
        assert "completeness_score" in data
        assert "is_complete" in data
        assert data["is_complete"] is True

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "subject": "A dog",
            "style": "realistic",
            "colors": ["brown", "black"],
            "mood": "happy",
            "composition": "centered",
            "include_elements": ["ball", "grass"],
            "avoid_elements": ["water"],
            "reference_image": "https://example.com/dog.jpg",
            "additional_notes": "Make it cute",
        }

        status = RequirementsStatus.from_dict(data)

        assert status.subject == "A dog"
        assert status.style == "realistic"
        assert status.colors == ["brown", "black"]
        assert status.mood == "happy"
        assert status.composition == "centered"
        assert status.include_elements == ["ball", "grass"]
        assert status.avoid_elements == ["water"]
        assert status.reference_image == "https://example.com/dog.jpg"
        assert status.additional_notes == "Make it cute"

    def test_from_dict_defaults(self):
        """Test from_dict with missing fields uses defaults."""
        data = {"subject": "A cat"}

        status = RequirementsStatus.from_dict(data)

        assert status.subject == "A cat"
        assert status.style is None
        assert status.colors == []
        assert status.include_elements == []

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = RequirementsStatus()
        original.subject = "A landscape"
        original.style = "impressionist"
        original.colors = ["green", "blue"]
        original.mood = "serene"

        data = original.to_dict()
        restored = RequirementsStatus.from_dict(data)

        assert restored.subject == original.subject
        assert restored.style == original.style
        assert restored.colors == original.colors
        assert restored.mood == original.mood
