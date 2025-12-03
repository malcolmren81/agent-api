"""Tests for palet8_agents.models.safety module."""

import pytest
from palet8_agents.models.safety import (
    SafetyCategory,
    SafetySeverity,
    SafetyClassification,
    SafetyFlag,
    SafetyResult,
)


class TestSafetyCategory:
    """Tests for SafetyCategory enum."""

    def test_values(self):
        """Test enum values exist."""
        assert SafetyCategory.NSFW.value == "nsfw"
        assert SafetyCategory.VIOLENCE.value == "violence"
        assert SafetyCategory.HATE.value == "hate"
        assert SafetyCategory.IP_TRADEMARK.value == "ip_trademark"
        assert SafetyCategory.ILLEGAL.value == "illegal"

    def test_from_string(self):
        """Test creating from string value."""
        assert SafetyCategory("nsfw") == SafetyCategory.NSFW
        assert SafetyCategory("violence") == SafetyCategory.VIOLENCE


class TestSafetySeverity:
    """Tests for SafetySeverity enum."""

    def test_values(self):
        """Test enum values exist."""
        assert SafetySeverity.NONE.value == "none"
        assert SafetySeverity.LOW.value == "low"
        assert SafetySeverity.MEDIUM.value == "medium"
        assert SafetySeverity.HIGH.value == "high"
        assert SafetySeverity.CRITICAL.value == "critical"


class TestSafetyClassification:
    """Tests for SafetyClassification dataclass."""

    def test_init(self):
        """Test initialization."""
        classification = SafetyClassification(
            is_safe=True,
            requires_review=False,
            risk_level="low",
            categories=["nsfw"],
        )
        assert classification.is_safe is True
        assert classification.requires_review is False
        assert classification.risk_level == "low"
        assert classification.categories == ["nsfw"]
        assert classification.flags == {}
        assert classification.reason == ""

    def test_init_with_optional(self):
        """Test initialization with optional fields."""
        classification = SafetyClassification(
            is_safe=False,
            requires_review=True,
            risk_level="high",
            categories=["violence", "hate"],
            flags={"nsfw_check": True},
            reason="Contains violent content",
        )
        assert classification.is_safe is False
        assert classification.flags == {"nsfw_check": True}
        assert classification.reason == "Contains violent content"

    def test_to_dict(self):
        """Test to_dict serialization."""
        classification = SafetyClassification(
            is_safe=True,
            requires_review=False,
            risk_level="low",
            categories=[],
        )
        data = classification.to_dict()

        assert data["is_safe"] is True
        assert data["requires_review"] is False
        assert data["risk_level"] == "low"
        assert data["categories"] == []

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "is_safe": False,
            "requires_review": True,
            "risk_level": "medium",
            "categories": ["nsfw"],
            "flags": {"check": True},
            "reason": "Test reason",
        }
        classification = SafetyClassification.from_dict(data)

        assert classification.is_safe is False
        assert classification.requires_review is True
        assert classification.risk_level == "medium"
        assert classification.categories == ["nsfw"]


class TestSafetyFlag:
    """Tests for SafetyFlag dataclass."""

    def test_init(self):
        """Test initialization."""
        flag = SafetyFlag(
            category=SafetyCategory.NSFW,
            severity=SafetySeverity.HIGH,
            score=0.85,
            description="Explicit content detected",
            source="prompt",
        )
        assert flag.category == SafetyCategory.NSFW
        assert flag.severity == SafetySeverity.HIGH
        assert flag.score == 0.85
        assert flag.description == "Explicit content detected"
        assert flag.source == "prompt"
        assert flag.metadata == {}

    def test_to_dict(self):
        """Test to_dict serialization."""
        flag = SafetyFlag(
            category=SafetyCategory.VIOLENCE,
            severity=SafetySeverity.MEDIUM,
            score=0.6,
            description="Violence reference",
            source="input",
            metadata={"keyword": "blood"},
        )
        data = flag.to_dict()

        assert data["category"] == "violence"
        assert data["severity"] == "medium"
        assert data["score"] == 0.6
        assert data["metadata"] == {"keyword": "blood"}

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "category": "hate",
            "severity": "low",
            "score": 0.3,
            "description": "Minor issue",
            "source": "image",
            "metadata": {},
        }
        flag = SafetyFlag.from_dict(data)

        assert flag.category == SafetyCategory.HATE
        assert flag.severity == SafetySeverity.LOW
        assert flag.score == 0.3


class TestSafetyResult:
    """Tests for SafetyResult dataclass."""

    def test_init_minimal(self):
        """Test minimal initialization."""
        result = SafetyResult(
            job_id="job-123",
            is_safe=True,
            overall_score=0.95,
        )
        assert result.job_id == "job-123"
        assert result.is_safe is True
        assert result.overall_score == 0.95
        assert result.flags == []
        assert result.blocked_categories == []
        assert result.user_message is None
        assert result.alternatives == []

    def test_init_with_flags(self):
        """Test initialization with safety flags."""
        flag = SafetyFlag(
            category=SafetyCategory.IP_TRADEMARK,
            severity=SafetySeverity.MEDIUM,
            score=0.5,
            description="Brand detected",
            source="prompt",
        )
        result = SafetyResult(
            job_id="job-456",
            is_safe=True,
            overall_score=0.8,
            flags=[flag],
            user_message="Content flagged for review",
            alternatives=["Use generic brand instead"],
        )
        assert len(result.flags) == 1
        assert result.flags[0].category == SafetyCategory.IP_TRADEMARK
        assert result.user_message == "Content flagged for review"

    def test_to_dict(self):
        """Test to_dict serialization with nested flags."""
        flag = SafetyFlag(
            category=SafetyCategory.NSFW,
            severity=SafetySeverity.CRITICAL,
            score=0.95,
            description="Blocked",
            source="input",
        )
        result = SafetyResult(
            job_id="job-789",
            is_safe=False,
            overall_score=0.1,
            flags=[flag],
            blocked_categories=["nsfw"],
        )
        data = result.to_dict()

        assert data["job_id"] == "job-789"
        assert data["is_safe"] is False
        assert len(data["flags"]) == 1
        assert data["flags"][0]["category"] == "nsfw"
        assert data["blocked_categories"] == ["nsfw"]

    def test_from_dict(self):
        """Test from_dict deserialization with nested flags."""
        data = {
            "job_id": "job-abc",
            "is_safe": True,
            "overall_score": 0.9,
            "flags": [
                {
                    "category": "violence",
                    "severity": "low",
                    "score": 0.2,
                    "description": "Minor",
                    "source": "prompt",
                    "metadata": {},
                }
            ],
            "blocked_categories": [],
            "user_message": None,
            "alternatives": [],
            "metadata": {},
        }
        result = SafetyResult.from_dict(data)

        assert result.job_id == "job-abc"
        assert result.is_safe is True
        assert len(result.flags) == 1
        assert result.flags[0].category == SafetyCategory.VIOLENCE

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        flag = SafetyFlag(
            category=SafetyCategory.HATE,
            severity=SafetySeverity.HIGH,
            score=0.75,
            description="Test flag",
            source="input",
        )
        original = SafetyResult(
            job_id="roundtrip-test",
            is_safe=False,
            overall_score=0.3,
            flags=[flag],
            blocked_categories=["hate"],
            user_message="Blocked",
            alternatives=["Try again"],
        )

        data = original.to_dict()
        restored = SafetyResult.from_dict(data)

        assert restored.job_id == original.job_id
        assert restored.is_safe == original.is_safe
        assert restored.overall_score == original.overall_score
        assert len(restored.flags) == len(original.flags)
        assert restored.flags[0].category == original.flags[0].category
