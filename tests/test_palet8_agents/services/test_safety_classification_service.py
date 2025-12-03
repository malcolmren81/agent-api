"""Tests for palet8_agents.services.safety_classification_service module."""

import pytest
from unittest.mock import AsyncMock, patch
from palet8_agents.services.safety_classification_service import (
    SafetyClassificationService,
    SafetyConfig,
)
from palet8_agents.models import (
    SafetyCategory,
    SafetySeverity,
    SafetyFlag,
    SafetyClassification,
    SafetyResult,
)


class TestSafetyConfig:
    """Tests for SafetyConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = SafetyConfig()
        assert config.blocking_behavior["nsfw"] == "block"
        assert config.blocking_behavior["ip_trademark"] == "tag"
        assert "nude" in config.nsfw_keywords
        assert "violence" in config.extreme_block_keywords

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SafetyConfig(
            nsfw_keywords=["custom_keyword"],
            tagging_thresholds={"nsfw": 0.3},
        )
        assert "custom_keyword" in config.nsfw_keywords
        assert config.tagging_thresholds["nsfw"] == 0.3


class TestSafetyClassificationService:
    """Tests for SafetyClassificationService."""

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        service = SafetyClassificationService()
        assert service._text_service is None
        assert service._owns_service is True
        assert service._config is not None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = SafetyConfig(nsfw_keywords=["test"])
        service = SafetyClassificationService(config=config)
        assert "test" in service._config.nsfw_keywords


class TestKeywordChecks:
    """Tests for keyword-based safety checks."""

    def test_nsfw_keyword_detection(self):
        """Test NSFW keyword detection."""
        service = SafetyClassificationService()
        flag = service._quick_keyword_check("I want nude content", "input")

        assert flag is not None
        assert flag.category == SafetyCategory.NSFW
        assert flag.severity == SafetySeverity.CRITICAL
        assert flag.score == 1.0

    def test_extreme_violence_keyword(self):
        """Test extreme violence keyword detection."""
        service = SafetyClassificationService()
        flag = service._quick_keyword_check("image of terror attack", "input")

        assert flag is not None
        assert flag.severity == SafetySeverity.CRITICAL

    def test_clean_content(self):
        """Test that clean content passes keyword check."""
        service = SafetyClassificationService()
        flag = service._quick_keyword_check("A beautiful sunset over mountains", "input")

        assert flag is None

    def test_case_insensitive(self):
        """Test that keyword check is case insensitive."""
        service = SafetyClassificationService()
        flag = service._quick_keyword_check("NUDE content", "input")

        assert flag is not None
        assert flag.category == SafetyCategory.NSFW


class TestIPTrademark:
    """Tests for IP/trademark detection."""

    def test_ip_detection(self):
        """Test IP detection."""
        config = SafetyConfig(ip_blocklist={"disney", "mickey mouse"})
        service = SafetyClassificationService(config=config)
        flag = service._check_ip_trademark("Draw Mickey Mouse", "input")

        assert flag is not None
        assert flag.category == SafetyCategory.IP_TRADEMARK
        assert flag.severity == SafetySeverity.MEDIUM  # IP never blocks
        assert flag.metadata["action"] == "tag"

    def test_ip_never_critical(self):
        """Test that IP violations are never critical."""
        config = SafetyConfig(ip_blocklist={"nike", "apple"})
        service = SafetyClassificationService(config=config)
        flag = service._check_ip_trademark("Apple logo design", "input")

        assert flag is not None
        assert flag.severity != SafetySeverity.CRITICAL

    def test_no_ip_detected(self):
        """Test when no IP is detected."""
        service = SafetyClassificationService()
        flag = service._check_ip_trademark("A generic cat illustration", "input")

        assert flag is None


class TestSeverityConversion:
    """Tests for severity level conversion."""

    def test_score_to_severity_critical(self):
        """Test critical severity."""
        service = SafetyClassificationService()
        severity = service._score_to_severity(0.95)
        assert severity == SafetySeverity.CRITICAL

    def test_score_to_severity_high(self):
        """Test high severity."""
        service = SafetyClassificationService()
        severity = service._score_to_severity(0.75)
        assert severity == SafetySeverity.HIGH

    def test_score_to_severity_medium(self):
        """Test medium severity."""
        service = SafetyClassificationService()
        severity = service._score_to_severity(0.6)
        assert severity == SafetySeverity.MEDIUM

    def test_score_to_severity_low(self):
        """Test low severity."""
        service = SafetyClassificationService()
        severity = service._score_to_severity(0.35)
        assert severity == SafetySeverity.LOW

    def test_score_to_severity_none(self):
        """Test none severity."""
        service = SafetyClassificationService()
        severity = service._score_to_severity(0.1)
        assert severity == SafetySeverity.NONE


class TestSeverityPenalty:
    """Tests for severity penalties."""

    def test_get_severity_penalty(self):
        """Test getting severity penalty."""
        service = SafetyClassificationService()

        assert service.get_severity_penalty(SafetySeverity.NONE) == 0.0
        assert service.get_severity_penalty(SafetySeverity.LOW) == 0.05
        assert service.get_severity_penalty(SafetySeverity.MEDIUM) == 0.15
        assert service.get_severity_penalty(SafetySeverity.HIGH) == 0.30
        assert service.get_severity_penalty(SafetySeverity.CRITICAL) == 0.50


class TestSafetyClassificationCreation:
    """Tests for SafetyClassification creation."""

    def test_create_classification_empty(self):
        """Test creating classification with no flags."""
        service = SafetyClassificationService()
        classification = service.create_safety_classification([])

        assert classification.is_safe is True
        assert classification.requires_review is False
        assert classification.risk_level == "low"
        assert classification.categories == []

    def test_create_classification_with_flags(self):
        """Test creating classification with flags."""
        service = SafetyClassificationService()
        flags = [
            SafetyFlag(
                category=SafetyCategory.VIOLENCE,
                severity=SafetySeverity.HIGH,
                score=0.8,
                description="Violence detected",
                source="input",
            ),
        ]
        classification = service.create_safety_classification(flags)

        assert classification.requires_review is True
        # Risk level could be "high" depending on severity
        assert classification.risk_level in ["high", "medium"]
        assert "violence" in classification.categories

    def test_create_classification_critical_unsafe(self):
        """Test that critical flags make classification unsafe."""
        service = SafetyClassificationService()
        flags = [
            SafetyFlag(
                category=SafetyCategory.NSFW,
                severity=SafetySeverity.CRITICAL,
                score=1.0,
                description="NSFW detected",
                source="input",
            ),
        ]
        classification = service.create_safety_classification(flags)

        assert classification.is_safe is False
        assert classification.requires_review is True


class TestSafetyResultCreation:
    """Tests for SafetyResult creation."""

    def test_create_result_safe(self):
        """Test creating safe result."""
        service = SafetyClassificationService()
        result = service.create_safety_result("job-123", [])

        assert result.is_safe is True
        assert result.overall_score == 1.0
        assert result.blocked_categories == []
        assert result.user_message is None

    def test_create_result_with_flags(self):
        """Test creating result with flags."""
        service = SafetyClassificationService()
        flags = [
            SafetyFlag(
                category=SafetyCategory.VIOLENCE,
                severity=SafetySeverity.MEDIUM,
                score=0.6,
                description="Mild violence",
                source="input",
            ),
        ]
        result = service.create_safety_result("job-123", flags)

        # Medium violence should be tagged but not blocked (behavior: tag)
        assert result.is_safe is True
        assert result.overall_score < 1.0
        assert "violence" not in result.blocked_categories

    def test_create_result_blocked(self):
        """Test creating blocked result."""
        service = SafetyClassificationService()
        flags = [
            SafetyFlag(
                category=SafetyCategory.NSFW,
                severity=SafetySeverity.CRITICAL,
                score=1.0,
                description="NSFW content",
                source="input",
            ),
        ]
        result = service.create_safety_result("job-123", flags)

        assert result.is_safe is False
        assert "nsfw" in result.blocked_categories
        assert result.user_message is not None
        assert len(result.alternatives) > 0


class TestAsyncOperations:
    """Async tests for SafetyClassificationService."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing service releases resources."""
        service = SafetyClassificationService()
        mock_service = AsyncMock()
        service._text_service = mock_service
        service._owns_service = True

        await service.close()

        mock_service.close.assert_called_once()
        assert service._text_service is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with SafetyClassificationService() as service:
            assert service is not None

    @pytest.mark.asyncio
    async def test_classify_content_nsfw(self):
        """Test classifying NSFW content."""
        service = SafetyClassificationService()
        flag = await service.classify_content("nude image", source="input", use_llm=False)

        assert flag is not None
        assert flag.category == SafetyCategory.NSFW
        assert flag.severity == SafetySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_classify_content_clean(self):
        """Test classifying clean content."""
        service = SafetyClassificationService()
        flag = await service.classify_content(
            "A beautiful mountain landscape at sunset",
            source="input",
            use_llm=False,  # Skip LLM for test
        )

        assert flag is None

    @pytest.mark.asyncio
    async def test_classify_content_empty(self):
        """Test classifying empty content."""
        service = SafetyClassificationService()
        flag = await service.classify_content("", source="input")

        assert flag is None

    @pytest.mark.asyncio
    async def test_classify_content_short(self):
        """Test classifying very short content."""
        service = SafetyClassificationService()
        flag = await service.classify_content("ab", source="input")

        assert flag is None
