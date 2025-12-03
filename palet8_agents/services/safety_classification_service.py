"""
Safety Classification Service

Service for content safety classification and IP/trademark checks.
This extracts the safety classification logic from SafetyAgent into a reusable service.

Documentation Reference: Section 5.2.4 (Safety Agent)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging
import yaml

from palet8_agents.models import (
    SafetyCategory,
    SafetySeverity,
    SafetyClassification,
    SafetyFlag,
    SafetyResult,
)
from palet8_agents.services.text_llm_service import TextLLMService, TextLLMServiceError

logger = logging.getLogger(__name__)


class SafetyClassificationError(Exception):
    """Base exception for SafetyClassificationService errors."""
    pass


@dataclass
class SafetyConfig:
    """Configuration for safety classification."""
    blocking_behavior: Dict[str, str] = field(default_factory=lambda: {
        "nsfw": "block",
        "violence": "tag",
        "hate": "tag",
        "ip_trademark": "tag",
        "illegal": "tag",
    })
    tagging_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "nsfw": 0.5,
        "violence": 0.7,
        "hate": 0.7,
        "ip_trademark": 0.5,
        "illegal": 0.8,
    })
    nsfw_keywords: List[str] = field(default_factory=lambda: [
        "nude", "naked", "xxx", "porn", "pornograph",
        "nsfw", "sexual", "erotic", "hentai"
    ])
    extreme_block_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "violence": ["child abuse", "terror attack", "mass shooting", "torture of children"],
        "hate": ["nazi propaganda", "genocide", "ethnic cleansing", "white supremacy"],
        "illegal": ["child exploitation", "human trafficking", "csam"],
    })
    ip_blocklist: Set[str] = field(default_factory=set)
    severity_penalties: Dict[str, float] = field(default_factory=lambda: {
        "none": 0.0,
        "low": 0.05,
        "medium": 0.15,
        "high": 0.30,
        "critical": 0.50,
    })
    score_to_severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "critical": 0.9,
        "high": 0.7,
        "medium": 0.5,
        "low": 0.3,
    })
    user_messages: Dict[str, str] = field(default_factory=dict)
    alternatives: Dict[str, List[str]] = field(default_factory=dict)


class SafetyClassificationService:
    """
    Service for content safety classification and IP checks.

    This service:
    - Performs fast keyword-based safety checks
    - Detects IP/trademark violations from a configurable blocklist
    - Uses LLM for nuanced content classification when needed
    - Calculates severity levels and applies appropriate penalties
    - NEVER blocks for IP - only tags for visibility control
    """

    # Default classification prompt
    DEFAULT_CLASSIFICATION_PROMPT = """You are the Safety monitor for an image generation system.

Classify the following content for safety issues. Rate each category from 0.0 to 1.0:
- nsfw: Adult/sexual content
- violence: Violence and harm
- hate: Hate speech and discrimination
- ip_trademark: Intellectual property (brands, characters, logos)
- illegal: Illegal activities

Return ONLY a JSON object with these scores, like:
{"nsfw": 0.0, "violence": 0.0, "hate": 0.0, "ip_trademark": 0.0, "illegal": 0.0}"""

    def __init__(
        self,
        text_service: Optional[TextLLMService] = None,
        config: Optional[SafetyConfig] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize SafetyClassificationService.

        Args:
            text_service: TextLLMService for LLM calls. Creates one if not provided.
            config: Optional SafetyConfig. Loaded from file if not provided.
            config_path: Path to config YAML file. Uses default if not provided.
        """
        self._text_service = text_service
        self._owns_service = text_service is None
        self._config = config or self._load_config(config_path)

    def _load_config(self, config_path: Optional[Path] = None) -> SafetyConfig:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try default config location
            default_paths = [
                Path("config/safety_config.yaml"),
                Path(__file__).parent.parent.parent / "config" / "safety_config.yaml",
            ]
            for path in default_paths:
                if path.exists():
                    config_path = path
                    break

        if config_path and config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)

                # Build IP blocklist from known entities
                ip_blocklist = set()
                known_entities = data.get("ip_detection", {}).get("known_entities", {})
                for category, entities in known_entities.items():
                    if isinstance(entities, list):
                        ip_blocklist.update(e.lower() for e in entities)

                return SafetyConfig(
                    blocking_behavior=data.get("blocking_behavior", {}),
                    tagging_thresholds=data.get("tagging_thresholds", {}),
                    nsfw_keywords=data.get("nsfw", {}).get("critical_keywords", []),
                    extreme_block_keywords=data.get("extreme_block_keywords", {}),
                    ip_blocklist=ip_blocklist,
                    user_messages=data.get("user_messages", {}),
                    alternatives=data.get("alternatives", {}),
                )
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return SafetyConfig()

    async def _get_text_service(self) -> TextLLMService:
        """Get or create text LLM service."""
        if self._text_service is None:
            self._text_service = TextLLMService(default_profile_name="safety")
        return self._text_service

    async def close(self) -> None:
        """Close the service and release resources."""
        if self._text_service and self._owns_service:
            await self._text_service.close()
            self._text_service = None

    async def classify_content(
        self,
        text: str,
        source: str = "input",
        use_llm: bool = True,
    ) -> Optional[SafetyFlag]:
        """
        Classify content for safety violations.

        Args:
            text: Content to classify
            source: Where the content came from (input, prompt, image)
            use_llm: Whether to use LLM for nuanced classification

        Returns:
            SafetyFlag if violation detected, None if safe
        """
        if not text or len(text.strip()) < 3:
            return None

        # 1. Fast keyword check first (always run)
        keyword_flag = self._quick_keyword_check(text, source)
        if keyword_flag and keyword_flag.severity == SafetySeverity.CRITICAL:
            return keyword_flag

        # 2. IP/trademark check
        ip_flag = self._check_ip_trademark(text, source)
        if ip_flag:
            return ip_flag

        # 3. Return keyword flag if found (non-critical)
        if keyword_flag:
            return keyword_flag

        # 4. LLM-based classification for edge cases
        if use_llm:
            return await self._llm_classify(text, source)

        return None

    def _quick_keyword_check(
        self,
        text: str,
        source: str,
    ) -> Optional[SafetyFlag]:
        """
        Fast keyword-based safety check.

        Args:
            text: Content to check
            source: Content source

        Returns:
            SafetyFlag if violation found
        """
        text_lower = text.lower()

        # Check NSFW keywords (always block)
        for keyword in self._config.nsfw_keywords:
            if keyword.lower() in text_lower:
                return SafetyFlag(
                    category=SafetyCategory.NSFW,
                    severity=SafetySeverity.CRITICAL,
                    score=1.0,
                    description=f"Prohibited adult content detected",
                    source=source,
                    metadata={"matched_keyword": keyword},
                )

        # Check extreme block keywords by category
        for category, keywords in self._config.extreme_block_keywords.items():
            if isinstance(keywords, list):
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        try:
                            cat = SafetyCategory(category)
                        except ValueError:
                            cat = SafetyCategory.ILLEGAL  # fallback
                        return SafetyFlag(
                            category=cat,
                            severity=SafetySeverity.CRITICAL,
                            score=1.0,
                            description=f"Prohibited content: {category}",
                            source=source,
                            metadata={"matched_keyword": keyword},
                        )

        return None

    def _check_ip_trademark(
        self,
        text: str,
        source: str,
    ) -> Optional[SafetyFlag]:
        """
        Check for IP/trademark violations.

        IP detection never blocks - only tags for visibility control.

        Args:
            text: Content to check
            source: Content source

        Returns:
            SafetyFlag if IP detected (MEDIUM severity, never blocks)
        """
        text_lower = text.lower()

        for blocked_term in self._config.ip_blocklist:
            if blocked_term in text_lower:
                return SafetyFlag(
                    category=SafetyCategory.IP_TRADEMARK,
                    severity=SafetySeverity.MEDIUM,
                    score=self._config.tagging_thresholds.get("ip_trademark", 0.5),
                    description=f"Known brand/character detected: {blocked_term}",
                    source=source,
                    metadata={
                        "matched_term": blocked_term,
                        "action": "tag",  # IP never blocks, just tags
                        "visibility_control": "user_private",
                    },
                )

        return None

    async def _llm_classify(
        self,
        text: str,
        source: str,
    ) -> Optional[SafetyFlag]:
        """
        Use LLM for nuanced safety classification.

        Args:
            text: Content to classify
            source: Content source

        Returns:
            SafetyFlag if violation detected
        """
        try:
            text_service = await self._get_text_service()

            result = await text_service.generate_text(
                prompt=f"Classify this content:\n\n\"{text[:1000]}\"",
                system_prompt=self.DEFAULT_CLASSIFICATION_PROMPT,
                temperature=0.0,
                max_tokens=200,
            )

            # Parse response
            scores = self._parse_safety_scores(result.content)

            # Find highest risk category
            max_category = None
            max_score = 0.0
            for category, score in scores.items():
                if score > max_score:
                    max_score = score
                    max_category = category

            if max_score >= 0.5:  # Moderate or higher risk
                severity = self._score_to_severity(max_score)
                try:
                    cat = SafetyCategory(max_category)
                except ValueError:
                    cat = SafetyCategory.ILLEGAL  # fallback

                return SafetyFlag(
                    category=cat,
                    severity=severity,
                    score=max_score,
                    description=f"Potential {max_category} content detected",
                    source=source,
                    metadata={"all_scores": scores},
                )

        except TextLLMServiceError as e:
            logger.warning(f"LLM safety check failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error in LLM classification: {e}")

        return None

    def _parse_safety_scores(self, response: str) -> Dict[str, float]:
        """Parse safety scores from LLM response."""
        import json
        import re

        try:
            # Try to parse JSON from response
            json_match = re.search(r'\{[^{}]+\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        # Default scores if parsing fails
        return {cat.value: 0.0 for cat in SafetyCategory}

    def _score_to_severity(self, score: float) -> SafetySeverity:
        """Convert score to severity level."""
        thresholds = self._config.score_to_severity_thresholds
        if score >= thresholds.get("critical", 0.9):
            return SafetySeverity.CRITICAL
        elif score >= thresholds.get("high", 0.7):
            return SafetySeverity.HIGH
        elif score >= thresholds.get("medium", 0.5):
            return SafetySeverity.MEDIUM
        elif score >= thresholds.get("low", 0.3):
            return SafetySeverity.LOW
        return SafetySeverity.NONE

    def get_severity_penalty(self, severity: SafetySeverity) -> float:
        """
        Get score penalty for a severity level.

        Args:
            severity: Severity level

        Returns:
            Penalty value to subtract from job safety score
        """
        return self._config.severity_penalties.get(severity.value, 0.1)

    def create_safety_classification(
        self,
        flags: List[SafetyFlag],
    ) -> SafetyClassification:
        """
        Create a SafetyClassification from a list of flags.

        Args:
            flags: List of safety flags

        Returns:
            SafetyClassification summary
        """
        if not flags:
            return SafetyClassification(
                is_safe=True,
                requires_review=False,
                risk_level="low",
                categories=[],
            )

        # Severity order for comparison (higher value = more severe)
        severity_order = {
            SafetySeverity.NONE: 0,
            SafetySeverity.LOW: 1,
            SafetySeverity.MEDIUM: 2,
            SafetySeverity.HIGH: 3,
            SafetySeverity.CRITICAL: 4,
        }

        # Determine overall risk level
        max_severity = SafetySeverity.NONE
        categories = []
        requires_review = False
        is_safe = True

        for flag in flags:
            if severity_order[flag.severity] > severity_order[max_severity]:
                max_severity = flag.severity
            if flag.category.value not in categories:
                categories.append(flag.category.value)

            if flag.severity in [SafetySeverity.HIGH, SafetySeverity.CRITICAL]:
                requires_review = True

            if flag.severity == SafetySeverity.CRITICAL:
                is_safe = False

        # Map severity to risk level string
        risk_map = {
            SafetySeverity.NONE: "low",
            SafetySeverity.LOW: "low",
            SafetySeverity.MEDIUM: "medium",
            SafetySeverity.HIGH: "high",
            SafetySeverity.CRITICAL: "high",
        }
        risk_level = risk_map.get(max_severity, "medium")

        # Get reason from highest severity flag
        reason = ""
        for flag in flags:
            if flag.severity == max_severity:
                reason = flag.description
                break

        return SafetyClassification(
            is_safe=is_safe,
            requires_review=requires_review,
            risk_level=risk_level,
            categories=categories,
            flags={f.category.value: True for f in flags},
            reason=reason,
        )

    def create_safety_result(
        self,
        job_id: str,
        flags: List[SafetyFlag],
        initial_score: float = 1.0,
    ) -> SafetyResult:
        """
        Create a SafetyResult from accumulated flags.

        Args:
            job_id: Job identifier
            flags: List of accumulated safety flags
            initial_score: Starting score before penalties

        Returns:
            SafetyResult with verdict
        """
        overall_score = initial_score

        # Calculate overall score
        for flag in flags:
            penalty = self.get_severity_penalty(flag.severity)
            overall_score = max(0.0, overall_score - penalty)

        # Determine blocked categories
        blocked_categories = []
        for flag in flags:
            threshold = self._config.tagging_thresholds.get(flag.category.value, 0.8)
            behavior = self._config.blocking_behavior.get(flag.category.value, "tag")

            if behavior == "block" and flag.score >= threshold:
                blocked_categories.append(flag.category.value)

        # Determine if safe
        is_safe = len(blocked_categories) == 0 and overall_score >= 0.5

        # Generate user message and alternatives if blocked
        user_message = None
        alternatives = []

        if not is_safe:
            user_message = self._generate_user_message(blocked_categories)
            alternatives = self._generate_alternatives(blocked_categories)

        return SafetyResult(
            job_id=job_id,
            is_safe=is_safe,
            overall_score=overall_score,
            flags=flags,
            blocked_categories=blocked_categories,
            user_message=user_message,
            alternatives=alternatives,
        )

    def _generate_user_message(self, blocked_categories: List[str]) -> str:
        """Generate user-friendly rejection message."""
        default_messages = {
            "nsfw": "Your request contains adult content that we cannot generate.",
            "violence": "Your request contains violent content that violates our policy.",
            "hate": "Your request contains content that violates our community guidelines.",
            "illegal": "Your request contains content that violates our terms of service.",
            "ip_trademark": "Your request may include copyrighted content. The image will be private to your account.",
        }

        messages = {**default_messages, **self._config.user_messages}

        if len(blocked_categories) == 1:
            category = blocked_categories[0]
            return messages.get(category, "Your request cannot be processed due to content policy.")
        else:
            return "Your request cannot be processed due to multiple content policy concerns."

    def _generate_alternatives(self, blocked_categories: List[str]) -> List[str]:
        """Generate alternative suggestions."""
        alternatives = []

        for category in blocked_categories:
            config_alts = self._config.alternatives.get(category, [])
            alternatives.extend(config_alts)

        if not alternatives:
            alternatives = self._config.alternatives.get("general", [
                "Try rephrasing your request with different terms",
                "Focus on general concepts rather than specific restricted content",
            ])

        return alternatives

    async def __aenter__(self) -> "SafetyClassificationService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
