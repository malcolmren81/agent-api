"""
Safety Agent - Content safety and IP/trademark checks.

This agent runs continuously alongside the entire task, monitoring for
safety violations in a non-blocking, event-driven manner.

Documentation Reference: Section 5.2.4
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import asyncio
import yaml

from src.utils.logger import get_logger

from palet8_agents.core.agent import BaseAgent, AgentContext, AgentResult, AgentState
from palet8_agents.core.config import get_config
from palet8_agents.tools.base import BaseTool

from palet8_agents.services.text_llm_service import TextLLMService, TextLLMServiceError
from palet8_agents.services.safety_classification_service import SafetyClassificationService

# Import from models package (refactored - no longer using inline classes)
from palet8_agents.models import (
    SafetyCategory,
    SafetySeverity,
    SafetyFlag,
    SafetyResult,
)

logger = get_logger(__name__)

# Config file path
SAFETY_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "safety_config.yaml"


SAFETY_SYSTEM_PROMPT = """You are the Safety monitor for Palet8's image generation system.

YOUR ROLE
Continuously scan for unsafe content and IP risks. Run alongside other agents, producing safety signals without blocking unless necessary.

WHAT YOU MONITOR
- User messages and briefs
- Generated prompts and plans
- Reference materials and context
- Image descriptions and metadata

RISK CATEGORIES
Assess risk levels for:
- Adult/sexual content
- Violence and harm
- Hate speech and discrimination
- Intellectual property (brands, characters, logos, trademarks - e.g., major entertainment franchises, sports brands, or other well-known marks)
- Illegal activities

EXECUTION MODEL: CONTINUOUS NON-BLOCKING

Monitor events as they flow. Your job is to produce safety signals and recommended actions. The orchestration layer decides whether to pause, block, or continue a job.

BEHAVIOR PRINCIPLES

1. CONTEXT MATTERS
   "Avoid violence" in negative prompt reduces risk. Brand mentions in "don't include" context differ from copy requests.

2. PROPORTIONAL RESPONSE
   - Low risk: Flag and note that processing can continue
   - Medium risk: Flag for review and note that processing should continue with caution
   - High risk: Recommend manual review before proceeding
   - Critical: Recommend blocking the job and explain why

3. BE HELPFUL
   When recommending a block, suggest safer directions. Help users achieve goals within bounds.

4. AGGREGATE VIEW
   Maintain job-level safety picture. Individual flags combine into overall recommendation.

OUTPUT
Produce structured safety signals (flags with category, severity, short explanation, and source) and, when needed, a job-level overall safety summary.

Your outputs are guidance for other agents and the orchestration layer; you never respond directly to the user. Pali handles all user communication."""


# =============================================================================
# NOTE: Data classes (SafetyCategory, SafetySeverity, SafetyFlag, SafetyResult)
# have been moved to palet8_agents.models package.
# Import them from there for consistency.
# =============================================================================


class SafetyAgent(BaseAgent):
    """
    Content safety monitoring agent.

    Execution Model: Continuous Non-Blocking Monitoring (from Swimlane)

    The Safety Agent:
    - Spawns at task start
    - Monitors events as they occur (user input, requirements, prompts, images)
    - Flags issues asynchronously
    - Only halts if critical violation detected
    - Accumulates safety score throughout task
    - NEVER blocks the main flow unless critical

    Event Types Monitored:
    - user_input: Initial user message
    - requirements: Gathered requirements
    - context: RAG results and references
    - prompt: Final generation prompt
    - image: Generated image
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        text_service: Optional[TextLLMService] = None,
        safety_classification_service: Optional[SafetyClassificationService] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize the Safety Agent.

        Args:
            tools: Optional list of tools for the agent
            text_service: Optional TextLLMService for LLM calls
            safety_classification_service: Optional SafetyClassificationService for classification
            config_path: Optional path to safety config YAML file
        """
        super().__init__(
            name="safety",
            description="Content safety monitoring agent (non-blocking, continuous)",
            tools=tools,
        )

        self._text_service = text_service
        self._safety_classification_service = safety_classification_service
        self._owns_services = {
            "text": text_service is None,
            "safety_classification": safety_classification_service is None,
        }

        # System prompt
        self.system_prompt = SAFETY_SYSTEM_PROMPT

        # Model profile
        self.model_profile = "safety"

        # Load safety configuration from YAML (single source of truth)
        self._config_path = config_path or SAFETY_CONFIG_PATH
        self._safety_config = self._load_safety_config()

        # Extract config values
        self.blocking_behavior = self._safety_config.get("blocking_behavior", {})
        self.tagging_thresholds = self._safety_config.get("tagging_thresholds", {})
        self.nsfw_keywords = self._safety_config.get("nsfw", {}).get("critical_keywords", [])
        self.extreme_block_keywords = self._safety_config.get("extreme_block_keywords", {})
        self.ip_detection = self._safety_config.get("ip_detection", {})
        self.user_messages = self._safety_config.get("user_messages", {})
        self.alternatives_config = self._safety_config.get("alternatives", {})

        # Build IP blocklist from known entities
        self.ip_blocklist = self._build_ip_blocklist()

        # Legacy config for backward compatibility
        self.config = get_config()
        self.blocking_thresholds = {
            cat: self.tagging_thresholds.get(cat, 0.8)
            for cat in ["nsfw", "violence", "hate", "ip_trademark", "illegal"]
        }

        # Accumulated state per job
        self._job_flags: Dict[str, List[SafetyFlag]] = {}
        self._job_scores: Dict[str, float] = {}

    def _load_safety_config(self) -> Dict[str, Any]:
        """Load safety configuration from YAML file."""
        try:
            if self._config_path.exists():
                with open(self._config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            else:
                logger.warning(
                    "safety.config.not_found",
                    config_path=str(self._config_path),
                )
                return {}
        except Exception as e:
            logger.error(
                "safety.config.load_error",
                config_path=str(self._config_path),
                error=str(e),
                error_type=type(e).__name__,
            )
            return {}

    def _build_ip_blocklist(self) -> Set[str]:
        """Build IP blocklist from known entities in config."""
        blocklist = set()
        known_entities = self.ip_detection.get("known_entities", {})

        for category, entities in known_entities.items():
            if isinstance(entities, list):
                blocklist.update(e.lower() for e in entities)

        return blocklist

    async def _get_text_service(self) -> TextLLMService:
        """Get or create text LLM service."""
        if self._text_service is None:
            self._text_service = TextLLMService(default_profile_name=self.model_profile)
        return self._text_service

    async def _get_safety_classification_service(self) -> SafetyClassificationService:
        """Get or create safety classification service."""
        if self._safety_classification_service is None:
            text_service = await self._get_text_service()
            self._safety_classification_service = SafetyClassificationService(
                text_service=text_service,
                config_path=self._config_path,
            )
        return self._safety_classification_service

    async def close(self) -> None:
        """Close resources."""
        # Close classification service first (it may depend on text service)
        if self._safety_classification_service and self._owns_services["safety_classification"]:
            await self._safety_classification_service.close()
            self._safety_classification_service = None

        if self._text_service and self._owns_services["text"]:
            await self._text_service.close()
            self._text_service = None

    async def run(
        self,
        context: AgentContext,
        user_input: Optional[str] = None,
    ) -> AgentResult:
        """
        Run a full safety check on the current context.

        For continuous monitoring, use on_event() instead.

        Args:
            context: Shared execution context
            user_input: Optional text to check

        Returns:
            AgentResult with safety verdict
        """
        self._start_execution()

        logger.info(
            "safety.run.start",
            job_id=context.job_id,
            has_user_input=user_input is not None,
            has_requirements=context.requirements is not None,
            has_plan=context.plan is not None,
        )

        try:
            # Check all available data in context
            if user_input:
                await self.on_event("user_input", user_input, context.job_id)

            if context.requirements:
                await self.on_event("requirements", context.requirements, context.job_id)

            if context.plan:
                prompt = context.plan.get("prompt", "")
                if prompt:
                    await self.on_event("prompt", prompt, context.job_id)

            # Get final verdict
            result = await self.get_safety_verdict(context.job_id)

            # Update context safety tracking
            context.safety_score = result.overall_score
            context.safety_flags = [f.category.value for f in result.flags]

            return self._create_result(
                success=True,
                data={"safety_result": result.to_dict()},
            )

        except Exception as e:
            logger.error(
                "safety.run.error",
                job_id=context.job_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return self._create_result(
                success=False,
                data=None,
                error=f"Safety check failed: {e}",
                error_code="SAFETY_ERROR",
            )

    async def start_monitoring(self, job_id: str) -> None:
        """
        Start background monitoring for a job.

        This initializes the accumulated state for continuous monitoring.

        Args:
            job_id: Job identifier to monitor
        """
        self._job_flags[job_id] = []
        self._job_scores[job_id] = 1.0  # Start with perfect score
        logger.info(
            "safety.monitoring.start",
            job_id=job_id,
        )

    async def on_event(
        self,
        event_type: str,
        data: Any,
        job_id: str,
    ) -> Optional[SafetyFlag]:
        """
        Handle an incoming event without blocking main flow.

        This is the core non-blocking monitoring method.

        Args:
            event_type: Type of event (user_input, requirements, prompt, image)
            data: Event data to check
            job_id: Associated job ID

        Returns:
            SafetyFlag if issue detected, None otherwise
        """
        logger.debug(
            "safety.event.received",
            event_type=event_type,
            job_id=job_id,
            data_length=len(str(data)) if data else 0,
        )

        try:
            # Initialize job tracking if needed
            if job_id not in self._job_flags:
                await self.start_monitoring(job_id)

            flag = None

            if event_type in ["user_input", "prompt"]:
                flag = await self._check_text_content(data, event_type, job_id)

            elif event_type == "requirements":
                # Check combined requirements text
                text = self._requirements_to_text(data)
                flag = await self._check_text_content(text, event_type, job_id)

            elif event_type == "context":
                # Check reference images/context for IP issues
                flag = await self._check_context(data, job_id)

            elif event_type == "image":
                # Check generated image
                flag = await self._check_image(data, job_id)

            if flag:
                self._job_flags[job_id].append(flag)
                self._update_job_score(job_id, flag)

                logger.warning(
                    "safety.flag.detected",
                    job_id=job_id,
                    category=flag.category.value,
                    severity=flag.severity.value,
                    score=flag.score,
                    source=flag.source,
                )

                # Check if this triggers a critical block
                if flag.severity == SafetySeverity.CRITICAL:
                    logger.error(
                        "safety.violation.critical",
                        job_id=job_id,
                        category=flag.category.value,
                        description=flag.description,
                    )

            return flag

        except Exception as e:
            logger.error(
                "safety.event.error",
                job_id=job_id,
                event_type=event_type,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    async def get_safety_verdict(self, job_id: str) -> SafetyResult:
        """
        Get final accumulated safety result for a job.

        Args:
            job_id: Job identifier

        Returns:
            SafetyResult with accumulated flags and decision
        """
        flags = self._job_flags.get(job_id, [])
        overall_score = self._job_scores.get(job_id, 1.0)

        # Determine blocked categories
        blocked_categories = []
        for flag in flags:
            threshold = self.blocking_thresholds.get(flag.category.value, 0.8)
            if flag.score >= threshold:
                blocked_categories.append(flag.category.value)

        # Determine if safe
        is_safe = len(blocked_categories) == 0 and overall_score >= 0.5

        # Generate user message if blocked
        user_message = None
        alternatives = []

        if not is_safe:
            user_message = self._generate_user_message(blocked_categories, flags)
            alternatives = self._generate_alternatives(blocked_categories)

        logger.info(
            "safety.verdict.generated",
            job_id=job_id,
            is_safe=is_safe,
            overall_score=overall_score,
            flags_count=len(flags),
            blocked_categories=blocked_categories,
        )

        return SafetyResult(
            job_id=job_id,
            is_safe=is_safe,
            overall_score=overall_score,
            flags=flags,
            blocked_categories=blocked_categories,
            user_message=user_message,
            alternatives=alternatives,
            metadata={
                "total_checks": len(flags),
                "blocking_thresholds": self.blocking_thresholds,
            },
        )

    async def _check_text_content(
        self,
        text: str,
        source: str,
        job_id: str,
    ) -> Optional[SafetyFlag]:
        """Check text content for safety issues."""
        if not text or len(text.strip()) < 3:
            return None

        # Quick keyword check first (fast, no LLM)
        keyword_flag = self._quick_keyword_check(text, source)
        if keyword_flag and keyword_flag.severity == SafetySeverity.CRITICAL:
            return keyword_flag

        # IP/Trademark check
        ip_flag = self._check_ip_trademark(text, source)
        if ip_flag:
            return ip_flag

        # LLM-based classification for nuanced content
        try:
            text_service = await self._get_text_service()

            result = await text_service.generate_text(
                prompt=f"Classify this content for safety issues:\n\n\"{text[:1000]}\"",
                system_prompt=self.system_prompt,
                temperature=0.0,  # Deterministic
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
                return SafetyFlag(
                    category=SafetyCategory(max_category),
                    severity=severity,
                    score=max_score,
                    description=f"Potential {max_category} content detected",
                    source=source,
                    metadata={"all_scores": scores},
                )

        except Exception as e:
            logger.warning(
                "safety.llm_check.failed",
                error=str(e),
                error_type=type(e).__name__,
            )

        return None

    async def _check_context(
        self,
        context_data: Any,
        job_id: str,
    ) -> Optional[SafetyFlag]:
        """Check context/references for IP issues."""
        # Placeholder for reference image IP checking
        return None

    async def _check_image(
        self,
        image_data: Any,
        job_id: str,
    ) -> Optional[SafetyFlag]:
        """Check generated image for safety issues."""
        # Placeholder for vision-based safety checking
        return None

    def _quick_keyword_check(
        self,
        text: str,
        source: str,
    ) -> Optional[SafetyFlag]:
        """Fast keyword-based safety check using config values."""
        text_lower = text.lower()

        # NSFW keywords - always block (from config)
        for keyword in self.nsfw_keywords:
            if keyword.lower() in text_lower:
                return SafetyFlag(
                    category=SafetyCategory.NSFW,
                    severity=SafetySeverity.CRITICAL,
                    score=1.0,
                    description="Prohibited adult content",
                    source=source,
                    metadata={"matched_keyword": keyword},
                )

        # Extreme block keywords by category (from config)
        for category, keywords in self.extreme_block_keywords.items():
            if isinstance(keywords, list):
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        return SafetyFlag(
                            category=SafetyCategory(category),
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
        """Check for IP/trademark violations using config blocklist."""
        text_lower = text.lower()

        # Check against blocklist (loaded from config/safety_config.yaml)
        # IP detection: never blocks, only tags (per config)
        for blocked_term in self.ip_blocklist:
            if blocked_term in text_lower:
                # Determine severity based on config behavior
                # IP never blocks per policy, just tags
                ip_blocks = self.blocking_behavior.get("ip_trademark") == "block"
                severity = SafetySeverity.HIGH if ip_blocks else SafetySeverity.MEDIUM

                logger.warning(
                    "safety.ip.detected",
                    term=blocked_term,
                    source=source,
                    action="tag",
                )

                return SafetyFlag(
                    category=SafetyCategory.IP_TRADEMARK,
                    severity=severity,
                    score=self.tagging_thresholds.get("ip_trademark", 0.5),
                    description=f"Known brand/character detected: {blocked_term}",
                    source=source,
                    metadata={
                        "matched_term": blocked_term,
                        "action": "tag",  # IP never blocks, just tags
                        "visibility": self.ip_detection.get("visibility_control", {}).get("mode", "user_private"),
                    },
                )

        return None

    def _requirements_to_text(self, requirements: Dict[str, Any]) -> str:
        """Convert requirements dict to text for checking."""
        parts = []
        for key, value in requirements.items():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(v) for v in value)
        return " ".join(parts)

    def _parse_safety_scores(self, response: str) -> Dict[str, float]:
        """Parse safety scores from LLM response."""
        import json
        try:
            # Try to parse JSON from response
            import re
            json_match = re.search(r'\{[^{}]+\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        # Default scores if parsing fails
        return {cat.value: 0.0 for cat in SafetyCategory}

    def _score_to_severity(self, score: float) -> SafetySeverity:
        """Convert score to severity level."""
        if score >= 0.9:
            return SafetySeverity.CRITICAL
        elif score >= 0.7:
            return SafetySeverity.HIGH
        elif score >= 0.5:
            return SafetySeverity.MEDIUM
        elif score >= 0.3:
            return SafetySeverity.LOW
        return SafetySeverity.NONE

    def _update_job_score(self, job_id: str, flag: SafetyFlag) -> None:
        """Update accumulated safety score for a job."""
        # Reduce score based on flag severity
        penalties = {
            SafetySeverity.NONE: 0.0,
            SafetySeverity.LOW: 0.05,
            SafetySeverity.MEDIUM: 0.15,
            SafetySeverity.HIGH: 0.3,
            SafetySeverity.CRITICAL: 0.5,
        }

        penalty = penalties.get(flag.severity, 0.1)
        self._job_scores[job_id] = max(0.0, self._job_scores[job_id] - penalty)

    def _generate_user_message(
        self,
        blocked_categories: List[str],
        flags: List[SafetyFlag],
    ) -> str:
        """Generate user-friendly rejection message from config."""
        # Messages from safety_config.yaml
        default_messages = {
            "nsfw": "Your request contains adult content that we cannot generate.",
            "extreme_violence": "Your request contains content that violates our safety policy.",
            "extreme_hate": "Your request contains content that violates our community guidelines.",
            "extreme_illegal": "Your request contains content that violates our terms of service.",
        }

        # Override with config values
        messages = {**default_messages, **self.user_messages}

        if len(blocked_categories) == 1:
            category = blocked_categories[0]
            # Map category to message key
            if category == "nsfw":
                return messages.get("nsfw", default_messages["nsfw"])
            elif category == "violence":
                return messages.get("extreme_violence", "Your request contains violent content that we cannot generate.")
            elif category == "hate":
                return messages.get("extreme_hate", "Your request contains hateful content that we cannot generate.")
            elif category == "illegal":
                return messages.get("extreme_illegal", default_messages["extreme_illegal"])
            elif category == "ip_trademark":
                return "Your request may include copyrighted or trademarked content. The image will be private to your account."
            else:
                return "Your request cannot be processed due to content policy."
        else:
            return "Your request cannot be processed due to multiple content policy concerns."

    def _generate_alternatives(self, blocked_categories: List[str]) -> List[str]:
        """Generate alternative suggestions from config."""
        alternatives = []

        # Get alternatives from config
        if "nsfw" in blocked_categories:
            config_alts = self.alternatives_config.get("nsfw", [])
            alternatives.extend(config_alts)

        if "ip_trademark" in blocked_categories:
            config_alts = self.alternatives_config.get("ip_trademark", [])
            alternatives.extend(config_alts)

        # Fallback to general alternatives
        if not alternatives:
            config_alts = self.alternatives_config.get("general", [])
            if config_alts:
                alternatives.extend(config_alts)
            else:
                alternatives.append("Try rephrasing your request with different terms")
                alternatives.append("Focus on general concepts rather than specific restricted content")

        return alternatives

    async def __aenter__(self) -> "SafetyAgent":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
