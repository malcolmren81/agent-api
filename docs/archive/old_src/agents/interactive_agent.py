"""
Interactive Agent - UI Gateway for user requests.

Uses Google ADK BaseAgent.

REFACTORED: Credit logic moved to CreditService (called by orchestrator).
This agent now only validates prompts and user info.
"""
from typing import Any, Dict, Optional
from src.agents.base_agent import BaseAgent, AgentContext, AgentResult
from src.utils import get_logger
from src.models.schemas import ReasoningModel, ImageModel

logger = get_logger(__name__)


class InteractiveAgent(BaseAgent):
    """
    Interactive Agent serves as the entry point for user requests.

    Responsibilities:
    - Validate English-only input
    - Validate prompt format and length
    - Validate required user information
    - Return formatted responses

    NOTE: Credit checking is handled by CreditService (called by orchestrator),
    NOT by this agent. The agent assumes context.username, context.avatar,
    and context.credit_balance have already been populated by orchestrator.
    """

    def __init__(
        self,
        name: str = "InteractiveAgent"
    ) -> None:
        """
        Initialize Interactive Agent.

        Args:
            name: Agent name
        """
        super().__init__(name=name)
        logger.info("Interactive Agent initialized")

    async def run(self, context: AgentContext) -> AgentResult:
        """
        Validate user input and prompt.

        This agent no longer handles credit operations - those are handled by
        CreditService in the orchestrator. It assumes context.username,
        context.avatar, and context.credit_balance have already been populated.

        Responsibilities:
        1. Validate prompt exists and meets minimum length
        2. Validate English-only input
        3. Validate required user fields (customer_id, email)
        4. Return validation result

        Args:
            context: AgentContext with user data already populated by orchestrator

        Returns:
            AgentResult with validation status
        """
        try:
            # Extract prompt from shared_data (set by orchestrator)
            prompt = context.shared_data.get("prompt", "")

            logger.info(
                "Interactive Agent validating request",
                task_id=context.task_id,
                user_id=context.user_id,
                customer_id=context.customer_id,
                shop_domain=context.shop_domain,
                prompt_length=len(prompt),
            )

            # Validate required fields
            if not prompt:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error="Missing required field: prompt",
                    metadata={"error_code": "MISSING_PROMPT"}
                )

            if not context.customer_id or not context.email:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error="Missing required fields: customer_id and email",
                    metadata={"error_code": "MISSING_USER_INFO"}
                )

            # Validate minimum prompt length (for accurate language detection)
            MIN_PROMPT_LENGTH = 10
            if len(prompt.strip()) < MIN_PROMPT_LENGTH:
                logger.warning("Prompt too short", length=len(prompt.strip()))
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error=f"Prompt must be at least {MIN_PROMPT_LENGTH} characters for accurate processing",
                    metadata={"error_code": "PROMPT_TOO_SHORT"}
                )

            # Validate English-only input
            if not self._is_english(prompt):
                logger.warning("Non-English prompt detected")
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error="Only English prompts are supported",
                    metadata={"error_code": "INVALID_LANGUAGE"}
                )

            # Store validated prompt in shared_data
            context.shared_data["validated_prompt"] = prompt

            # Phase 7.1.2: Extract and store template selection
            template_id = context.shared_data.get("template_id")
            template_category = context.shared_data.get("template_category")

            if template_id:
                logger.info(
                    "Template selection detected",
                    template_id=template_id,
                    template_category=template_category
                )
                context.shared_data["template_id"] = template_id
                context.shared_data["template_category"] = template_category
            else:
                logger.debug("No template selected, proceeding with AI art only")

            # Phase 7.3: Extract and store aesthetic selection
            aesthetic_id = context.shared_data.get("aesthetic_id")

            if aesthetic_id:
                logger.info(
                    "Aesthetic selection detected",
                    aesthetic_id=aesthetic_id
                )
                context.shared_data["aesthetic_id"] = aesthetic_id
            else:
                logger.debug("No aesthetic selected, proceeding without style reference")

            # Phase 7.4: Extract and store character selection
            character_id = context.shared_data.get("character_id")

            if character_id:
                logger.info(
                    "Character selection detected",
                    character_id=character_id
                )
                context.shared_data["character_id"] = character_id
            else:
                logger.debug("No character selected, proceeding without character reference")

            logger.info(
                "Interactive Agent validation complete",
                username=context.username,
                credit_balance=context.credit_balance,
                reasoning_model=context.reasoning_model,
                image_model=context.image_model,
                template_id=template_id,
                aesthetic_id=aesthetic_id,
                character_id=character_id,
            )

            return AgentResult(
                agent_name=self.name,
                success=True,
                data={
                    "message": "Request validated and ready for planning",
                    "user": {
                        "username": context.username,
                        "avatar": context.avatar,
                        "credit_balance": context.credit_balance
                    }
                },
                metadata={
                    "prompt_length": len(prompt),
                    "language_validated": True
                }
            )

        except Exception as e:
            logger.error("Unexpected error in Interactive Agent",
                        error=str(e),
                        exc_info=True)
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=f"Unexpected error: {str(e)}",
                metadata={"error_code": "INTERNAL_ERROR"}
            )

    def _is_english(self, text: str) -> bool:
        """
        Check if text is primarily English using robust language detection.

        Uses langdetect library with ASCII heuristic as fallback for performance.

        Args:
            text: Text to validate

        Returns:
            True if English, False otherwise
        """
        try:
            # Handle empty or very short text
            if not text or len(text) < 3:
                return False

            # Whitelist: URLs, emails, common patterns
            # These may contain non-English chars but should be allowed
            import re
            url_pattern = r'https?://\S+|www\.\S+'
            email_pattern = r'\S+@\S+\.\S+'

            # Remove URLs and emails for language detection
            text_for_detection = re.sub(url_pattern, '', text)
            text_for_detection = re.sub(email_pattern, '', text_for_detection)

            # Fast path: ASCII heuristic (90% threshold)
            ascii_chars = sum(1 for c in text_for_detection if ord(c) < 128)
            total_chars = len(text_for_detection)

            if total_chars == 0:
                return True  # Only URLs/emails, allow it

            ascii_ratio = ascii_chars / total_chars

            # If overwhelmingly ASCII, likely English
            if ascii_ratio > 0.95:
                logger.debug("English detected via ASCII heuristic", ascii_ratio=ascii_ratio)
                return True

            # If very low ASCII, likely not English
            if ascii_ratio < 0.7:
                logger.debug("Non-English detected via ASCII heuristic", ascii_ratio=ascii_ratio)
                return False

            # Gray area: use langdetect for robust detection
            try:
                from langdetect import detect_langs, LangDetectException

                # Strip to reduce noise for detection
                text_clean = text_for_detection.strip()
                if len(text_clean) < 15:  # Increased from 10 to 15 for better accuracy
                    # Too short for reliable langdetect, use stricter ASCII threshold
                    return ascii_ratio > 0.90  # Increased from 0.85

                # Use detect_langs to get confidence scores
                detected_langs = detect_langs(text_clean)

                if not detected_langs:
                    # No language detected, fallback to ASCII
                    return ascii_ratio > 0.85

                # Get the most confident language and its probability
                top_lang = detected_langs[0]
                is_english = top_lang.lang == 'en' and top_lang.prob > 0.9  # Require 90% confidence

                logger.debug(
                    "Language detected via langdetect",
                    detected_lang=top_lang.lang,
                    confidence=top_lang.prob,
                    is_english=is_english,
                    ascii_ratio=ascii_ratio,
                )

                return is_english

            except (LangDetectException, ImportError) as e:
                # Fallback to ASCII heuristic if langdetect unavailable or fails
                logger.warning(
                    "Language detection library unavailable, using ASCII fallback",
                    error=str(e)
                )
                return ascii_ratio > 0.90  # Stricter threshold

        except Exception as e:
            logger.error("Language detection error", error=str(e), exc_info=True)
            # Default to allowing the request on error
            return True

    def _validate_product_types(self, product_types: list, prompt: str) -> list:
        """
        Validate and suggest product types.

        Args:
            product_types: User-provided product types
            prompt: User prompt for intelligent defaults

        Returns:
            List of validated product type strings
        """
        from src.models.schemas import ProductType

        MAX_PRODUCT_TYPES = 4
        ALLOWED_TYPES = [pt.value for pt in ProductType]

        # If user provided types, validate them
        if product_types:
            validated = []
            for pt in product_types:
                if pt in ALLOWED_TYPES:
                    validated.append(pt)
                else:
                    logger.warning(f"Invalid product type '{pt}', skipping")

            if len(validated) > MAX_PRODUCT_TYPES:
                logger.warning(f"Too many product types ({len(validated)}), limiting to {MAX_PRODUCT_TYPES}")
                validated = validated[:MAX_PRODUCT_TYPES]

            if validated:
                return validated

        # No valid types provided, use intelligent defaults based on prompt
        return self._suggest_product_types(prompt)

    def _suggest_product_types(self, prompt: str) -> list:
        """
        Suggest product types based on prompt content.

        Args:
            prompt: User prompt

        Returns:
            List of suggested product type strings
        """
        from src.models.schemas import ProductType

        prompt_lower = prompt.lower()

        # Check for explicit mentions
        if "t-shirt" in prompt_lower or "tshirt" in prompt_lower or "shirt" in prompt_lower:
            return [ProductType.TSHIRT.value, ProductType.MUG.value]
        elif "mug" in prompt_lower or "cup" in prompt_lower or "coffee" in prompt_lower:
            return [ProductType.MUG.value, ProductType.TSHIRT.value]
        elif "poster" in prompt_lower or "print" in prompt_lower or "wall art" in prompt_lower:
            return [ProductType.POSTER.value, ProductType.TSHIRT.value]
        elif "phone" in prompt_lower or "case" in prompt_lower or "mobile" in prompt_lower:
            return [ProductType.PHONE_CASE.value, ProductType.TSHIRT.value]

        # Default: t-shirt and mug (most common print-on-demand products)
        logger.info("Using default product types: tshirt, mug")
        return [ProductType.TSHIRT.value, ProductType.MUG.value]

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
