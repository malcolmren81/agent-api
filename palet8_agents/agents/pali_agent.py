"""
Pali Agent - User-facing orchestrator.

This agent handles direct user interaction, gathers requirements through
multi-turn conversation, and coordinates with other agents to complete tasks.

Refactored to delegate to RequirementsAnalysisService for extraction logic
while maintaining orchestration responsibilities.

Documentation Reference: Section 5.2.1
"""

from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from src.utils.logger import get_logger, set_correlation_context

from palet8_agents.core.agent import BaseAgent, AgentContext, AgentResult, AgentState
from palet8_agents.core.message import Conversation, Message, MessageRole
from palet8_agents.core.config import get_config, get_model_profile
from palet8_agents.tools.base import BaseTool

from palet8_agents.services.text_llm_service import TextLLMService, TextLLMServiceError
from palet8_agents.services.requirements_analysis_service import (
    RequirementsAnalysisService,
    RequirementsAnalysisError,
)

# Import RequirementsStatus from models package
from palet8_agents.models import RequirementsStatus

logger = get_logger(__name__)


# Field priority for multiple missing fields (ask highest priority first)
FIELD_PRIORITY = {
    "complexity": 0,  # Ask first - determines generation mode
    "subject": 1,
    "product_type": 2,
    "style": 3,
    "dimensions": 4,
    "character": 5,
    "mood": 6,
    "colors": 7,
    "reference_image": 8,
    "text_content": 9,
}

# Mapping of fields to UI selectors
FIELD_TO_SELECTOR = {
    "complexity": "task_complexity",  # Relax/Standard/Complex
    "product_type": "product_category",
    "style": "aesthetic_style",
    "dimensions": "aspect_ratio",
    "character": "system_character",
    "reference_image": "reference_image",
    "text_content": "text_in_image",
}

# Fallback messages if LLM fails
FALLBACK_MESSAGES = {
    "ui_selector": "Please make a selection from the options above.",
    "general": "Could you tell me more about what you'd like?",
}


PALI_SYSTEM_PROMPT = """You are Pali, the friendly design assistant for Palet8's print-on-demand platform.

YOUR ROLE
Help users turn their ideas into clear design briefs. You're the only agent that talks to users directly.

CONVERSATION STYLE
- Be warm, encouraging, and creative
- Keep exchanges short and natural
- Guide users through the generator options naturally
- When users seem unsure, offer suggestions based on their idea

UI SELECTORS
The generator page has pre-defined selectors for these choices. Reference them instead of asking as plain text questions:
- Product category (apparel, drinkware, wall art, accessories, stickers)
- Product template (specific product within category)
- Aspect ratio (square, landscape, portrait, wide, tall)
- Aesthetic style (realistic, illustration, cartoon, minimal, vintage, streetwear, etc.)
- System character (pre-defined characters from library)
- User character (user's uploaded characters)
- Reference image (upload or URL)
- Text in image (text content and font style)

When these choices are needed, prompt users to use the corresponding selector rather than typing answers.
If the system requires it, you may also output a small marker like USE_SELECTOR(<selector_id>) so the UI knows which control to highlight.

WHAT TO UNCOVER
Through conversation or selector choices, gather:
- Product category and specific template
- Aspect ratio appropriate for the product
- Desired aesthetic style
- Any characters to include (system or user-uploaded)
- Reference images if provided
- Text to include in the design
- The core design concept/idea

WHEN TO HAND OFF
Once selectors are filled and concept is clear:
- Summarize the selections
- Output a structured, machine-readable brief for the planning system
- Mark it ready for planning

If critical choices are missing:
- Highlight which selectors need attention
- Suggest options based on their idea

TONE
Be the creative partner users enjoy working with. Make the design process feel easy and fun."""


class PaliAgent(BaseAgent):
    """
    User-facing orchestrator agent.

    Responsibilities (from Documentation Section 5.2.1):
    - Receive user input and validate (English only, format check)
    - Gather requirements through Q&A
    - Determine when requirements are complete
    - Delegate to Planner Agent
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        text_service: Optional[TextLLMService] = None,
        requirements_service: Optional[RequirementsAnalysisService] = None,
    ):
        """
        Initialize the Pali Agent.

        Args:
            tools: Optional list of tools for the agent
            text_service: Optional TextLLMService for response generation
            requirements_service: Optional RequirementsAnalysisService for requirements extraction
        """
        super().__init__(
            name="pali",
            description="User-facing orchestrator for requirement gathering and task coordination",
            tools=tools,
        )

        self._text_service = text_service
        self._owns_text_service = text_service is None
        self._requirements_service = requirements_service
        self._owns_requirements_service = requirements_service is None

        self.system_prompt = PALI_SYSTEM_PROMPT
        self.model_profile = "pali"
        self.max_qa_rounds = None  # No limit
        self.completeness_threshold = 0.5

    async def _get_text_service(self) -> TextLLMService:
        """Get or create text LLM service."""
        if self._text_service is None:
            self._text_service = TextLLMService(default_profile_name=self.model_profile)
        return self._text_service

    async def _get_requirements_service(self) -> RequirementsAnalysisService:
        """Get or create requirements analysis service."""
        if self._requirements_service is None:
            # Share text service with requirements service for efficiency
            text_service = await self._get_text_service()
            self._requirements_service = RequirementsAnalysisService(text_service=text_service)
            self._owns_requirements_service = True
        return self._requirements_service

    def _get_priority_field(self, missing_fields: List[str]) -> str:
        """
        Get highest priority field from missing fields.
        Returns first field if none have defined priority.
        """
        if not missing_fields:
            return "subject"  # Default fallback

        # Sort by priority (lower number = higher priority)
        sorted_fields = sorted(
            missing_fields,
            key=lambda f: FIELD_PRIORITY.get(f, 99)
        )

        return sorted_fields[0]

    def _get_selector_for_field(
        self,
        missing_fields: List[str],
        current_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Check if UI selector template exists for missing fields.
        Returns selector_id if found, None otherwise.
        """
        for field in missing_fields:
            if field in FIELD_TO_SELECTOR:
                return FIELD_TO_SELECTOR[field]

        return None

    async def close(self) -> None:
        """Close resources."""
        # Close requirements service first (if we own it)
        if self._requirements_service and self._owns_requirements_service:
            await self._requirements_service.close()
            self._requirements_service = None

        # Close text service
        if self._text_service and self._owns_text_service:
            await self._text_service.close()
            self._text_service = None

    async def run(
        self,
        context: AgentContext,
        user_input: Optional[str] = None,
        conversation: Optional[Conversation] = None,
    ) -> AgentResult:
        """
        Execute the Pali Agent's task.

        Args:
            context: Shared execution context
            user_input: User message to process
            conversation: Existing conversation history

        Returns:
            AgentResult with response or delegation instruction
        """
        self._start_execution()

        # Set correlation context for all downstream logs
        set_correlation_context(
            job_id=context.job_id,
            user_id=context.user_id,
            conversation_id=conversation.id if conversation else None,
        )

        logger.info(
            "pali.session.start",
            has_input=user_input is not None,
            input_length=len(user_input) if user_input else 0,
            has_conversation=conversation is not None,
        )

        try:
            # Validate user input
            if user_input:
                validation = await self.validate_input(user_input)

                logger.info(
                    "pali.input.validated",
                    is_valid=validation["is_valid"],
                    issues_count=len(validation.get("issues", [])),
                )

                if not validation["is_valid"]:
                    logger.warning(
                        "pali.input.validation_failed",
                        issues=validation["issues"],
                    )
                    return self._create_result(
                        success=False,
                        data={"issues": validation["issues"]},
                        error="Invalid input: " + ", ".join(validation["issues"]),
                        error_code="INVALID_INPUT",
                        requires_user_input=True,
                    )

            # Initialize or update conversation
            if conversation is None:
                conversation = Conversation(
                    user_id=context.user_id,
                    job_id=context.job_id,
                )

            # Add user message to conversation
            if user_input:
                conversation.add_message(Message(
                    role=MessageRole.USER,
                    content=user_input,
                ))

            # Check if requirements are complete
            requirements_status = await self.analyze_requirements(context, conversation)

            logger.info(
                "pali.requirements.analyzed",
                is_complete=requirements_status.is_complete,
                missing_count=len(requirements_status.missing_fields),
                missing_fields=requirements_status.missing_fields[:5],
                subject=requirements_status.subject,
            )

            if requirements_status.is_complete:
                # Requirements complete - delegate to Planner
                context.requirements = requirements_status.to_dict()

                logger.info(
                    "pali.delegation.triggered",
                    next_agent="planner",
                    subject=requirements_status.subject,
                    style=requirements_status.style,
                )

                return self._create_result(
                    success=True,
                    data={
                        "requirements": requirements_status.to_dict(),
                        "message": f"Great! I have all the information I need. Let me create your design based on: {requirements_status.subject}",
                        "action": "delegate_to_planner",
                    },
                    next_agent="planner",
                )
            else:
                # Need more information - generate clarifying questions
                response = await self.generate_response(context, conversation, requirements_status)

                # Add assistant response to conversation
                conversation.add_message(Message(
                    role=MessageRole.ASSISTANT,
                    content=response,
                ))

                logger.info(
                    "pali.clarification.requested",
                    missing_fields=requirements_status.missing_fields,
                    response_length=len(response),
                )

                return self._create_result(
                    success=True,
                    data={
                        "message": response,
                        "requirements_status": requirements_status.to_dict(),
                        "action": "request_more_info",
                        "conversation_id": conversation.id,
                    },
                    requires_user_input=True,
                )

        except TextLLMServiceError as e:
            logger.error(
                "pali.run.llm_error",
                error_detail=str(e),
                exception_type=type(e).__name__,
            )
            return self._create_result(
                success=False,
                data=None,
                error=f"Failed to process request: {e}",
                error_code="LLM_ERROR",
            )
        except Exception as e:
            logger.error(
                "pali.run.error",
                error_detail=str(e),
                exception_type=type(e).__name__,
                exc_info=True,
            )
            return self._create_result(
                success=False,
                data=None,
                error=f"Unexpected error: {e}",
                error_code="AGENT_ERROR",
            )

    async def validate_input(self, user_input: str) -> Dict[str, Any]:
        """
        Validate user input.

        Delegates to RequirementsAnalysisService for consistent validation.

        Args:
            user_input: Raw user input

        Returns:
            Validation result with is_valid flag and any issues
        """
        requirements_service = await self._get_requirements_service()
        return requirements_service.validate_input(user_input)

    async def analyze_requirements(
        self,
        context: AgentContext,
        conversation: Conversation,
    ) -> RequirementsStatus:
        """
        Analyze conversation to extract requirements.

        Delegates to RequirementsAnalysisService for LLM-based extraction.

        Args:
            context: Current execution context
            conversation: Conversation history

        Returns:
            RequirementsStatus with extracted information
        """
        requirements_service = await self._get_requirements_service()

        # Get existing requirements from context to merge
        existing_requirements = context.requirements if context.requirements else None

        # Use service to analyze conversation
        try:
            status = await requirements_service.analyze_conversation(
                conversation=conversation,
                existing_requirements=existing_requirements,
            )
            return status
        except RequirementsAnalysisError as e:
            logger.warning(
                "pali.requirements.analysis_error",
                error_detail=str(e),
                exception_type=type(e).__name__,
            )
            # Return empty status on error
            return RequirementsStatus()

    async def generate_response(
        self,
        context: AgentContext,
        conversation: Conversation,
        requirements_status: RequirementsStatus,
    ) -> str:
        """
        Generate a response to the user.

        Args:
            context: Current execution context
            conversation: Conversation history
            requirements_status: Current requirements status

        Returns:
            Generated response string
        """
        text_service = await self._get_text_service()

        # Build conversation messages for LLM
        messages_for_llm = [{"role": "system", "content": self.system_prompt}]

        for msg in conversation.messages:
            messages_for_llm.append({
                "role": msg.role.value,
                "content": msg.content,
            })

        # Add context about what's missing
        missing = requirements_status.missing_fields
        if missing:
            context_note = f"\n\n[Note: Still need information about: {', '.join(missing)}. Ask about these naturally.]"
            messages_for_llm[-1]["content"] += context_note

        try:
            result = await text_service.generate_text(
                prompt=messages_for_llm[-1]["content"] if messages_for_llm else "",
                system_prompt=self.system_prompt,
                temperature=0.7,
            )

            return result.content.strip()

        except TextLLMServiceError as e:
            logger.error(
                "pali.response.generation_error",
                error_detail=str(e),
                exception_type=type(e).__name__,
            )
            # Fallback response
            if missing:
                return f"I'd love to help you create something amazing! Could you tell me more about what you'd like the main subject of your design to be?"
            else:
                return "Let me understand your vision better. What kind of design are you looking for?"

    async def generate_clarification_response(
        self,
        questions: List[str],
        missing_fields: List[str],
        planner_message: str,
        current_context: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        Generate clarification response with appropriate message type.

        Handles:
        - Multiple missing fields (prioritizes most critical)
        - LLM failures (uses fallback messages)

        Returns:
            Tuple of (message, message_type, selector_id)
            - message_type: "ui_selector" or "general"
            - selector_id: Only if message_type == "ui_selector"
        """
        # Prioritize: ask most critical field first
        prioritized_field = self._get_priority_field(missing_fields)

        # Check if UI template exists for prioritized field
        selector_id = self._get_selector_for_field([prioritized_field], current_context)

        if selector_id:
            message_type = "ui_selector"
            prompt = f"""UI selector "{selector_id}" will be shown.
Write a SHORT (1 sentence) friendly message guiding user to the selector.
Example: "Pick a style from the options!" or "Choose your product below!"
"""
        else:
            message_type = "general"
            selector_id = None
            prompt = f"""Need to ask user about: {prioritized_field}
Planner's question: {questions}

Write a friendly, conversational question. Keep it short.
Example: "What kind of mood do you want your design to have?"
"""

        # Generate with error handling
        try:
            text_service = await self._get_text_service()
            result = await text_service.generate_text(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.7,
            )
            message = result.content.strip()

            # Validate message is not empty
            if not message:
                raise ValueError("Empty message generated")

        except Exception as e:
            # Use fallback message
            logger.warning(
                "pali.clarification.llm_failed",
                error_detail=str(e),
                message_type=message_type,
                using_fallback=True,
            )
            message = FALLBACK_MESSAGES.get(message_type, FALLBACK_MESSAGES["general"])

        logger.info(
            "pali.clarification.generated",
            message_type=message_type,
            selector_id=selector_id,
            prioritized_field=prioritized_field,
            total_missing_fields=len(missing_fields),
        )

        return message, message_type, selector_id

    async def chat_turn(
        self,
        context: AgentContext,
        user_message: str,
        conversation: Optional[Conversation] = None,
        on_status: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process a single chat turn and return structured response.

        This method is optimized for ChatUI integration with SSE streaming.
        It provides status callbacks for real-time progress updates.

        Args:
            context: Shared execution context
            user_message: User's message
            conversation: Existing conversation history
            on_status: Optional callback for status updates (stage, message, progress)

        Returns:
            Dict with:
                - success: bool
                - message: str (assistant response)
                - requirements_status: dict
                - action: str ("request_more_info" | "delegate_to_planner")
                - next_agent: str | None
        """
        async def emit_status(stage: str, progress: float = 0.0):
            """Emit status update if callback provided."""
            if on_status:
                await on_status(stage, progress)

        logger.info(
            "pali.chat_turn.start",
            message_length=len(user_message),
            has_conversation=conversation is not None,
        )

        try:
            # Emit understanding status
            await emit_status("understanding", 0.1)

            # Validate input
            validation = await self.validate_input(user_message)
            if not validation["is_valid"]:
                return {
                    "success": False,
                    "message": "I couldn't understand that. " + ", ".join(validation["issues"]),
                    "requirements_status": {},
                    "action": "request_more_info",
                    "error": "Invalid input",
                }

            # Initialize conversation if needed
            if conversation is None:
                conversation = Conversation(
                    user_id=context.user_id,
                    job_id=context.job_id,
                )

            # Add user message
            conversation.add_message(Message(
                role=MessageRole.USER,
                content=user_message,
            ))

            # Emit analyzing status
            await emit_status("analyzing", 0.3)

            # Analyze requirements
            requirements_status = await self.analyze_requirements(context, conversation)

            # Check if complete
            if requirements_status.is_complete:
                # Requirements complete - ready to delegate
                await emit_status("complete", 1.0)

                context.requirements = requirements_status.to_dict()

                summary_message = f"Great! I have all the information I need. Let me create your design based on: {requirements_status.subject}"
                if requirements_status.style:
                    summary_message += f" in {requirements_status.style} style"
                if requirements_status.mood:
                    summary_message += f" with a {requirements_status.mood} mood"

                logger.info(
                    "pali.chat_turn.complete",
                    action="delegate_to_planner",
                    subject=requirements_status.subject,
                    style=requirements_status.style,
                )

                return {
                    "success": True,
                    "message": summary_message,
                    "requirements_status": requirements_status.to_dict(),
                    "action": "delegate_to_planner",
                    "next_agent": "planner",
                }

            # Need more information - generate response
            await emit_status("responding", 0.6)

            response = await self.generate_response(context, conversation, requirements_status)

            # Add assistant message to conversation
            conversation.add_message(Message(
                role=MessageRole.ASSISTANT,
                content=response,
            ))

            await emit_status("clarifying", 0.8)

            return {
                "success": True,
                "message": response,
                "requirements_status": requirements_status.to_dict(),
                "action": "request_more_info",
                "next_agent": None,
                "conversation_id": conversation.id,
            }

        except Exception as e:
            logger.error(
                "pali.chat_turn.error",
                error_detail=str(e),
                exception_type=type(e).__name__,
                exc_info=True,
            )
            return {
                "success": False,
                "message": "I'm sorry, something went wrong. Could you try again?",
                "requirements_status": {},
                "action": "request_more_info",
                "error": str(e),
            }

    async def handle_edit_request(
        self,
        context: AgentContext,
        edit_message: str,
        original_requirements: Dict[str, Any],
        conversation: Optional[Conversation] = None,
    ) -> Dict[str, Any]:
        """
        Handle an edit request from the user after viewing a generated result.

        This method understands the context of an existing design and gathers
        requirements for modifications.

        Args:
            context: Shared execution context
            edit_message: User's edit request
            original_requirements: Requirements from the original generation
            conversation: Existing conversation history

        Returns:
            Dict with updated requirements and response
        """
        # Enhance the system prompt for edit context
        edit_context = f"""
The user has already generated a design with these requirements:
- Subject: {original_requirements.get('subject', 'unknown')}
- Style: {original_requirements.get('style', 'not specified')}
- Colors: {', '.join(original_requirements.get('colors', []))}
- Mood: {original_requirements.get('mood', 'not specified')}

They are now requesting modifications. Understand what they want to change
and update the requirements accordingly. Be concise and focused on the changes.
"""
        # Add edit context to the conversation
        if conversation is None:
            conversation = Conversation(
                user_id=context.user_id,
                job_id=context.job_id,
            )

        # Add system context
        conversation.add_message(Message(
            role=MessageRole.SYSTEM,
            content=edit_context,
        ))

        # Process as normal chat turn
        return await self.chat_turn(context, edit_message, conversation)

    async def handle_generate_request(
        self,
        context: AgentContext,
        requirements: Dict[str, Any],
        on_progress: Optional[Callable[[str, float, Optional[str]], None]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle a generation request - Pali stays on as the communication layer.

        Pali delegates orchestration to Planner while remaining the communication
        channel between the system and the user. This method yields events that
        can be streamed to the frontend.

        Args:
            context: Shared execution context
            requirements: Requirements gathered from conversation
            on_progress: Optional callback for progress updates (stage, progress, message)

        Yields:
            Dict events with types: "status", "result", "clarification", "complete", "error"
        """
        from palet8_agents.agents.planner_agent_v2 import PlannerAgentV2

        logger.info(
            "pali.generate.start",
            job_id=context.job_id,
            has_requirements=bool(requirements),
        )

        try:
            # Emit initial status
            yield {
                "type": "status",
                "stage": "starting",
                "progress": 0.0,
                "message": "Starting generation...",
            }

            # Set requirements in context
            context.requirements = requirements

            logger.info(
                "pali.generate.delegating",
                job_id=context.job_id,
                next_agent="planner",
            )

            # Define callback for Planner to communicate through Pali
            async def pali_relay(event_type: str, data: Dict[str, Any]):
                """Relay events from Planner to user via Pali."""
                logger.debug(
                    "pali.relay.event",
                    event_type=event_type,
                    data_keys=list(data.keys()) if data else [],
                )
                # Progress updates are handled via the generator yield

            # Delegate to Planner for orchestration
            yield {
                "type": "status",
                "stage": "planning",
                "progress": 0.1,
                "message": "Planning your design...",
            }

            async with PlannerAgentV2() as planner:
                # Planner orchestrates the full generation flow
                result = await planner.orchestrate_generation(
                    context=context,
                    pali_callback=pali_relay,
                )

            if not result.success:
                # Check if it's a clarification request
                if result.error_code == "CLARIFICATION_NEEDED":
                    missing_fields = result.data.get("missing_fields", [])

                    # Generate response with message type
                    message, message_type, selector_id = await self.generate_clarification_response(
                        questions=result.data.get("questions", []),
                        missing_fields=missing_fields,
                        planner_message=result.data.get("message", ""),
                        current_context=context.requirements,
                    )

                    logger.info(
                        "pali.clarification.from_planner",
                        message_type=message_type,
                        selector_id=selector_id,
                        missing_fields=missing_fields,
                    )

                    yield {
                        "type": "message",
                        "role": "assistant",
                        "content": message,
                        "metadata": {
                            "requires_input": True,
                            "message_type": message_type,
                            "selector_id": selector_id,
                            "missing_fields": missing_fields,
                        },
                    }
                    return

                # Handle other errors
                logger.error(
                    "pali.generate.planner_failed",
                    error=result.error,
                    error_code=result.error_code,
                )
                yield {
                    "type": "error",
                    "error": result.error or "Generation failed",
                    "error_code": result.error_code,
                }
                return

            # Generation successful - present result to user
            images = result.data.get("images", [])
            evaluation = result.data.get("evaluation", {})

            logger.info(
                "pali.result.presenting",
                job_id=context.job_id,
                images_count=len(images),
            )

            yield {
                "type": "result",
                "images": images,
                "evaluation": evaluation,
                "prompt_plan": result.data.get("prompt_plan", {}),
                "message": "Here's your design! Let me know if you'd like any changes.",
            }

            # Note: User confirmation is handled by the endpoint
            # The endpoint will call handle_edit_request if user wants changes
            logger.info(
                "pali.user.confirmation_pending",
                job_id=context.job_id,
            )

            yield {
                "type": "awaiting_confirmation",
                "message": "Waiting for your feedback...",
            }

        except Exception as e:
            logger.error(
                "pali.generate.error",
                error_detail=str(e),
                exception_type=type(e).__name__,
                exc_info=True,
            )
            yield {
                "type": "error",
                "error": str(e),
                "error_code": "GENERATION_ERROR",
            }

    async def confirm_and_complete(
        self,
        context: AgentContext,
        confirmed: bool = True,
    ) -> Dict[str, Any]:
        """
        Complete the session after user confirms the result.

        Args:
            context: Shared execution context
            confirmed: Whether user confirmed the result

        Returns:
            Dict with completion status
        """
        if confirmed:
            logger.info(
                "pali.user.confirmed",
                job_id=context.job_id,
            )
            logger.info(
                "pali.session.complete",
                job_id=context.job_id,
            )
            return {
                "success": True,
                "action": "completed",
                "message": "Great! Your design is ready.",
            }
        else:
            logger.info(
                "pali.user.cancelled",
                job_id=context.job_id,
            )
            return {
                "success": True,
                "action": "cancelled",
                "message": "No problem. Let me know if you'd like to try again.",
            }

    async def handle_stop_request(
        self,
        context: AgentContext,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle user request to stop current task.

        Args:
            context: Shared execution context
            reason: Optional reason for stopping

        Returns:
            Dict with stop confirmation message
        """
        logger.info(
            "pali.stop.requested",
            job_id=context.job_id,
            reason=reason,
        )

        text_service = await self._get_text_service()
        try:
            result = await text_service.generate_text(
                prompt="Generate a short, friendly message confirming you've stopped the design generation. Mention that the conversation is saved and they can try again when ready.",
                system_prompt=self.system_prompt,
                temperature=0.7,
            )
            message = result.content.strip()
        except Exception:
            message = "I've stopped the generation. Your conversation is saved - let me know when you'd like to try again!"

        return {
            "success": True,
            "action": "stopped",
            "message": message,
        }

    async def __aenter__(self) -> "PaliAgent":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
