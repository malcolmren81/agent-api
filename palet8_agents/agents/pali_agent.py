"""
Pali Agent - User-facing orchestrator.

This agent handles direct user interaction, gathers requirements through
multi-turn conversation, and coordinates with other agents to complete tasks.

Refactored to delegate to RequirementsAnalysisService for extraction logic
while maintaining orchestration responsibilities.

Documentation Reference: Section 5.2.1
"""

from typing import Any, Dict, List, Optional

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
                error=str(e),
                error_type=type(e).__name__,
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
                error=str(e),
                error_type=type(e).__name__,
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
                error=str(e),
                error_type=type(e).__name__,
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
                error=str(e),
                error_type=type(e).__name__,
            )
            # Fallback response
            if missing:
                return f"I'd love to help you create something amazing! Could you tell me more about what you'd like the main subject of your design to be?"
            else:
                return "Let me understand your vision better. What kind of design are you looking for?"

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
                error=str(e),
                error_type=type(e).__name__,
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

    async def __aenter__(self) -> "PaliAgent":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
