"""
Chat API routes for Pali agent conversations.

This module provides endpoints for multi-turn chat conversations with the Pali agent,
enabling the ChatUI in the generator page to communicate with the agent system.

Endpoints:
- POST /chat/start - Start a new conversation
- POST /chat/message - Send a message and receive SSE stream response
- GET /chat/history/{conversation_id} - Get conversation history
- POST /chat/generate - Trigger full generation pipeline after requirements gathered
"""

from uuid import uuid4
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json
import asyncio
import re

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.utils import get_logger
from src.database import prisma
from palet8_agents.agents.pali_agent import PaliAgent
from palet8_agents.core.agent import AgentContext
from palet8_agents.core.message import Conversation, Message, MessageRole
from palet8_agents.services.assembly_service import AssemblyService
from palet8_agents.models import AssemblyRequest, GenerationParameters, PipelineConfig

logger = get_logger(__name__)
router = APIRouter()


# ============================================================================
# Selector Response Parsing
# ============================================================================

# Map selector_id to the field name used in requirements/context
SELECTOR_TO_FIELD = {
    "task_complexity": "complexity",
    "aesthetic_style": "style",
    "product_category": "product_type",
    "aspect_ratio": "dimensions",
    "system_character": "character",
    "reference_image": "reference_image",
    "text_in_image": "text_content",
}

# Regex to match [SELECTOR:selector_id:value] format
SELECTOR_PATTERN = re.compile(r'\[SELECTOR:([^:]+):([^\]]+)\]')


def parse_selector_response(message: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse a selector response message to extract selector_id and value.

    Format: [SELECTOR:selector_id:value] optional_label
    Example: [SELECTOR:task_complexity:standard] ⚡

    Returns:
        Tuple of (selector_id, value, field_name) if found, None otherwise
    """
    match = SELECTOR_PATTERN.search(message)
    if match:
        selector_id = match.group(1)
        value = match.group(2)
        field_name = SELECTOR_TO_FIELD.get(selector_id, selector_id)
        return (selector_id, value, field_name)
    return None


async def update_job_requirements(job_id: str, field_name: str, value: str) -> None:
    """
    Update a job's requirements with a new field value.

    Args:
        job_id: The job ID to update
        field_name: The requirement field name (e.g., 'complexity')
        value: The value to set
    """
    try:
        from prisma import Json

        # Get current job
        job = await prisma.job.find_unique(where={"id": job_id})
        if not job:
            logger.warning(f"Job {job_id} not found for requirements update")
            return

        # Merge new field into existing requirements
        current_reqs = job.requirements or {}
        if isinstance(current_reqs, str):
            current_reqs = json.loads(current_reqs)

        current_reqs[field_name] = value

        # Update job
        await prisma.job.update(
            where={"id": job_id},
            data={"requirements": Json(current_reqs)}
        )

        logger.info(
            "job.requirements.updated",
            job_id=job_id,
            field=field_name,
            value=value,
        )
    except Exception as e:
        logger.error(f"Failed to update job requirements: {e}")


# ============================================================================
# Natural Language Field Extraction
# ============================================================================

# Fields that can be extracted from natural language replies
EXTRACTABLE_FIELDS = ["subject", "style", "mood", "colors"]


def extract_fields_from_reply(message: str, missing_fields: List[str]) -> Dict[str, str]:
    """
    Extract field values from natural language user replies.

    When user is asked "What would you like the main subject to be?" and replies
    "a cute cat", we should extract subject="a cute cat".

    Args:
        message: User's reply message
        missing_fields: Fields that were asked about (from previous AI message metadata)

    Returns:
        Dict of field_name -> extracted_value
    """
    extracted = {}

    # Skip if message looks like a selector response
    if message.startswith("[SELECTOR:"):
        return extracted

    # Clean up the message
    cleaned = message.strip()

    # Remove common conversational prefixes
    prefixes_to_remove = [
        "I want ", "I'd like ", "I would like ",
        "Make me ", "Create ", "Generate ", "Draw ",
        "The subject is ", "It's ", "It is ",
        "How about ", "Let's do ", "Let's make ",
        "Please ", "Can you ", "Could you ",
    ]
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
            break

    # If subject was missing and user provides a reasonable reply, treat it as subject
    if "subject" in missing_fields and cleaned and len(cleaned) < 500:
        extracted["subject"] = cleaned
        logger.info("chat.natural_language.subject_extracted", value=cleaned[:50])

    # TODO: Add extraction for other fields (style, mood, colors) if needed

    return extracted


async def get_last_assistant_message_metadata(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata from the last assistant message.

    Used to determine what fields were being asked about, so we can
    properly extract values from the user's reply.

    Args:
        conversation_id: The conversation ID

    Returns:
        Metadata dict or None
    """
    try:
        messages = await prisma.message.find_many(
            where={
                "conversationId": conversation_id,
                "role": "assistant",
            },
            order_by={"createdAt": "desc"},
            take=1,
        )

        if messages and messages[0].metadata:
            metadata = messages[0].metadata
            if isinstance(metadata, str):
                return json.loads(metadata)
            return metadata
        return None
    except Exception as e:
        logger.warning(f"Failed to get last assistant message metadata: {e}")
        return None


# ============================================================================
# Request/Response Models
# ============================================================================

class GeneratorContext(BaseModel):
    """Context from the generator page selectors."""
    template_id: Optional[str] = None
    template_category: Optional[str] = None
    aesthetic_id: Optional[str] = None
    aesthetic_name: Optional[str] = None
    character_id: Optional[str] = None
    character_name: Optional[str] = None
    dimensions: str = "1024x1024"
    creativity: int = 50
    image_reference_url: Optional[str] = None
    text_prompt: Optional[str] = None  # User's text prompt from Step 4


class StartConversationRequest(BaseModel):
    """Request to start a new conversation."""
    user_id: str
    customer_id: Optional[str] = None
    email: Optional[str] = None
    shop_domain: Optional[str] = None
    context: Optional[GeneratorContext] = None
    initial_message: Optional[str] = None


class StartConversationResponse(BaseModel):
    """Response after starting a conversation."""
    conversation_id: str
    job_id: str
    status: str
    initial_response: Optional[str] = None
    requirements_status: Optional[Dict[str, Any]] = None
    action: Optional[str] = None  # "delegate_to_planner" when requirements complete
    ready_to_generate: bool = False  # True when generation can proceed


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    conversation_id: str
    message: str
    user_id: str
    context: Optional[GeneratorContext] = None


class ChatMessageResponse(BaseModel):
    """A single chat message."""
    id: str
    role: str
    content: Optional[str] = None
    created_at: str
    metadata: Optional[Dict[str, Any]] = None


class ConversationHistoryResponse(BaseModel):
    """Response with conversation history."""
    conversation_id: str
    job_id: Optional[str] = None
    status: str
    messages: List[ChatMessageResponse]
    requirements_status: Optional[Dict[str, Any]] = None


class TriggerGenerationRequest(BaseModel):
    """Request to trigger full generation pipeline."""
    conversation_id: str
    user_id: str
    customer_id: Optional[str] = None
    email: Optional[str] = None
    shop_domain: Optional[str] = None
    context: Optional[GeneratorContext] = None


# ============================================================================
# Status Message Mapping (User-facing "we" messages)
# ============================================================================

STATUS_MESSAGES = {
    "understanding": "We're understanding your design requirements...",
    "clarifying": "We need a bit more information...",
    "planning": "We're planning your design...",
    "rag": "We're finding similar styles for inspiration...",
    "prompt": "We're crafting the perfect prompt...",
    "model_selection": "We're selecting the best model for your design...",
    "generation_start": "We're generating your image...",
    "generation_progress": "We're still working on it...",
    "evaluation": "We're checking the quality...",
    "evaluation_retry": "We're making some improvements...",
    "product_mockup": "We're creating product mockups...",
    "complete": "Your design is ready!",
    "error": "Something went wrong. Let's try again...",
}


def create_status_event(stage: str, progress: Optional[float] = None, internal: Optional[str] = None) -> str:
    """Create a status event for SSE streaming."""
    message = STATUS_MESSAGES.get(stage, f"We're working on your design...")
    if progress is not None:
        message = f"{message} ({int(progress * 100)}%)"

    event_data = {
        "type": "status",
        "stage": stage,
        "message": message,
        "progress": progress,
    }

    # Internal notes for logging (not sent to client, but useful for debugging)
    if internal:
        logger.debug(f"Status event internal: {internal}")

    return f"event: status\ndata: {json.dumps(event_data)}\n\n"


def create_message_event(role: str, content: str, metadata: Optional[Dict] = None) -> str:
    """Create a message event for SSE streaming."""
    event_data = {
        "type": "message",
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if metadata:
        event_data["metadata"] = metadata

    return f"event: message\ndata: {json.dumps(event_data)}\n\n"


def create_requirements_event(requirements: Dict[str, Any], is_complete: bool) -> str:
    """Create a requirements status event."""
    event_data = {
        "type": "requirements",
        "requirements": requirements,
        "is_complete": is_complete,
    }
    return f"event: requirements\ndata: {json.dumps(event_data)}\n\n"


# ============================================================================
# Helper Functions
# ============================================================================

async def get_or_create_conversation(conversation_id: str, user_id: str) -> Dict[str, Any]:
    """Get existing conversation or raise 404."""
    conversation = await prisma.conversation.find_unique(
        where={"id": conversation_id},
        include={"messages": True, "job": True}
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if conversation.userId != user_id:
        raise HTTPException(status_code=403, detail="Conversation belongs to different user")

    return conversation


async def save_message(conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> None:
    """Save a message to the database."""
    from prisma import Json
    await prisma.chatmessage.create(
        data={
            "conversationId": conversation_id,
            "role": role,
            "content": content,
            "metadata": Json(metadata) if metadata else Json({}),
        }
    )


def build_conversation_from_db(db_conversation: Any) -> Conversation:
    """Build a Conversation object from database record."""
    conversation = Conversation(
        id=db_conversation.id,
        user_id=db_conversation.userId,
        job_id=db_conversation.jobId,
        status=db_conversation.status,
    )

    if db_conversation.messages:
        for msg in db_conversation.messages:
            message = Message(
                role=MessageRole(msg.role),
                content=msg.content,
                id=msg.id,
            )
            conversation.add_message(message)

    return conversation


async def trigger_generation_internal(
    conversation_id: str,
    user_id: str,
    job_id: str,
):
    """
    Resume generation after clarification answered.

    This is an internal helper that re-triggers the generation pipeline
    when a user has provided the clarification Planner needed.

    Yields:
        SSE events for the generation progress
    """
    from prisma import Json

    job = await prisma.job.find_unique(where={"id": job_id})
    if not job:
        yield f"data: {json.dumps({'type': 'error', 'error': 'Job not found'})}\n\n"
        return

    requirements = json.loads(job.requirements) if isinstance(job.requirements, str) else (job.requirements or {})

    await prisma.job.update(
        where={"id": job_id},
        data={"status": "GENERATING"}
    )

    logger.info(
        "pali.generation.internal_trigger",
        job_id=job_id,
        conversation_id=conversation_id,
    )

    agent_context = AgentContext(
        user_id=user_id,
        job_id=job_id,
        conversation_id=conversation_id,
    )
    agent_context.requirements = requirements

    async with PaliAgent() as pali:
        async for event in pali.handle_generate_request(
            context=agent_context,
            requirements=requirements,
        ):
            event_type = event.get("type")

            if event_type == "status":
                yield create_status_event(
                    event.get("stage", "working"),
                    progress=event.get("progress", 0.5),
                )

            elif event_type == "result":
                yield create_status_event("complete", progress=1.0)
                await prisma.job.update(
                    where={"id": job_id},
                    data={"status": "COMPLETED"}
                )

                images = event.get("images", [])
                best_image = images[0] if images else {}
                best_image["score"] = 0.9
                best_image["approved"] = True

                yield f"data: {json.dumps({'type': 'complete', 'success': True, 'best_image': best_image})}\n\n"

            elif event_type == "message" and event.get("metadata", {}).get("requires_input"):
                # Another clarification needed
                metadata = event.get("metadata", {})
                await prisma.job.update(
                    where={"id": job_id},
                    data={"status": "AWAITING_CLARIFICATION"}
                )

                # Save clarification message
                await save_message(conversation_id, "assistant", event.get("content", ""), {
                    "message_type": metadata.get("message_type"),
                    "selector_id": metadata.get("selector_id"),
                    "missing_fields": metadata.get("missing_fields", []),
                })

                yield create_message_event("assistant", event.get("content", ""), metadata)

            elif event_type == "error":
                yield create_status_event("error")
                yield f"data: {json.dumps({'type': 'error', 'error': event.get('error')})}\n\n"


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/start", response_model=StartConversationResponse)
async def start_conversation(request: StartConversationRequest) -> StartConversationResponse:
    """
    Start a new conversation with the Pali agent.

    This creates a new conversation and job in the database, and optionally
    processes an initial message from the user.

    Args:
        request: Start conversation request with user info and optional context

    Returns:
        Conversation ID, job ID, and optional initial response
    """
    try:
        logger.info(
            "Starting new conversation",
            user_id=request.user_id,
            has_initial_message=request.initial_message is not None,
        )

        # Create job first
        job = await prisma.job.create(
            data={
                "userId": request.user_id,
                "status": "COLLECTING_REQUIREMENTS",
                "metadata": json.dumps({
                    "customer_id": request.customer_id,
                    "email": request.email,
                    "shop_domain": request.shop_domain,
                    "generator_context": request.context.model_dump() if request.context else None,
                }),
            }
        )

        # Create conversation linked to job
        conversation = await prisma.conversation.create(
            data={
                "userId": request.user_id,
                "jobId": job.id,
                "status": "active",
                "metadata": json.dumps({
                    "generator_context": request.context.model_dump() if request.context else None,
                }),
            }
        )

        initial_response = None
        requirements_status = None

        # If there's an initial message, process it with Pali
        if request.initial_message:
            # Save user message
            await save_message(conversation.id, "user", request.initial_message)

            # Process with Pali agent
            async with PaliAgent() as pali:
                agent_context = AgentContext(
                    user_id=request.user_id,
                    job_id=job.id,
                    conversation_id=conversation.id,
                )

                # Pass generator context to agent as pre-filled requirements
                if request.context:
                    agent_context.requirements = {
                        "template_id": request.context.template_id,
                        "template_category": request.context.template_category,
                        "aesthetic_id": request.context.aesthetic_id,
                        "aesthetic_name": request.context.aesthetic_name,
                        "character_id": request.context.character_id,
                        "character_name": request.context.character_name,
                        "dimensions": request.context.dimensions,
                        "creativity": request.context.creativity,
                        "image_reference_url": request.context.image_reference_url,
                    }
                    logger.info(
                        "Passing generator context to Pali",
                        template_id=request.context.template_id,
                        aesthetic_name=request.context.aesthetic_name,
                        character_name=request.context.character_name,
                    )

                # Build conversation object
                conv = Conversation(
                    id=conversation.id,
                    user_id=request.user_id,
                    job_id=job.id,
                )
                conv.add_message(Message(role=MessageRole.USER, content=request.initial_message))

                # Run Pali
                result = await pali.run(
                    context=agent_context,
                    user_input=request.initial_message,
                    conversation=conv,
                )

                if result.success and result.data:
                    initial_response = result.data.get("message")
                    requirements_status = result.data.get("requirements_status")
                    action = result.data.get("action")

                    # Save assistant response
                    if initial_response:
                        await save_message(conversation.id, "assistant", initial_response)

                    # If requirements are complete, update job status to PLANNING
                    # so /chat/generate can proceed
                    if action == "delegate_to_planner" or result.next_agent == "planner":
                        from prisma import Json
                        update_data = {"status": "PLANNING"}
                        if requirements_status:
                            update_data["requirements"] = Json(requirements_status)
                        await prisma.job.update(
                            where={"id": job.id},
                            data=update_data
                        )
                        logger.info(
                            "Requirements complete, job ready for generation",
                            job_id=job.id,
                            conversation_id=conversation.id,
                        )

        # Determine if ready to generate
        action = None
        ready_to_generate = False
        if request.initial_message:
            # Check if Pali determined requirements are complete
            action = result.data.get("action") if result and result.success and result.data else None
            ready_to_generate = action == "delegate_to_planner" or (result and result.next_agent == "planner")

        return StartConversationResponse(
            conversation_id=conversation.id,
            job_id=job.id,
            status="active",
            initial_response=initial_response,
            requirements_status=requirements_status,
            action=action,
            ready_to_generate=ready_to_generate,
        )

    except Exception as e:
        logger.error(f"Failed to start conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message")
async def send_message(request: SendMessageRequest):
    """
    Send a message in an existing conversation and receive SSE stream response.

    This endpoint processes the user's message with the Pali agent and streams
    back status updates and the assistant's response.

    Args:
        request: Message request with conversation ID and message content

    Returns:
        StreamingResponse with SSE events
    """
    async def event_stream():
        """Generate SSE events for the chat response."""
        try:
            # Get conversation from database
            db_conversation = await get_or_create_conversation(
                request.conversation_id,
                request.user_id
            )

            logger.info(
                "Processing chat message",
                conversation_id=request.conversation_id,
                message_length=len(request.message),
            )

            # Send status: understanding
            yield create_status_event("understanding", progress=0.1)

            # Save user message
            await save_message(request.conversation_id, "user", request.message)

            # Parse selector response if present (e.g., [SELECTOR:task_complexity:standard])
            selector_data = parse_selector_response(request.message)
            if selector_data:
                selector_id, value, field_name = selector_data
                logger.info(
                    "chat.selector_response.parsed",
                    selector_id=selector_id,
                    value=value,
                    field_name=field_name,
                    conversation_id=request.conversation_id,
                )

                # Update job requirements in database
                if db_conversation.jobId:
                    await update_job_requirements(db_conversation.jobId, field_name, value)

            # Extract fields from natural language reply if not a selector response
            extracted_fields = {}
            if not selector_data:
                # Get last assistant message to see what fields were being asked about
                last_metadata = await get_last_assistant_message_metadata(request.conversation_id)
                if last_metadata:
                    missing_fields = last_metadata.get("missing_fields", [])
                    if missing_fields:
                        extracted_fields = extract_fields_from_reply(request.message, missing_fields)
                        # Update job requirements with extracted fields
                        if extracted_fields and db_conversation.jobId:
                            for field_name, value in extracted_fields.items():
                                await update_job_requirements(db_conversation.jobId, field_name, value)
                                logger.info(
                                    "chat.natural_language.field_persisted",
                                    field_name=field_name,
                                    value=value[:50] if len(value) > 50 else value,
                                )

            # Build conversation object from history
            conversation = build_conversation_from_db(db_conversation)
            conversation.add_message(Message(role=MessageRole.USER, content=request.message))

            # Create agent context
            agent_context = AgentContext(
                user_id=request.user_id,
                job_id=db_conversation.jobId,
                conversation_id=request.conversation_id,
            )

            # CRITICAL: Load existing job requirements first
            # This ensures previously extracted requirements (subject, style, etc.) are preserved
            # Otherwise, auto-resume after clarification won't work
            if db_conversation.job and db_conversation.job.requirements:
                existing_reqs = db_conversation.job.requirements
                if isinstance(existing_reqs, str):
                    existing_reqs = json.loads(existing_reqs)
                agent_context.requirements = dict(existing_reqs)
                logger.info(
                    "chat.existing_requirements.loaded",
                    subject=existing_reqs.get("subject"),
                    is_complete=existing_reqs.get("is_complete"),
                )

            # Add generator context to requirements if provided (merge, not replace)
            if request.context:
                if agent_context.requirements is None:
                    agent_context.requirements = {}
                agent_context.requirements.update({
                    "template_id": request.context.template_id,
                    "aesthetic_id": request.context.aesthetic_id,
                    "dimensions": request.context.dimensions,
                    "creativity": request.context.creativity,
                })

            # Add selector response to requirements if parsed
            # Note: Selector is already persisted to DB at line 688
            if selector_data:
                if agent_context.requirements is None:
                    agent_context.requirements = {}
                agent_context.requirements[field_name] = value
                logger.info(
                    "chat.selector_response.added_to_context",
                    field_name=field_name,
                    value=value,
                )

            # Add extracted natural language fields to requirements
            if extracted_fields:
                if agent_context.requirements is None:
                    agent_context.requirements = {}
                for field_name, value in extracted_fields.items():
                    agent_context.requirements[field_name] = value
                    logger.info(
                        "chat.natural_language.added_to_context",
                        field_name=field_name,
                        value=value[:50] if len(value) > 50 else value,
                    )

            # Process with Pali agent
            async with PaliAgent() as pali:
                result = await pali.run(
                    context=agent_context,
                    user_input=request.message,
                    conversation=conversation,
                )

            if not result.success:
                yield create_status_event("error")
                yield create_message_event(
                    "assistant",
                    result.error or "I'm sorry, something went wrong. Could you try again?"
                )
                return

            # Extract response data
            response_message = result.data.get("message", "")
            action = result.data.get("action")

            # CRITICAL FIX: Pali returns different keys based on action:
            # - "requirements_status" when action="request_more_info"
            # - "requirements" when action="delegate_to_planner"
            # Check both keys to get the correct requirements data
            requirements_status = result.data.get("requirements_status") or result.data.get("requirements", {})

            # Check if requirements are complete
            # When action is "delegate_to_planner", requirements are complete by definition
            is_complete = (
                action == "delegate_to_planner" or
                requirements_status.get("is_complete", False)
            )

            # Send requirements status
            yield create_requirements_event(requirements_status, is_complete)

            # Send appropriate status based on action
            if action == "delegate_to_planner":
                yield create_status_event("complete", progress=1.0)
            elif action == "request_more_info":
                yield create_status_event("clarifying", progress=0.5)
            else:
                yield create_status_event("understanding", progress=0.8)

            # Save and send assistant response
            await save_message(request.conversation_id, "assistant", response_message)
            yield create_message_event("assistant", response_message, {
                "requirements_complete": is_complete,
                "action": action,
            })

            # Update job status if requirements complete
            if is_complete and db_conversation.jobId:
                from prisma import Json
                update_data = {"status": "PLANNING"}
                if requirements_status:
                    # CRITICAL: Merge requirements_status with existing job requirements
                    # This preserves selector responses (complexity, character, etc.)
                    # that were saved earlier but aren't in Pali's response
                    job = await prisma.job.find_unique(where={"id": db_conversation.jobId})
                    existing_reqs = {}
                    if job and job.requirements:
                        existing_reqs = job.requirements if isinstance(job.requirements, dict) else json.loads(job.requirements)

                    # Merge: existing first, then requirements_status (so Pali's response takes precedence for design fields)
                    merged_requirements = {**existing_reqs, **requirements_status}
                    update_data["requirements"] = Json(merged_requirements)
                    logger.info(
                        "chat.requirements.saving",
                        job_id=db_conversation.jobId,
                        has_subject=bool(merged_requirements.get("subject")),
                        has_complexity=bool(merged_requirements.get("complexity")),
                        subject=merged_requirements.get("subject", "")[:50] if merged_requirements.get("subject") else None,
                    )
                await prisma.job.update(
                    where={"id": db_conversation.jobId},
                    data=update_data
                )

            # Auto-resume if was awaiting clarification and user provided answer
            logger.info(
                "chat.auto_resume.check",
                job_status=db_conversation.job.status if db_conversation.job else None,
                is_complete=is_complete,
                action=action,
                will_resume=bool(
                    db_conversation.job and
                    db_conversation.job.status == "AWAITING_CLARIFICATION" and
                    (is_complete or action == "delegate_to_planner")
                ),
            )
            if db_conversation.job and db_conversation.job.status == "AWAITING_CLARIFICATION":
                if is_complete or action == "delegate_to_planner":
                    logger.info(
                        "pali.generation.auto_resuming",
                        conversation_id=request.conversation_id,
                    )

                    yield create_message_event(
                        "assistant",
                        "Perfect! Let me continue creating your design...",
                        {"action": "resuming_generation"}
                    )

                    # Resume generation
                    async for gen_event in trigger_generation_internal(
                        conversation_id=request.conversation_id,
                        user_id=request.user_id,
                        job_id=db_conversation.jobId,
                    ):
                        yield gen_event
                    return

            logger.info(
                "Chat message processed",
                conversation_id=request.conversation_id,
                requirements_complete=is_complete,
                action=action,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing chat message: {e}", exc_info=True)
            yield create_status_event("error")
            yield create_message_event(
                "assistant",
                "I'm sorry, something went wrong. Could you try again?"
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/history/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: str,
    user_id: str = Query(..., description="User ID for authorization"),
) -> ConversationHistoryResponse:
    """
    Get the history of a conversation.

    Args:
        conversation_id: ID of the conversation
        user_id: User ID for authorization

    Returns:
        Conversation history with all messages
    """
    try:
        db_conversation = await get_or_create_conversation(conversation_id, user_id)

        messages = []
        for msg in db_conversation.messages:
            messages.append(ChatMessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                created_at=msg.createdAt.isoformat(),
                metadata=json.loads(msg.metadata) if msg.metadata else None,
            ))

        # Get requirements status from job if available
        requirements_status = None
        if db_conversation.job and db_conversation.job.requirements:
            requirements_status = json.loads(db_conversation.job.requirements) if isinstance(db_conversation.job.requirements, str) else db_conversation.job.requirements

        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            job_id=db_conversation.jobId,
            status=db_conversation.status,
            messages=messages,
            requirements_status=requirements_status,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def trigger_generation(request: TriggerGenerationRequest):
    """
    Trigger the full generation pipeline after requirements are gathered.

    NEW ARCHITECTURE:
    - Pali is always on as the communication layer
    - Pali delegates to Planner for inline orchestration
    - Planner stays in control until generation completes

    Flow: Endpoint → Pali → Planner (inline) → Result

    Args:
        request: Generation request with conversation ID and user info

    Returns:
        StreamingResponse with SSE events for generation progress
    """
    async def event_stream():
        """Generate SSE events for the generation pipeline."""
        from palet8_agents.agents.pali_agent import PaliAgent
        from palet8_agents.core.agent import AgentContext as PaletAgentContext

        task_id = str(uuid4())

        try:
            # Get conversation and verify requirements
            db_conversation = await get_or_create_conversation(
                request.conversation_id,
                request.user_id
            )

            if not db_conversation.job:
                yield f"data: {json.dumps({'type': 'error', 'error': 'No job associated with conversation'})}\n\n"
                return

            job = db_conversation.job

            # Check if requirements are ready
            if job.status not in ["PLANNING", "GENERATING"]:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Requirements not yet complete. Please continue chatting.'})}\n\n"
                return

            # Log incoming request context for debugging
            logger.info(
                "chat.generate.request_received",
                conversation_id=request.conversation_id,
                job_id=job.id,
                task_id=task_id,
                has_context=request.context is not None,
                context_text_prompt=request.context.text_prompt if request.context else None,
                context_aesthetic_name=request.context.aesthetic_name if request.context else None,
                context_character_name=request.context.character_name if request.context else None,
                context_template_category=request.context.template_category if request.context else None,
                context_dimensions=request.context.dimensions if request.context else None,
            )

            logger.info(
                "Starting generation from chat",
                conversation_id=request.conversation_id,
                job_id=job.id,
                task_id=task_id,
            )

            # Send started event
            yield f"data: {json.dumps({'type': 'started', 'task_id': task_id, 'job_id': job.id})}\n\n"

            # Update job status
            await prisma.job.update(
                where={"id": job.id},
                data={"status": "GENERATING"}
            )

            # Build requirements from job and context
            requirements = json.loads(job.requirements) if isinstance(job.requirements, str) else (job.requirements or {})

            # Log job requirements before enrichment
            logger.info(
                "chat.generate.job_requirements_loaded",
                job_id=job.id,
                job_req_keys=list(requirements.keys()) if requirements else [],
                job_req_subject=requirements.get("subject", "")[:100] if requirements.get("subject") else None,
                job_req_style=requirements.get("style"),
                job_req_product_type=requirements.get("product_type"),
            )

            # Enrich requirements with context from frontend
            if request.context:
                if request.context.text_prompt:
                    requirements["subject"] = request.context.text_prompt
                if request.context.aesthetic_name:
                    requirements["style"] = request.context.aesthetic_name
                if request.context.character_name:
                    requirements["character"] = request.context.character_name
                    # Integrate character into subject
                    current_subject = requirements.get("subject", "")
                    if current_subject:
                        # Add character to subject if not already mentioned
                        if request.context.character_name.lower() not in current_subject.lower():
                            requirements["subject"] = f"{current_subject} featuring {request.context.character_name}"
                    else:
                        # No subject but has character - use character as subject base
                        requirements["subject"] = f"design featuring {request.context.character_name}"
                if request.context.template_id:
                    requirements["template_id"] = request.context.template_id
                if request.context.template_category:
                    requirements["product_type"] = request.context.template_category
                if request.context.dimensions:
                    requirements["dimensions"] = request.context.dimensions
                    parts = request.context.dimensions.split("x")
                    if len(parts) == 2:
                        requirements["width"] = int(parts[0])
                        requirements["height"] = int(parts[1])

            # Persist enriched requirements to Job for auto-resume scenarios
            from prisma import Json as PrismaJson
            await prisma.job.update(
                where={"id": job.id},
                data={"requirements": PrismaJson(requirements)}
            )
            logger.info(
                "job.requirements.persisted",
                job_id=job.id,
                keys=list(requirements.keys()),
                has_subject=bool(requirements.get("subject")),
            )

            # Validate we have enough to proceed
            if not requirements.get("subject"):
                yield create_status_event("error")
                yield f"data: {json.dumps({'type': 'error', 'task_id': task_id, 'error': 'Please provide a prompt describing what you want to create'})}\n\n"
                return

            logger.info(
                "Building generation request",
                subject=requirements.get("subject", "")[:100],
                has_style=bool(requirements.get("style")),
            )

            # Create agent context
            agent_context = PaletAgentContext(
                user_id=request.user_id,
                job_id=job.id,
                conversation_id=request.conversation_id,
            )
            agent_context.requirements = requirements

            # ================================================================
            # NEW FLOW: Pali is always on, delegates to Planner
            # ================================================================

            async with PaliAgent() as pali:
                # Pali handles the generation, delegates to Planner internally
                async for event in pali.handle_generate_request(
                    context=agent_context,
                    requirements=requirements,
                ):
                    event_type = event.get("type")

                    if event_type == "status":
                        # Forward progress status
                        yield create_status_event(
                            event.get("stage", "working"),
                            progress=event.get("progress", 0.5),
                        )

                    elif event_type == "result":
                        # Generation complete - send result
                        yield create_status_event("complete", progress=1.0)

                        images = event.get("images", [])
                        best_image = images[0] if images else {}
                        best_image["score"] = 0.9
                        best_image["approved"] = True

                        # Update job with results
                        await prisma.job.update(
                            where={"id": job.id},
                            data={
                                "status": "COMPLETED",
                                "evaluation": json.dumps([{"score": 0.9}]),
                                "creditCost": 1,
                            }
                        )

                        # Update conversation status
                        await prisma.conversation.update(
                            where={"id": request.conversation_id},
                            data={"status": "completed"}
                        )

                        # Send final result
                        final_result = {
                            "type": "complete",
                            "task_id": task_id,
                            "success": True,
                            "best_image": best_image,
                            "products": [],
                        }
                        yield f"data: {json.dumps(final_result)}\n\n"

                        logger.info(
                            "Generation completed from chat",
                            conversation_id=request.conversation_id,
                            task_id=task_id,
                            success=True,
                        )

                    elif event_type == "message" and event.get("metadata", {}).get("requires_input"):
                        # Planner needs more info - send clarification as message event
                        metadata = event.get("metadata", {})
                        message_type = metadata.get("message_type", "general")

                        # Save clarification message
                        await save_message(request.conversation_id, "assistant", event.get("content", ""), {
                            "message_type": message_type,
                            "selector_id": metadata.get("selector_id"),
                            "missing_fields": metadata.get("missing_fields", []),
                        })

                        # Update job status to awaiting clarification
                        await prisma.job.update(
                            where={"id": job.id},
                            data={"status": "AWAITING_CLARIFICATION"}
                        )

                        # Send message event (frontend can handle this)
                        yield create_message_event("assistant", event.get("content", ""), {
                            "requires_input": True,
                            "message_type": message_type,
                            "selector_id": metadata.get("selector_id"),
                            "missing_fields": metadata.get("missing_fields", []),
                        })

                        logger.info(
                            "pali.clarification.sent",
                            conversation_id=request.conversation_id,
                            message_type=message_type,
                            selector_id=metadata.get("selector_id"),
                        )

                    elif event_type == "awaiting_confirmation":
                        # Result presented, waiting for user confirmation
                        # This is handled by the result event above
                        pass

                    elif event_type == "error":
                        # Error from Pali/Planner
                        yield create_status_event("error")
                        yield f"data: {json.dumps({'type': 'error', 'task_id': task_id, 'error': event.get('error', 'Generation failed')})}\n\n"

                        await prisma.job.update(
                            where={"id": job.id},
                            data={"status": "FAILED", "metadata": json.dumps({"error": event.get("error")})}
                        )

        except HTTPException as e:
            yield f"data: {json.dumps({'type': 'error', 'error': e.detail})}\n\n"
        except Exception as e:
            logger.error(f"Generation pipeline error: {e}", exc_info=True)
            yield create_status_event("error")
            yield f"data: {json.dumps({'type': 'error', 'task_id': task_id, 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/edit")
async def request_edit(request: SendMessageRequest):
    """
    Request an edit to a previously generated design.

    This routes the edit request back through Pali agent to gather
    edit requirements, then triggers a new generation cycle.

    Args:
        request: Message request containing the edit instructions

    Returns:
        StreamingResponse with SSE events for the edit flow
    """
    # Edit requests go through the same message flow
    # Pali will understand it's an edit based on context
    return await send_message(request)


@router.post("/stop/{conversation_id}")
async def stop_generation(
    conversation_id: str,
    user_id: str = Query(..., description="User ID for authorization"),
):
    """
    Stop an in-progress generation.

    Cancels the current generation job but keeps the conversation intact,
    allowing the user to restart or modify their request.

    Args:
        conversation_id: ID of the conversation
        user_id: User ID for authorization

    Returns:
        JSON with success status and confirmation message
    """
    try:
        db_conversation = await get_or_create_conversation(conversation_id, user_id)

        if not db_conversation.job:
            raise HTTPException(status_code=404, detail="No job found for this conversation")

        job = db_conversation.job

        # Check if there's an active generation to stop
        if job.status not in ["GENERATING", "PLANNING", "AWAITING_CLARIFICATION"]:
            return {
                "success": True,
                "message": "No active generation to stop",
                "job_status": job.status,
            }

        # Update job status to cancelled
        await prisma.job.update(
            where={"id": job.id},
            data={"status": "CANCELLED"}
        )

        # Generate stop confirmation via Pali
        async with PaliAgent() as pali:
            agent_context = AgentContext(
                user_id=user_id,
                job_id=job.id,
                conversation_id=conversation_id,
            )
            result = await pali.handle_stop_request(agent_context)

        # Save the stop confirmation message
        message = result.get("message", "Generation stopped. Your conversation is saved.")
        await save_message(conversation_id, "assistant", message)

        logger.info(
            "pali.generation.stopped",
            conversation_id=conversation_id,
            job_id=job.id,
        )

        return {
            "success": True,
            "message": message,
            "job_status": "CANCELLED",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
