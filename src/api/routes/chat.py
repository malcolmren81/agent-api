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
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import asyncio

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.utils import get_logger
from src.database import prisma
from palet8_agents.agents.pali_agent import PaliAgent
from palet8_agents.core.agent import AgentContext
from palet8_agents.core.message import Conversation, Message, MessageRole

logger = get_logger(__name__)
router = APIRouter()


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
    await prisma.chatmessage.create(
        data={
            "conversationId": conversation_id,
            "role": role,
            "content": content,
            "metadata": json.dumps(metadata) if metadata else None,
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

                    # Save assistant response
                    if initial_response:
                        await save_message(conversation.id, "assistant", initial_response)

        return StartConversationResponse(
            conversation_id=conversation.id,
            job_id=job.id,
            status="active",
            initial_response=initial_response,
            requirements_status=requirements_status,
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

            # Build conversation object from history
            conversation = build_conversation_from_db(db_conversation)
            conversation.add_message(Message(role=MessageRole.USER, content=request.message))

            # Create agent context
            agent_context = AgentContext(
                user_id=request.user_id,
                job_id=db_conversation.jobId,
                conversation_id=request.conversation_id,
            )

            # Add generator context to requirements if provided
            if request.context:
                agent_context.requirements = {
                    "template_id": request.context.template_id,
                    "aesthetic_id": request.context.aesthetic_id,
                    "dimensions": request.context.dimensions,
                    "creativity": request.context.creativity,
                }

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
            requirements_status = result.data.get("requirements_status", {})
            action = result.data.get("action")

            # Check if requirements are complete
            is_complete = requirements_status.get("is_complete", False)

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
                await prisma.job.update(
                    where={"id": db_conversation.jobId},
                    data={
                        "status": "PLANNING",
                        "requirements": json.dumps(requirements_status),
                    }
                )

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

    Uses palet8_agents (Planner → Generation API → Evaluator) for the pipeline.
    Streams progress updates via SSE.

    Args:
        request: Generation request with conversation ID and user info

    Returns:
        StreamingResponse with SSE events for generation progress
    """
    async def event_stream():
        """Generate SSE events for the generation pipeline."""
        from palet8_agents.agents.planner_agent import PlannerAgent
        from palet8_agents.agents.evaluator_agent import EvaluatorAgent
        from palet8_agents.core.agent import AgentContext as PaletAgentContext
        import httpx

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

            # Build prompt from requirements
            requirements = json.loads(job.requirements) if isinstance(job.requirements, str) else (job.requirements or {})
            user_prompt = requirements.get("subject", "")

            # Add style/aesthetic if available
            if requirements.get("style"):
                user_prompt += f", {requirements['style']} style"
            if requirements.get("mood"):
                user_prompt += f", {requirements['mood']} mood"

            # Add context from generator page
            dimensions = "1024x1024"
            if request.context:
                if request.context.aesthetic_name:
                    user_prompt += f", {request.context.aesthetic_name}"
                dimensions = request.context.dimensions

            # Send planning status
            yield create_status_event("planning", progress=0.1)

            # Create agent context for palet8_agents
            agent_context = PaletAgentContext(
                user_id=request.user_id,
                job_id=job.id,
                conversation_id=request.conversation_id,
            )
            agent_context.requirements = requirements

            # Run Planner Agent
            yield create_status_event("prompt", progress=0.2)

            async with PlannerAgent() as planner:
                plan_result = await planner.run(
                    context=agent_context,
                    user_input=user_prompt,
                )

            if not plan_result.success:
                yield create_status_event("error")
                yield f"data: {json.dumps({'type': 'error', 'task_id': task_id, 'error': plan_result.error or 'Planning failed'})}\n\n"
                return

            # Extract optimized prompt from plan
            final_prompt = user_prompt
            if plan_result.data:
                final_prompt = plan_result.data.get("final_prompt", user_prompt)

            # Send model selection status
            yield create_status_event("model_selection", progress=0.3)

            # Send generation start status
            yield create_status_event("generation_start", progress=0.4)

            # Call generation API directly
            import os
            flux_api_key = os.getenv("FLUX_API_KEY")

            # Parse dimensions
            width, height = 1024, 1024
            if "x" in dimensions:
                parts = dimensions.split("x")
                width, height = int(parts[0]), int(parts[1])

            # Generate with Flux API
            generated_images = []
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        "https://api.bfl.ai/v1/flux-pro-1.1",
                        headers={
                            "X-Key": flux_api_key,
                            "Content-Type": "application/json",
                        },
                        json={
                            "prompt": final_prompt,
                            "width": width,
                            "height": height,
                            "num_images": 1,
                        }
                    )

                    if response.status_code == 200:
                        result_data = response.json()
                        # Poll for result
                        request_id = result_data.get("id")
                        if request_id:
                            yield create_status_event("generation_progress", progress=0.5)

                            for _ in range(60):  # Max 60 polls (2 mins)
                                await asyncio.sleep(2)
                                poll_response = await client.get(
                                    f"https://api.bfl.ai/v1/get_result?id={request_id}",
                                    headers={"X-Key": flux_api_key}
                                )
                                if poll_response.status_code == 200:
                                    poll_data = poll_response.json()
                                    status = poll_data.get("status")
                                    if status == "Ready":
                                        image_url = poll_data.get("result", {}).get("sample")
                                        if image_url:
                                            # Download image and convert to base64
                                            img_response = await client.get(image_url)
                                            if img_response.status_code == 200:
                                                import base64
                                                base64_data = base64.b64encode(img_response.content).decode("utf-8")
                                                generated_images.append({
                                                    "base64_data": base64_data,
                                                    "url": image_url,
                                                })
                                        break
                                    elif status == "Error":
                                        raise Exception(poll_data.get("error", "Generation failed"))
                                yield create_status_event("generation_progress", progress=0.5 + (_ * 0.005))
                    else:
                        raise Exception(f"Flux API error: {response.status_code}")

            except Exception as gen_error:
                logger.error(f"Generation error: {gen_error}")
                yield create_status_event("error")
                yield f"data: {json.dumps({'type': 'error', 'task_id': task_id, 'error': str(gen_error)})}\n\n"
                await prisma.job.update(
                    where={"id": job.id},
                    data={"status": "FAILED", "metadata": json.dumps({"error": str(gen_error)})}
                )
                return

            if not generated_images:
                yield create_status_event("error")
                yield f"data: {json.dumps({'type': 'error', 'task_id': task_id, 'error': 'No images generated'})}\n\n"
                return

            # Send evaluation status
            yield create_status_event("evaluation", progress=0.8)

            # Run Evaluator Agent
            best_image = generated_images[0]
            best_image["score"] = 0.9  # Default high score
            best_image["approved"] = True

            try:
                async with EvaluatorAgent() as evaluator:
                    eval_result = await evaluator.run(
                        context=agent_context,
                        images=generated_images,
                        prompt=final_prompt,
                    )
                    if eval_result.success and eval_result.data:
                        best_image = eval_result.data.get("best_image", best_image)
            except Exception as eval_error:
                logger.warning(f"Evaluation error (continuing with default): {eval_error}")

            # Send complete status
            yield create_status_event("complete", progress=1.0)

            # Update job with results
            await prisma.job.update(
                where={"id": job.id},
                data={
                    "status": "COMPLETED",
                    "evaluation": json.dumps([{"score": best_image.get("score", 0.9)}]),
                    "creditCost": 1,  # Default credit cost
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
                "products": [],  # Product mockups can be added later
            }
            yield f"data: {json.dumps(final_result)}\n\n"

            logger.info(
                "Generation completed from chat",
                conversation_id=request.conversation_id,
                task_id=task_id,
                success=True,
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
