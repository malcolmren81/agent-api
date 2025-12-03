"""
WebSocket manager for real-time workflow updates.

Provides Server-Sent Events (SSE) for streaming workflow status
to connected clients.
"""
from typing import Dict, Set
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.utils import get_logger
import asyncio
import json

logger = get_logger(__name__)
router = APIRouter()

# Global registry of active workflow subscriptions
# Format: {task_id: set(client_generators)}
workflow_subscribers: Dict[str, Set] = {}


class WorkflowEventManager:
    """Manages workflow event broadcasting to subscribers"""

    def __init__(self):
        self.subscribers: Dict[str, Set] = {}

    def subscribe(self, task_id: str) -> asyncio.Queue:
        """Subscribe to workflow updates for a task"""
        queue = asyncio.Queue()

        if task_id not in self.subscribers:
            self.subscribers[task_id] = set()

        self.subscribers[task_id].add(queue)
        logger.info(f"New subscriber for task {task_id}")

        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue):
        """Unsubscribe from workflow updates"""
        if task_id in self.subscribers:
            self.subscribers[task_id].discard(queue)
            if not self.subscribers[task_id]:
                del self.subscribers[task_id]
            logger.info(f"Unsubscribed from task {task_id}")

    async def broadcast(self, task_id: str, event_data: dict):
        """Broadcast event to all subscribers of a task"""
        if task_id not in self.subscribers:
            return

        dead_queues = set()

        for queue in self.subscribers[task_id]:
            try:
                await queue.put(event_data)
            except Exception as e:
                logger.error(f"Failed to broadcast to subscriber: {e}")
                dead_queues.add(queue)

        # Clean up dead subscribers
        for queue in dead_queues:
            self.subscribers[task_id].discard(queue)


# Global event manager
event_manager = WorkflowEventManager()


@router.get("/workflow/stream/{task_id}")
async def stream_workflow(task_id: str):
    """
    Server-Sent Events endpoint for real-time workflow updates.

    Clients connect to this endpoint and receive events as agents
    complete their executions.

    Args:
        task_id: Task ID to stream updates for

    Returns:
        StreamingResponse with SSE events
    """
    async def event_generator():
        """Generate SSE events for workflow updates"""
        queue = event_manager.subscribe(task_id)

        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'taskId': task_id})}\n\n"

            # Stream updates until client disconnects or workflow completes
            while True:
                try:
                    # Wait for next event with timeout
                    event_data = await asyncio.wait_for(queue.get(), timeout=30.0)

                    # Format as SSE event
                    yield f"data: {json.dumps(event_data)}\n\n"

                    # If workflow is complete, end stream
                    if event_data.get('type') == 'workflow_complete':
                        logger.info(f"Workflow {task_id} complete, ending stream")
                        break

                except asyncio.TimeoutError:
                    # Send keepalive ping
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    continue

        except asyncio.CancelledError:
            logger.info(f"Client disconnected from task {task_id}")
        finally:
            event_manager.unsubscribe(task_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


async def notify_agent_update(task_id: str, agent_name: str, agent_data: dict):
    """
    Notify subscribers of agent execution update.

    This should be called from the orchestrator when an agent completes.

    Args:
        task_id: Task ID
        agent_name: Name of agent that updated
        agent_data: Agent execution data
    """
    event = {
        "type": "agent_update",
        "taskId": task_id,
        "agentName": agent_name,
        "data": agent_data,
    }

    await event_manager.broadcast(task_id, event)
    logger.info(f"Broadcasted agent update for {agent_name} on task {task_id}")


async def notify_workflow_complete(task_id: str, final_status: str, result_data: dict = None):
    """
    Notify subscribers that workflow is complete with full result data.

    Args:
        task_id: Task ID
        final_status: Final workflow status
        result_data: Full orchestrator result including best_image, products, credits, etc.
    """
    event = {
        "type": "workflow_complete",  # Progress tracking stream expects "workflow_complete"
        "taskId": task_id,
        "status": final_status,
    }

    # Include result data if provided (contains best_image, products, etc.)
    if result_data:
        event["result"] = result_data

    await event_manager.broadcast(task_id, event)
    logger.info(f"Broadcasted workflow complete for task {task_id} with result data: {bool(result_data)}")
