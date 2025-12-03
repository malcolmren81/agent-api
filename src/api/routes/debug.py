"""
Debug routes for testing and diagnostics.
"""
import json
from fastapi import APIRouter
from src.database.client import prisma
from src.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/insert-test-logs")
async def insert_test_logs():
    """
    Insert test AgentLog records for verification.

    This endpoint helps verify:
    1. Database connection is working
    2. AgentLog table exists and has correct structure
    3. Admin panel can read the inserted data
    """
    try:

        # Test task ID
        test_task_id = "test-task-debug-67890"
        test_shop = "debug-shop.myshopify.com"

        # Test data for different agents
        test_logs = [
            {
                "shop": test_shop,
                "taskId": test_task_id,
                "agentName": "interactive",
                "input": json.dumps({"prompt": "Debug test product image"}),
                "output": json.dumps({"success": True, "validated": True}),
                "reasoning": "Debug test reasoning for interactive agent",
                "executionTime": 150,
                "status": "success",
                "routingMode": "rule",
                "usedLlm": False,
                "confidence": 0.95,
                "fallbackUsed": False,
                "creditsUsed": 0,
            },
            {
                "shop": test_shop,
                "taskId": test_task_id,
                "agentName": "planner",
                "input": json.dumps({"context": {"user_prompt": "Debug test"}}),
                "output": json.dumps({"success": True, "plan": "simple"}),
                "reasoning": "Debug test reasoning for planner",
                "executionTime": 200,
                "status": "success",
                "routingMode": "rule",
                "usedLlm": False,
                "confidence": 0.90,
                "fallbackUsed": False,
                "creditsUsed": 0,
            },
            {
                "shop": test_shop,
                "taskId": test_task_id,
                "agentName": "evaluation",
                "input": json.dumps({"num_images": 2}),
                "output": json.dumps({"success": True, "approved": True, "score": 0.85}),
                "reasoning": "Debug test - image passed evaluation",
                "executionTime": 450,
                "status": "success",
                "routingMode": "vision",
                "usedLlm": True,
                "confidence": 0.85,
                "fallbackUsed": False,
                "creditsUsed": 2,
            },
        ]

        # Insert each test log
        inserted_records = []
        for log_data in test_logs:
            result = await prisma.agentlog.create(data=log_data)
            inserted_records.append({
                "id": result.id,
                "agentName": result.agentName,
                "taskId": result.taskId,
                "createdAt": result.createdAt.isoformat()
            })
            logger.info(f"Inserted test log: {result.agentName} - ID: {result.id}")

        # Verify by querying back
        verify_logs = await prisma.agentlog.find_many(
            where={"taskId": test_task_id},
            order={"createdAt": "asc"}
        )

        # ALSO INSERT A TASK RECORD so it shows in the list page
        task_record = await prisma.task.create(
            data={
                "taskId": test_task_id,
                "shop": test_shop,
                "originalPrompt": "Debug test: Verify AgentLog and Task display",
                "userRequest": json.dumps({"prompt": "Debug test", "test": True}),
                "stages": json.dumps([
                    {"agent": "interactive", "status": "success", "duration": 150},
                    {"agent": "planner", "status": "success", "duration": 200},
                    {"agent": "evaluation", "status": "success", "duration": 450}
                ]),
                "promptJourney": json.dumps([
                    {"stage": "initial", "prompt": "Debug test prompt"},
                    {"stage": "final", "prompt": "Professional product photography"}
                ]),
                "totalDuration": 800,
                "creditsCost": 2,
                "performanceBreakdown": json.dumps({
                    "totalDuration": 800,
                    "byStage": [
                        {"stage": "interactive", "duration": 150, "credits": 0, "percentage": 18.75},
                        {"stage": "planner", "duration": 200, "credits": 0, "percentage": 25.0},
                        {"stage": "evaluation", "duration": 450, "credits": 2, "percentage": 56.25}
                    ],
                    "bottlenecks": ["evaluation"]
                }),
                "evaluationResults": json.dumps({
                    "score": 0.85,
                    "approved": True
                }),
                "generatedImageUrl": "https://example.com/test-image.png",
                "mockupUrls": json.dumps(["https://example.com/tshirt.png", "https://example.com/mug.png"]),
                "finalPrompt": "Professional product photography on white background",
                "status": "completed"
            }
        )

        logger.info(f"Inserted Task record: {task_record.id}")

        return {
            "success": True,
            "message": f"Successfully inserted {len(inserted_records)} AgentLog records + 1 Task record",
            "test_task_id": test_task_id,
            "test_shop": test_shop,
            "task_record_id": task_record.id,
            "inserted_agent_logs": inserted_records,
            "verification": {
                "agent_logs_count": len(verify_logs),
                "agent_logs": [
                    {
                        "id": log.id,
                        "agentName": log.agentName,
                        "status": log.status,
                        "executionTime": log.executionTime
                    }
                    for log in verify_logs
                ],
                "task_record": {
                    "id": task_record.id,
                    "taskId": task_record.taskId,
                    "status": task_record.status,
                    "totalDuration": task_record.totalDuration
                }
            }
        }

    except Exception as e:
        logger.error(f"Failed to insert test logs: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@router.get("/query-logs/{task_id}")
async def query_logs(task_id: str):
    """Query AgentLog records by task_id for debugging."""
    try:

        logs = await prisma.agentlog.find_many(
            where={"taskId": task_id},
            order={"createdAt": "asc"}
        )

        return {
            "success": True,
            "task_id": task_id,
            "count": len(logs),
            "logs": [
                {
                    "id": log.id,
                    "agentName": log.agentName,
                    "status": log.status,
                    "executionTime": log.executionTime,
                    "routingMode": log.routingMode,
                    "usedLlm": log.usedLlm,
                    "creditsUsed": log.creditsUsed,
                    "createdAt": log.createdAt.isoformat()
                }
                for log in logs
            ]
        }

    except Exception as e:
        logger.error(f"Failed to query logs: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


@router.delete("/delete-test-data/{task_id}")
async def delete_test_data(task_id: str):
    """Delete test data by task_id."""
    try:
        # Delete AgentLog records
        deleted_logs_count = await prisma.agentlog.delete_many(
            where={"taskId": task_id}
        )

        # Delete Task record
        deleted_task_count = await prisma.task.delete_many(
            where={"taskId": task_id}
        )

        logger.info(f"Deleted test data: {deleted_logs_count} AgentLogs, {deleted_task_count} Tasks")

        return {
            "success": True,
            "task_id": task_id,
            "deleted_agent_logs": deleted_logs_count,
            "deleted_tasks": deleted_task_count
        }

    except Exception as e:
        logger.error(f"Failed to delete test data: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }
