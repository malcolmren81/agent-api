"""
Script to insert test AgentLog records directly into the database.
This verifies that:
1. We're writing to the correct database
2. The table structure is correct
3. The admin panel can read the data
"""
import asyncio
import json
from datetime import datetime
from prisma import Prisma

async def insert_test_logs():
    """Insert test AgentLog records."""
    prisma = Prisma()
    await prisma.connect()

    print("✅ Connected to database")

    # Test task ID
    test_task_id = "test-task-12345"
    test_shop = "test-shop.myshopify.com"

    # Test data for different agents
    test_logs = [
        {
            "shop": test_shop,
            "taskId": test_task_id,
            "agentName": "interactive",
            "input": json.dumps({"prompt": "Test product image"}),
            "output": json.dumps({"success": True, "validated": True}),
            "reasoning": "Test reasoning for interactive agent",
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
            "input": json.dumps({"context": {"user_prompt": "Test product"}}),
            "output": json.dumps({"success": True, "plan": "simple"}),
            "reasoning": "Test reasoning for planner agent",
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
            "agentName": "prompt_manager",
            "input": json.dumps({"user_input": "test"}),
            "output": json.dumps({"success": True, "prompt": "Professional product photography"}),
            "reasoning": "Test reasoning for prompt manager",
            "executionTime": 180,
            "status": "success",
            "routingMode": "database",
            "usedLlm": False,
            "confidence": 0.88,
            "fallbackUsed": False,
            "creditsUsed": 0,
        },
        {
            "shop": test_shop,
            "taskId": test_task_id,
            "agentName": "model_selection",
            "input": json.dumps({"bucket": "product:realistic:white-bg"}),
            "output": json.dumps({"success": True, "selected_model": "flux-kontext-pro"}),
            "reasoning": "Test reasoning for model selection",
            "executionTime": 120,
            "status": "success",
            "routingMode": "ucb1",
            "usedLlm": False,
            "confidence": 0.92,
            "fallbackUsed": False,
            "creditsUsed": 0,
        },
        {
            "shop": test_shop,
            "taskId": test_task_id,
            "agentName": "generation",
            "input": json.dumps({"prompt": "Professional product photography", "model": "flux-kontext-pro"}),
            "output": json.dumps({"success": True, "num_images": 2}),
            "reasoning": "Test reasoning for generation agent",
            "executionTime": 5000,
            "status": "success",
            "routingMode": "api",
            "usedLlm": False,
            "confidence": None,
            "fallbackUsed": False,
            "creditsUsed": 25,
        },
        {
            "shop": test_shop,
            "taskId": test_task_id,
            "agentName": "evaluation",
            "input": json.dumps({"num_images": 2}),
            "output": json.dumps({"success": True, "approved": True, "score": 0.78}),
            "reasoning": "Test reasoning for evaluation agent",
            "executionTime": 450,
            "status": "success",
            "routingMode": "vision",
            "usedLlm": True,
            "confidence": 0.78,
            "fallbackUsed": False,
            "creditsUsed": 2,
        },
        {
            "shop": test_shop,
            "taskId": test_task_id,
            "agentName": "product_generator",
            "input": json.dumps({"image_url": "test.png"}),
            "output": json.dumps({"success": True, "mockups": ["tshirt.png", "mug.png"]}),
            "reasoning": "Test reasoning for product generator",
            "executionTime": 800,
            "status": "success",
            "routingMode": "local",
            "usedLlm": False,
            "confidence": None,
            "fallbackUsed": False,
            "creditsUsed": 0,
        },
    ]

    # Insert each test log
    inserted_ids = []
    for log_data in test_logs:
        try:
            result = await prisma.agentlog.create(data=log_data)
            inserted_ids.append(result.id)
            print(f"✅ Inserted {log_data['agentName']} log - ID: {result.id}")
        except Exception as e:
            print(f"❌ Failed to insert {log_data['agentName']}: {e}")
            continue

    print(f"\n✅ Successfully inserted {len(inserted_ids)} test AgentLog records")
    print(f"📋 Test task_id: {test_task_id}")
    print(f"🏪 Test shop: {test_shop}")

    # Verify the records exist by querying
    print("\n🔍 Verifying records exist...")
    logs = await prisma.agentlog.find_many(
        where={"taskId": test_task_id},
        order={"createdAt": "asc"}
    )

    print(f"✅ Found {len(logs)} records with task_id={test_task_id}")
    for log in logs:
        print(f"  - {log.agentName}: {log.status} ({log.executionTime}ms)")

    await prisma.disconnect()
    print("\n✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(insert_test_logs())
