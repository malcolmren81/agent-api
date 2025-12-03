"""
Task Aggregator Service.

Aggregates completed AgentLog records into comprehensive Task records.
This is a POST-PROCESSING service that runs AFTER workflow completion.
It does NOT interfere with agent execution timing.
"""
from typing import Optional, Dict, List, Any
from datetime import datetime
from src.database import prisma
from src.utils import get_logger

logger = get_logger(__name__)


class TaskAggregator:
    """
    Aggregates completed AgentLog records into comprehensive Task records.

    This service:
    - Runs ONLY after workflow completion (post-processing)
    - Reads AgentLog records without modifying them
    - Computes aggregated metrics from completed logs
    - Creates new Task records for comprehensive logging
    - Does NOT interfere with execution timing
    """

    async def create_task_from_logs(
        self,
        task_id: str,
        shop: str = "default.myshopify.com",
        original_prompt: str = "",
        actual_credit_cost: Optional[int] = None,
        generated_image_url: Optional[str] = None,
        mockup_urls: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create comprehensive Task record from AgentLog records.

        This is a post-processing function called AFTER workflow completes.
        It aggregates data from completed AgentLog records.

        Args:
            task_id: Task identifier
            shop: Shop domain
            original_prompt: User's original prompt
            actual_credit_cost: Actual credit cost from orchestrator (overrides calculated sum)
            generated_image_url: URL of final generated image
            mockup_urls: List of product mockup URLs

        Returns:
            Created task record or None if failed
        """
        try:
            # Use print() for Cloud Run visibility
            print(f"=" * 80)
            print(f"🔍 TASK AGGREGATOR: Starting for task_id={task_id}")
            print(f"=" * 80)
            logger.info(f"Starting task aggregation for task_id={task_id}")

            # Step 1: Fetch all completed AgentLog records for this task with retry logic
            # This is a READ-ONLY operation - does not modify logs
            # Retry with exponential backoff to handle database transaction timing issues
            import asyncio

            agent_logs = None
            max_retries = 3
            retry_delays = [0.5, 1.0, 2.0]  # Exponential backoff in seconds

            for attempt in range(max_retries):
                print(f"🔍 Querying AgentLog records for task_id={task_id} (attempt {attempt + 1}/{max_retries})")
                agent_logs = await prisma.agentlog.find_many(
                    where={"taskId": task_id},
                    order={"createdAt": "asc"}
                )

                if agent_logs:
                    print(f"✅ Found {len(agent_logs)} agent logs for task_id={task_id} on attempt {attempt + 1}")
                    logger.info(f"Found {len(agent_logs)} agent logs for task_id={task_id} on attempt {attempt + 1}")
                    break
                elif attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    print(f"⚠️ No agent logs found on attempt {attempt + 1}, retrying in {delay}s...")
                    logger.warning(f"No agent logs found for task_id={task_id} on attempt {attempt + 1}, retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ No agent logs found after {max_retries} attempts for task_id={task_id}")
                    logger.error(f"No agent logs found for task_id={task_id} after {max_retries} attempts")
                    return None

            # Step 2: Aggregate data (read-only, no modifications to logs)
            print(f"🔍 Building aggregated data structures...")
            stages = self._build_stage_results(agent_logs)
            print(f"  - stages: {len(stages)} items")
            prompt_journey = self._extract_prompt_journey(agent_logs)
            print(f"  - prompt_journey: {len(prompt_journey)} items")
            performance = self._build_performance_breakdown(agent_logs)
            print(f"  - performance: computed")
            evaluation = self._extract_evaluation_results(agent_logs)
            print(f"  - evaluation: {evaluation}")
            final_prompt = self._extract_final_prompt(agent_logs)
            print(f"  - final_prompt: {final_prompt[:50] if final_prompt else 'None'}...")

            # Step 3: Compute totals from existing durations (preserves original timing)
            total_duration = sum(log.executionTime for log in agent_logs)
            # Use actual credit cost from orchestrator if provided, otherwise sum from logs
            total_credits = actual_credit_cost if actual_credit_cost is not None else sum(log.creditsUsed for log in agent_logs)
            print(f"🔍 Totals: duration={total_duration}ms, credits={total_credits} {'(from orchestrator)' if actual_credit_cost is not None else '(summed from logs)'}")

            # Step 4: Determine status
            # Smart status determination: check if workflow actually produced results
            # Don't fail the task if one attempt failed but retries succeeded
            status = "completed"
            error_message = None

            # Get final critical stages that indicate success
            evaluation_success = any(
                log.agentName == "evaluation" and log.status == "success"
                for log in agent_logs
            )
            product_gen_success = any(
                log.agentName == "product_generator" and log.status == "success"
                for log in agent_logs
            )
            generation_success = any(
                log.agentName == "generation" and log.status == "success"
                for log in agent_logs
            )

            # Task is successful if critical final stages succeeded
            # even if earlier attempts/retries failed
            if generation_success or product_gen_success or evaluation_success:
                status = "completed"
                print(f"🔍 Status: {status} (final stages succeeded despite earlier failures)")
            elif any(log.status == "failed" for log in agent_logs):
                # Only mark as failed if ALL attempts failed and no success stages
                status = "failed"
                failed_logs = [log for log in agent_logs if log.status == "failed"]
                error_message = f"Failed at stage: {failed_logs[0].agentName}"
                print(f"🔍 Status: {status}")
            else:
                print(f"🔍 Status: {status}")

            # Step 5: Create Task record (new record, doesn't touch AgentLog)
            print(f"🔍 Creating Task record in database...")

            # Prisma Python Json fields require JSON strings, not raw Python objects
            import json as json_lib
            print(f"🔍 Converting Python objects to JSON strings...")
            stages_json = json_lib.dumps(stages) if stages else json_lib.dumps([])
            journey_json = json_lib.dumps(prompt_journey) if prompt_journey else json_lib.dumps([])
            performance_json = json_lib.dumps(performance) if performance else json_lib.dumps({})
            print(f"🔍 stages_json type: {type(stages_json)}, length: {len(stages_json)}")
            print(f"🔍 journey_json type: {type(journey_json)}, length: {len(journey_json)}")

            # Build data dict with required Json fields as JSON strings
            task_data = {
                "taskId": task_id,
                "shop": shop,
                "originalPrompt": original_prompt,
                "userRequest": json_lib.dumps({}),  # Empty object as JSON string
                "stages": stages_json,
                "promptJourney": journey_json,
                "totalDuration": total_duration,
                "creditsCost": total_credits,
                "performanceBreakdown": performance_json,
                "status": status,
            }

            # Add optional fields only if they have values (Json fields need JSON strings)
            if evaluation:
                task_data["evaluationResults"] = json_lib.dumps(evaluation)
            if generated_image_url:
                task_data["generatedImageUrl"] = generated_image_url
            if mockup_urls:
                task_data["mockupUrls"] = json_lib.dumps(mockup_urls)
            if final_prompt:
                task_data["finalPrompt"] = final_prompt
            if error_message:
                task_data["errorMessage"] = error_message
            if status == "completed":
                task_data["completedAt"] = datetime.now()

            task = await prisma.task.create(data=task_data)

            print(f"✅ Task record created successfully! ID: {task.id}")
            print(f"=" * 80)
            logger.info(f"Successfully created Task record for task_id={task_id}")
            return task.model_dump() if hasattr(task, 'model_dump') else dict(task)

        except Exception as e:
            print(f"=" * 80)
            print(f"❌ ERROR in TaskAggregator for task_id={task_id}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"=" * 80)

            import traceback
            print(f"Full traceback:")
            print(traceback.format_exc())
            print(f"=" * 80)

            logger.error(
                f"Error creating task from logs for task_id={task_id}",
                error=str(e),
                exc_info=True
            )
            return None

    def _build_stage_results(self, agent_logs: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract key input/output from each stage.

        This is a READ-ONLY operation that extracts data from completed logs.
        """
        stages = []

        for log in agent_logs:
            # Extract key fields from input/output
            key_input = self._extract_key_fields(log.input, log.agentName)
            key_output = self._extract_key_fields(log.output, log.agentName)

            stage = {
                "stage": log.agentName,
                "keyInput": key_input,
                "keyOutput": key_output,
                "duration": log.executionTime,  # Preserved from original log
                "creditsUsed": log.creditsUsed,   # Preserved from original log
                "status": log.status,
                "reasoning": log.reasoning if log.reasoning else None,
                "llmTokens": log.llmTokens if log.llmTokens else None,  # Token usage from LLMs
                "modelName": log.modelName if log.modelName else None,  # Model used (e.g., "gemini-2.0-flash")
            }
            stages.append(stage)

        return stages

    def _extract_key_fields(self, data: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """
        Extract the most relevant fields from input/output based on agent type.
        """
        # Handle JSON strings from database
        if isinstance(data, str):
            try:
                import json
                data = json.loads(data)
            except:
                return {"raw": str(data)}

        if not isinstance(data, dict):
            return {"raw": str(data)}

        # Define key fields for each agent
        key_field_map = {
            "planner": ["prompt", "product_types", "plan", "steps"],
            "prompt_manager": ["original_prompt", "enhanced_prompt", "style"],
            "model_selection": ["selected_model", "reasoning_model", "image_model"],
            "template_manager": ["template_id", "template_name", "template_text"],
            "credit_manager": ["credits_required", "credits_available", "approved"],
            "generation_agent": ["prompt", "image_url", "model_used"],
            "evaluation_agent": ["scores", "overall_score", "feedback"],
            "product_compositor": ["product_types", "mockups", "composite_urls"],
            "delivery_agent": ["delivery_method", "urls", "status"],
            "feedback_collector": ["rating", "feedback", "accepted"]
        }

        relevant_fields = key_field_map.get(agent_name, [])

        # Extract only relevant fields
        extracted = {}
        for key in relevant_fields:
            if key in data:
                extracted[key] = data[key]

        # If no relevant fields found, include first 3 top-level keys
        if not extracted and data:
            for key in list(data.keys())[:3]:
                extracted[key] = data[key]

        return extracted if extracted else data

    def _extract_prompt_journey(self, agent_logs: List[Any]) -> List[Dict[str, Any]]:
        """
        Track how the prompt evolved through the pipeline.
        """
        import json
        journey = []

        def parse_data(data):
            """Parse JSON string to dict if needed."""
            if isinstance(data, str):
                try:
                    return json.loads(data)
                except:
                    return {}
            return data if isinstance(data, dict) else {}

        def find_prompt(data_dict):
            """Find prompt in various possible keys with improved extraction."""
            if not isinstance(data_dict, dict):
                return None

            prompt_keys = [
                "prompt", "prompts", "enhanced_prompt", "final_prompt",
                "original_prompt", "primary", "refined_prompt", "user_prompt"
            ]
            for key in prompt_keys:
                if key in data_dict:
                    value = data_dict[key]
                    # Handle string prompts
                    if isinstance(value, str) and value.strip():
                        return value
                    # Handle dict prompts (like {primary: "...", optimized: {...}})
                    elif isinstance(value, dict):
                        if "primary" in value and isinstance(value["primary"], str):
                            return value["primary"]
                        # Check other keys in the dict
                        for subkey in ["enhanced_prompt", "final_prompt", "prompt"]:
                            if subkey in value and isinstance(value[subkey], str):
                                return value[subkey]

            # Check nested 'data' field
            if "data" in data_dict and isinstance(data_dict["data"], dict):
                for key in prompt_keys:
                    if key in data_dict["data"]:
                        value = data_dict["data"][key]
                        if isinstance(value, str) and value.strip():
                            return value
                        elif isinstance(value, dict) and "primary" in value:
                            return value["primary"]

            return None

        for log in agent_logs:
            # Parse both input and output
            input_data = parse_data(log.input)
            output_data = parse_data(log.output)

            # Check output first (higher priority)
            prompt = find_prompt(output_data)

            # If no prompt in output, check input
            if not prompt:
                prompt = find_prompt(input_data)

            # Only add if we found a valid string prompt
            if prompt and isinstance(prompt, str) and prompt.strip():
                journey.append({
                    "stage": log.agentName,
                    "prompt": prompt,
                    "timestamp": log.createdAt.isoformat() if isinstance(log.createdAt, datetime) else str(log.createdAt)
                })

        return journey

    def _build_performance_breakdown(self, agent_logs: List[Any]) -> Dict[str, Any]:
        """
        Analyze performance by stage.
        """
        total_duration = sum(log.executionTime for log in agent_logs)

        by_stage = []
        for log in agent_logs:
            percentage = (log.executionTime / total_duration * 100) if total_duration > 0 else 0

            by_stage.append({
                "stage": log.agentName,
                "duration": log.executionTime,
                "percentage": round(percentage, 2),
                "credits": log.creditsUsed
            })

        # Identify bottlenecks (stages taking >20% of total time)
        bottlenecks = [
            stage["stage"]
            for stage in by_stage
            if stage["percentage"] > 20
        ]

        return {
            "byStage": by_stage,
            "bottlenecks": bottlenecks,
            "totalDuration": total_duration
        }

    def _extract_evaluation_results(self, agent_logs: List[Any]) -> Optional[Dict[str, Any]]:
        """
        Extract evaluation results from evaluation agent log.
        """
        evaluation_log = next(
            (log for log in agent_logs if log.agentName == "evaluation_agent"),
            None
        )

        if not evaluation_log or not isinstance(evaluation_log.output, dict):
            return None

        output = evaluation_log.output

        return {
            "overallScore": output.get("overall_score"),
            "objectiveScores": output.get("objective_scores"),
            "subjectiveScores": output.get("subjective_scores"),
            "reasoning": output.get("reasoning") or evaluation_log.reasoning,
            "recommendations": output.get("recommendations", [])
        }

    def _extract_final_prompt(self, agent_logs: List[Any]) -> Optional[str]:
        """
        Extract the final prompt sent to the generation model.
        """
        # Look for generation agent
        generation_log = next(
            (log for log in agent_logs if log.agentName == "generation_agent"),
            None
        )

        if generation_log and isinstance(generation_log.input, dict):
            return generation_log.input.get("prompt")

        return None


# Singleton instance
task_aggregator = TaskAggregator()
