"""
Agent Orchestrator - Coordinates the full agent workflow.

Manages the sequential execution of all agents in the pipeline.

REFACTORED: Now uses CreditService for centralized credit operations.
The orchestrator handles user data fetching, credit checking, and deduction.
"""
from typing import Any, Dict, Optional
from datetime import datetime
from uuid import uuid4
import numpy as np
from src.agents.base_agent import SequentialAgent, AgentContext, AgentResult
from src.agents.interactive_agent import InteractiveAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.prompt_manager_agent import PromptManagerAgent
from src.agents.model_selection_agent import ModelSelectionAgent
from src.agents.generation_agent import GenerationAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.product_generator_agent import ProductGeneratorAgent
from src.api.a2a_client import A2AClient
from src.api.clients import Palet8APIClient, GenerationType
from src.services.credit_service import CreditService
from src.services.asset_service import AssetService
from src.services.template_service import TemplateService
from src.services.session_service import SessionService
from src.utils import get_logger
from src.models.schemas import ImageModel, ReasoningModel
from config import settings
from src.database import prisma
from src.api.websocket import notify_agent_update, notify_workflow_complete

logger = get_logger(__name__)


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert NumPy types and other non-JSON-serializable types to native Python types.

    Args:
        obj: Object to sanitize

    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'item'):  # Other NumPy scalars
        return obj.item()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif hasattr(obj, '__dict__'):
        # For custom objects, try to convert to dict
        try:
            return sanitize_for_json(vars(obj))
        except:
            return str(obj)
    else:
        # For other types, try to convert to string
        try:
            return str(obj)
        except:
            return None


class AgentOrchestrator:
    """
    Orchestrates the full multi-agent workflow.

    Pipeline:
    1. Interactive Agent - Validates input
    2. Planner Agent - Creates execution plan
    3. Prompt Manager Agent - Optimizes prompts
    4. Model-Selection Agent - Chooses optimal models
    5. Generation Agent - Creates images
    6. Evaluation Agent - Scores and approves images
    7. Product Generator Agent - Creates product mockups
    """

    def __init__(self) -> None:
        """Initialize the orchestrator."""
        logger.info("Initializing Agent Orchestrator")

        # REFACTORED: Initialize CreditService for centralized credit operations
        self.credit_service = CreditService()

        # Initialize database service layer
        self.asset_service = AssetService(prisma)
        self.template_service = TemplateService(prisma)
        self.session_service = SessionService(prisma)

        # Initialize all agents (no longer passing api_client to interactive_agent)
        self.interactive_agent = InteractiveAgent()
        self.planner_agent = PlannerAgent()
        self.prompt_manager = PromptManagerAgent()
        self.model_selection = ModelSelectionAgent()
        # Generation agent still needs API client for fetching generation costs
        self.generation_agent = GenerationAgent(api_client=Palet8APIClient())
        self.evaluation_agent = EvaluationAgent()

        # Product Generator can run as A2A microservice or local agent
        self.use_a2a = getattr(settings, "use_product_a2a", False)
        if self.use_a2a:
            self.a2a_client = A2AClient()
            logger.info("Using A2A Product Generator microservice")
        else:
            self.product_generator = ProductGeneratorAgent()
            logger.info("Using local Product Generator agent")

        logger.info("Agent Orchestrator initialized with 7 agents + CreditService")

    def _print_agent_error_if_present(
        self,
        agent_name: str,
        task_id: str,
        agent_output: Dict[str, Any],
    ) -> None:
        """
        Print agent error details to stdout immediately for debugging.

        This makes errors visible in Cloud Run logs without needing to query the database.
        """
        try:
            if isinstance(agent_output, dict):
                success = agent_output.get("success", True)  # Default to True if not specified
                error = agent_output.get("error", "")

                if not success or error:
                    print(f"")
                    print(f"{'='*80}")
                    print(f"❌ AGENT FAILURE DETECTED")
                    print(f"{'='*80}")
                    print(f"Agent: {agent_name}")
                    print(f"Task ID: {task_id}")
                    print(f"Success: {success}")
                    print(f"Error: {error}")
                    print(f"{'='*80}")
                    print(f"")
                else:
                    print(f"✓ {agent_name} completed successfully (task_id={task_id})")
        except Exception as e:
            # Don't let error logging break the workflow
            print(f"⚠️  Error while logging agent error: {e}")

    async def _log_agent_execution(
        self,
        agent_name: str,
        task_id: str,
        shop_domain: Optional[str],
        agent_input: Dict[str, Any],
        agent_output: Dict[str, Any],
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """
        Log agent execution to AgentLog table.

        Args:
            agent_name: Name of the agent (e.g., "planner", "evaluation")
            task_id: Unique task identifier
            shop_domain: Shop domain (can be None for legacy customers)
            agent_input: Input data passed to agent
            agent_output: Output data from agent
            start_time: Execution start time
            end_time: Execution end time
        """
        # DEBUG: Use print() since logger.info() doesn't show in Cloud Run logs
        print(f"🔍 _log_agent_execution CALLED for {agent_name}, task_id={task_id}")

        try:
            # PRE-PROCESS: Replace ALL complex objects in agent_input with simple summaries
            # This prevents Prisma JSON validation errors from nested Pydantic models, large data, etc.
            if isinstance(agent_input, dict):
                agent_input = agent_input.copy()  # Don't mutate original

                # Simplify AgentContext objects
                if "context" in agent_input and hasattr(agent_input["context"], 'task_id'):
                    agent_input["context"] = {
                        "task_id": getattr(agent_input["context"], 'task_id', None),
                        "user_id": getattr(agent_input["context"], 'user_id', None),
                        "shop_domain": getattr(agent_input["context"], 'shop_domain', None),
                    }
                    print(f"🔍 Simplified context for {agent_name}")

                # Simplify image objects (contains base64 data which is huge)
                if "best_image" in agent_input and isinstance(agent_input["best_image"], dict):
                    img = agent_input["best_image"]
                    agent_input["best_image"] = {
                        "image_index": img.get("image_index"),
                        "score": img.get("score"),
                        "approved": img.get("approved"),
                        "has_base64": "base64_data" in img,
                        "feedback": str(img.get("feedback", ""))[:100],  # Truncate
                    }
                    print(f"🔍 Simplified best_image for {agent_name}")

                # Simplify images array (also contains base64)
                if "images" in agent_input and isinstance(agent_input["images"], list):
                    agent_input["images"] = {"count": len(agent_input["images"])}
                    print(f"🔍 Simplified images array for {agent_name}")

                # Simplify prompts dict (can be large)
                if "prompts" in agent_input and isinstance(agent_input["prompts"], dict):
                    agent_input["prompts"] = {
                        "primary": str(agent_input["prompts"].get("primary", ""))[:100],
                        "keys": list(agent_input["prompts"].keys())
                    }
                    print(f"🔍 Simplified prompts for {agent_name}")

            # Convert AgentResult to dict if needed
            from dataclasses import is_dataclass, asdict
            if is_dataclass(agent_output):
                agent_output_dict = asdict(agent_output)
            else:
                agent_output_dict = agent_output

            # Calculate execution time
            execution_time = int((end_time - start_time).total_seconds() * 1000)  # milliseconds

            # Determine status
            status = "success" if agent_output_dict.get("success", False) else "failed"

            # Extract routing metadata
            routing_metadata = agent_output_dict.get("routing_metadata", {}) or {}
            routing_mode = routing_metadata.get("mode")
            used_llm = routing_metadata.get("used_llm", False)
            confidence = routing_metadata.get("confidence")
            fallback_used = routing_metadata.get("fallback_used", False)

            # Extract reasoning
            reasoning = routing_metadata.get("reasoning")

            # Estimate credits used (will be more accurate in future)
            credits_used = 0
            if used_llm:
                # Rough estimate: 1 credit per 1000 tokens, assume 500 tokens avg
                credits_used = 1

            # Extract LLM tokens and model name if available
            llm_tokens = routing_metadata.get("tokens") or routing_metadata.get("llm_tokens") or 0
            model_name = routing_metadata.get("model_name")

            # Sanitize input and output to ensure Prisma JSON compatibility
            # Use custom encoder that handles Pydantic models, dataclasses, and all edge cases
            import json
            from pydantic import BaseModel

            class PrismaJSONEncoder(json.JSONEncoder):
                """Custom JSON encoder that aggressively converts all types to JSON-safe types."""
                def default(self, obj):
                    # Handle Pydantic models
                    if isinstance(obj, BaseModel):
                        return obj.model_dump()
                    # Handle dataclasses
                    if is_dataclass(obj):
                        return asdict(obj)
                    # Handle bytes
                    if isinstance(obj, bytes):
                        try:
                            return obj.decode('utf-8')
                        except:
                            return f"<bytes: {len(obj)} bytes>"
                    # Handle enums
                    if hasattr(obj, 'value'):
                        return obj.value
                    # Handle objects with __dict__
                    if hasattr(obj, '__dict__'):
                        return vars(obj)
                    # Convert everything else to string
                    return str(obj)

            # SMART APPROACH: Preserve critical data while simplifying complex objects
            # This ensures Prisma JSON validator accepts it AND we can extract prompts/data
            try:
                # Helper function to safely extract dict values
                def safe_dict_extract(d, allowed_keys):
                    """Safely extract string values from dict."""
                    result = {}
                    if not isinstance(d, dict):
                        return result
                    for k in allowed_keys:
                        if k in d:
                            v = d[k]
                            if isinstance(v, str):
                                result[k] = v[:500]
                            elif isinstance(v, (int, float, bool)):
                                result[k] = v
                    return result

                # Build sanitized input with preserved prompt data
                sanitized_input = {
                    "keys": list(agent_input.keys()) if isinstance(agent_input, dict) else []
                }

                # Preserve prompt-related fields for prompt journey extraction
                if isinstance(agent_input, dict):
                    prompt_keys = ["prompt", "prompts", "enhanced_prompt", "final_prompt",
                                   "original_prompt", "primary", "refined_prompt", "user_prompt"]
                    for key in prompt_keys:
                        if key in agent_input:
                            value = agent_input[key]
                            if isinstance(value, str):
                                sanitized_input[key] = value[:1000]  # Limit string length
                            elif isinstance(value, dict):
                                # Only extract safe fields from nested dict
                                sanitized_input[key] = safe_dict_extract(value, ["primary", "optimized", "style"])

                # Build sanitized output with preserved critical data
                sanitized_output = {
                    "success": agent_output_dict.get("success", False) if isinstance(agent_output_dict, dict) else False,
                    "error": str(agent_output_dict.get("error", ""))[:500] if isinstance(agent_output_dict, dict) else "",
                }

                # Preserve prompt data from output too
                if isinstance(agent_output_dict, dict):
                    prompt_keys = ["prompt", "prompts", "enhanced_prompt", "final_prompt",
                                   "original_prompt", "primary", "refined_prompt"]
                    for key in prompt_keys:
                        if key in agent_output_dict:
                            value = agent_output_dict[key]
                            if isinstance(value, str):
                                sanitized_output[key] = value[:1000]
                            elif isinstance(value, dict):
                                sanitized_output[key] = safe_dict_extract(value, ["primary", "optimized", "style"])

                    # Preserve 'data' field if it contains prompts
                    if "data" in agent_output_dict and isinstance(agent_output_dict["data"], dict):
                        data = agent_output_dict["data"]
                        sanitized_data = {}
                        for key in prompt_keys:
                            if key in data:
                                value = data[key]
                                if isinstance(value, str):
                                    sanitized_data[key] = value[:1000]
                                elif isinstance(value, dict):
                                    sanitized_data[key] = safe_dict_extract(value, ["primary", "optimized", "style"])
                        if sanitized_data:
                            sanitized_output["data"] = sanitized_data

                print(f"🔍 Using smart preservation for {agent_name}, input keys: {list(sanitized_input.keys())}, output keys: {list(sanitized_output.keys())}")
            except Exception as e:
                # Absolute fallback: use string representation
                print(f"⚠️  JSON serialization warning for {agent_name}: {str(e)}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                sanitized_input = {"error": "serialization_failed", "summary": str(agent_input)[:200]}
                sanitized_output = {"error": "serialization_failed", "summary": str(agent_output_dict)[:200]}

            # Use shared Prisma instance (already connected on app startup)
            print(f"🔍 About to call prisma.agentlog.create for {agent_name}, task_id={task_id}")
            print(f"🔍 sanitized_input type: {type(sanitized_input)}, keys: {list(sanitized_input.keys()) if isinstance(sanitized_input, dict) else 'N/A'}")
            print(f"🔍 sanitized_output type: {type(sanitized_output)}, is_dict: {isinstance(sanitized_output, dict)}")

            # Convert to JSON strings - Prisma Python Json fields require JSON strings
            import json as json_module
            print(f"🔍 Converting to JSON strings for {agent_name}")
            try:
                input_json_str = json_module.dumps(sanitized_input)
                output_json_str = json_module.dumps(sanitized_output)
                print(f"🔍 Conversion successful - input length: {len(input_json_str)}, output length: {len(output_json_str)}")
            except Exception as e:
                print(f"⚠️ JSON conversion failed: {e}")
                input_json_str = json_module.dumps({"error": "conversion_failed"})
                output_json_str = json_module.dumps({"error": "conversion_failed"})

            result = await prisma.agentlog.create(
                data={
                    "shop": shop_domain or "legacy_customer",
                    "taskId": task_id,
                    "agentName": agent_name,
                    "input": input_json_str,  # JSON string required
                    "output": output_json_str,  # JSON string required
                    "reasoning": reasoning,
                    "executionTime": execution_time,
                    "status": status,
                    "routingMode": routing_mode,
                    "usedLlm": used_llm,
                    "confidence": confidence,
                    "fallbackUsed": fallback_used,
                    "creditsUsed": credits_used,
                    "llmTokens": llm_tokens,
                    "modelName": model_name,
                }
            )

            print(f"🔍 prisma.agentlog.create completed, result ID: {result.id if result else 'None'}, task_id={task_id}")

            logger.info(
                f"✅ Successfully logged execution for {agent_name}",
                task_id=task_id,
                status=status,
                execution_time=execution_time,
                routing_mode=routing_mode,
            )

        except Exception as e:
            # DEBUG: Use print() since logger.error() doesn't show in Cloud Run logs
            print(f"❌ ERROR in _log_agent_execution for {agent_name}, task_id={task_id}: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")

            logger.error(
                f"❌ Failed to log execution for {agent_name}",
                error=str(e),
                error_type=type(e).__name__,
                task_id=task_id,
                agent_name=agent_name,
                exc_info=True,
            )
            # Don't fail the pipeline if logging fails

    async def run_full_pipeline(
        self,
        user_prompt: str,
        user_id: str,
        customer_id: str,
        email: str,
        shop_domain: Optional[str] = None,
        reasoning_model: Optional[str] = None,
        image_model: Optional[str] = None,
        dimensions: str = "1024x1024",
        num_images: int = 1,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full agent pipeline with centralized credit management.

        REFACTORED: Uses CreditService for all credit operations.
        Flow:
        1. Create AgentContext
        2. CreditService: Fetch user profile and credit balance
        3. CreditService: Pre-check sufficient credits
        4. Interactive Agent: Validate prompt
        5. Planner → Prompt Manager → Model Selection agents
        6. Generation Agent: Generate images
        7. CreditService: Deduct credits after successful generation
        8. Evaluation → Product Generator agents

        Args:
            user_prompt: User's input prompt
            user_id: User identifier
            customer_id: Shopify customer ID
            email: User email address
            shop_domain: Multi-tenant shop domain
            reasoning_model: Optional reasoning model preference
            image_model: Optional image model preference
            dimensions: Image dimensions (default: 1024x1024)
            num_images: Number of images to generate

        Returns:
            Complete pipeline result with images and credit transaction info
        """
        # Use provided task_id or generate unique task ID for this pipeline run
        task_id = task_id or str(uuid4())

        # DEBUG: Use print() instead of logger since structured logging doesn't show in Cloud Run
        print(f"🔍 ORCHESTRATOR STARTED: task_id={task_id}, user_id={user_id}")

        logger.info(
            "Starting full pipeline with credit integration",
            task_id=task_id,
            user_id=user_id,
            customer_id=customer_id,
            shop_domain=shop_domain,
            prompt_length=len(user_prompt),
        )

        try:
            # Prisma is already connected on app startup via lifespan manager

            # Step 0: Create AgentContext
            context = AgentContext(
                task_id=task_id,
                user_id=user_id,
                customer_id=customer_id,
                email=email,
                shop_domain=shop_domain,
                reasoning_model=ReasoningModel(reasoning_model or "gemini"),
                image_model=ImageModel(image_model or "flux"),
                shared_data={
                    "prompt": user_prompt,
                    "dimensions": dimensions,
                    "num_images": num_images,
                },
                metadata={
                    "source": "orchestrator",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Step 0.5: Create asset record in database
            logger.info("Creating asset record")
            try:
                asset = await self.asset_service.create_asset(
                    shop=shop_domain,
                    task_id=task_id,
                    prompt=user_prompt[:500],  # Truncate long prompts
                    cost=0,  # Will be updated after credit deduction
                    status="processing"
                )
                logger.info(f"Asset created: {asset['id']}")
            except Exception as e:
                logger.error(f"Failed to create asset record: {e}", exc_info=True)
                # Continue execution even if asset creation fails

            # Step 1: CreditService - Fetch user profile and credit balance
            logger.info("Step 1: Fetching user profile and credit balance")
            user_data_success = await self.credit_service.fetch_and_populate_user_data(context)

            if not user_data_success:
                return {
                    "success": False,
                    "error": "Failed to fetch user profile and credit data",
                    "error_code": "USER_DATA_FETCH_FAILED",
                    "stage": "credit_service",
                    "task_id": task_id,
                }

            logger.info(
                "User profile fetched",
                username=context.username,
                credit_balance=context.credit_balance,
            )

            # Step 2: CreditService - Pre-check sufficient credits
            logger.info("Step 2: Checking credit balance")
            min_credits = self.credit_service.get_minimum_required_credits()
            credit_check = await self.credit_service.check_sufficient_credits(
                context=context,
                required_credits=min_credits
            )

            if not credit_check.sufficient:
                logger.warning(
                    "Insufficient credits",
                    current=credit_check.current_balance,
                    required=credit_check.required_credits,
                    shortfall=credit_check.shortfall
                )
                return {
                    "success": False,
                    "error": "Insufficient credits for generation",
                    "error_code": "INSUFFICIENT_CREDITS",
                    "stage": "credit_check",
                    "task_id": task_id,
                    "credits": {
                        "current_balance": credit_check.current_balance,
                        "minimum_required": credit_check.required_credits,
                        "shortfall": credit_check.shortfall,
                    },
                    "user": {
                        "username": credit_check.username,
                        "avatar": credit_check.avatar,
                    }
                }

            logger.info("Credit check passed", balance=credit_check.current_balance)

            # Step 3: Interactive Agent - Validate prompt only
            logger.info("Step 3: Interactive Agent")
            start_time = datetime.now()

            interactive_result = await self.interactive_agent.execute(context)

            end_time = datetime.now()

            # Print errors immediately if present
            self._print_agent_error_if_present(
                "interactive",
                task_id,
                {
                    "success": interactive_result.success,
                    "error": interactive_result.error,
                    "data": interactive_result.data,
                }
            )

            # Log execution
            await self._log_agent_execution(
                agent_name="interactive",
                task_id=task_id,
                shop_domain=shop_domain,
                agent_input={"prompt": user_prompt, "customer_id": customer_id},
                agent_output={
                    "success": interactive_result.success,
                    "error": interactive_result.error,
                    "data": interactive_result.data,
                },
                start_time=start_time,
                end_time=end_time,
            )

            # Notify real-time subscribers
            await notify_agent_update(task_id, "interactive", {
                "status": interactive_result.success,
                "duration": int((end_time - start_time).total_seconds() * 1000),
            })

            # Interactive agent now only validates prompts, check for validation errors
            if not interactive_result.success:
                error_code = interactive_result.metadata.get("error_code", "VALIDATION_FAILED")
                return {
                    "success": False,
                    "error": interactive_result.error,
                    "error_code": error_code,
                    "stage": "interactive",
                    "task_id": task_id,
                }

            logger.info("Prompt validation passed")

            # Step 4: Planner Agent - Create execution plan
            logger.info("Step 4: Planner Agent")
            start_time = datetime.now()
            planner_input = {"context": context}
            planner_result = await self.planner_agent.run(planner_input)
            end_time = datetime.now()

            # Log execution            # Print errors immediately if present
            # Log execution            self._print_agent_error_if_present("planner", task_id, planner_result)

            # Log execution
            await self._log_agent_execution(
                agent_name="planner",
                task_id=task_id,
                shop_domain=shop_domain,
                agent_input=planner_input,
                agent_output=planner_result,
                start_time=start_time,
                end_time=end_time,
            )

            if not planner_result.get("success"):
                return {
                    "success": False,
                    "error": planner_result.get("error"),
                    "stage": "planner",
                    "task_id": task_id,
                }

            # Don't overwrite context - planner doesn't modify it

            # Step 3: Prompt Manager - Optimize prompts
            logger.info("Step 3: Prompt Manager Agent")
            start_time = datetime.now()
            prompt_input = {"context": context}
            prompt_result = await self.prompt_manager.run(prompt_input)
            end_time = datetime.now()

            # Log execution
            await self._log_agent_execution(
                agent_name="prompt_manager",
                task_id=task_id,
                shop_domain=shop_domain,
                agent_input=prompt_input,
                agent_output=prompt_result,
                start_time=start_time,
                end_time=end_time,
            )

            if not prompt_result.get("success"):
                return {
                    "success": False,
                    "error": prompt_result.get("error"),
                    "stage": "prompt_manager",
                    "task_id": task_id,
                }

            prompts = prompt_result.get("prompts", {})

            # Step 4: Model-Selection Agent - Choose optimal models
            logger.info("Step 4: Model-Selection Agent")
            start_time = datetime.now()
            model_selection_input = {
                "context": context,
                "prompts": prompts,
            }
            model_selection_result = await self.model_selection.run(model_selection_input)
            end_time = datetime.now()

            # Log execution
            await self._log_agent_execution(
                agent_name="model_selection",
                task_id=task_id,
                shop_domain=shop_domain,
                agent_input=model_selection_input,
                agent_output=model_selection_result,
                start_time=start_time,
                end_time=end_time,
            )

            if not model_selection_result.get("success"):
                return {
                    "success": False,
                    "error": model_selection_result.get("error"),
                    "stage": "model_selection",
                    "task_id": task_id,
                }

            selected_models = model_selection_result.get("selected_models", {})
            # Update context fields from model_selection result without overwriting the object
            returned_context = model_selection_result.get("context", {})
            if isinstance(returned_context, dict):
                context.reasoning_model = returned_context.get("reasoning_model", context.reasoning_model)
                context.image_model = returned_context.get("image_model", context.image_model)

            # Step 5-6: Generation and Evaluation Loop with Iterative Refinement
            # Allow up to max_refinement_iterations retries if all images are rejected
            max_refinement_iterations = getattr(context, "max_refinement_iterations", 2)
            best_image = None
            evaluation_result = None
            refinement_history = []

            for attempt in range(max_refinement_iterations + 1):
                logger.info(
                    f"Generation attempt {attempt + 1}/{max_refinement_iterations + 1}",
                    iteration=attempt,
                )

                # Step 5: Generation Agent - Create images
                logger.info(f"Step 5: Generation Agent (attempt {attempt + 1})")
                start_time = datetime.now()
                # Add prompts and selected_models to context.shared_data for generation agent
                context.shared_data["prompts"] = prompts
                context.shared_data["selected_models"] = selected_models
                generation_result = await self.generation_agent.run(context)
                end_time = datetime.now()

                # Log execution                # Print errors immediately if present
                # Log execution                self._print_agent_error_if_present("generation", task_id, generation_result)

                # Log execution
                await self._log_agent_execution(
                    agent_name="generation",
                    task_id=task_id,
                    shop_domain=shop_domain,
                    agent_input={"context": context.task_id, "prompts": len(prompts), "models": len(selected_models)},
                    agent_output=generation_result,
                    start_time=start_time,
                    end_time=end_time,
                )

                if not generation_result.success:
                    logger.warning(f"Generation failed on attempt {attempt + 1}")
                    if attempt == max_refinement_iterations:
                        # Final attempt failed
                        return {
                            "success": False,
                            "error": generation_result.error,
                            "stage": "generation",
                            "refinement_attempts": attempt + 1,
                            "task_id": task_id,
                        }
                    # Try again with refined prompts
                    continue

                images = generation_result.data.get("images", []) if generation_result.data else []

                # Step 6: Evaluation Agent - Score and approve images
                logger.info(f"Step 6: Evaluation Agent (attempt {attempt + 1})")
                start_time = datetime.now()
                evaluation_input = {
                    "images": images,
                    "context": context,
                    "prompts": prompts,
                }
                evaluation_result = await self.evaluation_agent.run(evaluation_input)
                end_time = datetime.now()

                # Log execution                # Print errors immediately if present
                # Log execution                self._print_agent_error_if_present("evaluation", task_id, evaluation_result)

                # Log execution
                await self._log_agent_execution(
                    agent_name="evaluation",
                    task_id=task_id,
                    shop_domain=shop_domain,
                    agent_input=evaluation_input,
                    agent_output=evaluation_result,
                    start_time=start_time,
                    end_time=end_time,
                )

                if not evaluation_result.get("success"):
                    logger.warning(f"Evaluation failed on attempt {attempt + 1}")
                    if attempt == max_refinement_iterations:
                        return {
                            "success": False,
                            "error": evaluation_result.get("error"),
                            "stage": "evaluation",
                            "refinement_attempts": attempt + 1,
                            "task_id": task_id,
                        }
                    continue

                best_image = evaluation_result.get("best_image", {})

                # Check if we have approved images
                if best_image and best_image.get("approved"):
                    logger.info(
                        f"Approved image found on attempt {attempt + 1}",
                        score=best_image.get("score"),
                    )
                    break

                # No approved images on this attempt
                logger.warning(
                    f"No approved images on attempt {attempt + 1}",
                    best_score=best_image.get("score", 0) if best_image else 0,
                )

                # Save feedback for history
                refinement_history.append({
                    "attempt": attempt + 1,
                    "best_score": best_image.get("score", 0) if best_image else 0,
                    "evaluations": evaluation_result.get("evaluations", []),
                })

                # If this is not the last attempt, refine prompts based on feedback
                if attempt < max_refinement_iterations:
                    logger.info(f"Refining prompts for attempt {attempt + 2}")
                    prompts = await self._refine_prompts_from_feedback(
                        prompts,
                        evaluation_result.get("evaluations", []),
                        attempt + 1
                    )

            # After all attempts, check if we have an approved image
            if not best_image or not best_image.get("approved"):
                logger.error("No approved images after all refinement attempts")
                return {
                    "success": False,
                    "error": f"No images met approval threshold after {max_refinement_iterations + 1} attempts",
                    "stage": "evaluation",
                    "refinement_attempts": max_refinement_iterations + 1,
                    "refinement_history": refinement_history,
                    "best_score": best_image.get("score", 0) if best_image else 0,
                    "all_evaluations": evaluation_result.get("evaluations", []),
                    "task_id": task_id,
                }

            # Step 6: CreditService - Deduct credits after successful generation
            logger.info("Step 6: Deducting credits for successful generation")

            # Get cost information from generation result
            generation_cost = context.shared_data.get("generation_cost", {})
            generation_type = GenerationType(generation_cost.get("generation_type", "M"))
            num_images_generated = generation_cost.get("num_images", 1)
            prompt_used = generation_cost.get("prompt", user_prompt[:100])
            model_used = generation_cost.get("model", context.image_model.value)

            deduction_result = await self.credit_service.deduct_credits_for_generation(
                context=context,
                generation_type=generation_type,
                num_images=num_images_generated,
                prompt=prompt_used,
                model=model_used
            )

            if not deduction_result.success:
                # Log the failure but don't fail the pipeline
                # Customer experience: they still get their images
                logger.error(
                    "Credit deduction failed but images were generated",
                    error=deduction_result.error,
                    task_id=task_id,
                    customer_id=context.customer_id
                )
                # The COMPENSATION_REQUIRED log is already created by CreditService

            logger.info(
                "Credit deduction completed",
                success=deduction_result.success,
                amount_deducted=deduction_result.amount_deducted,
                new_balance=deduction_result.balance_after
            )

            # Step 7: Product Generator - Create product mockups with graceful fallback
            logger.info("Step 7: Product Generator Agent")
            product_result = None
            used_fallback = False

            if self.use_a2a:
                # Try A2A microservice for GPU-enabled product generation
                logger.info("Attempting A2A Product Generator microservice")
                start_time = datetime.now()
                product_input = {
                    "best_image": best_image,
                    "context": context,
                }
                product_result = await self.a2a_client.generate_products(
                    best_image=best_image,
                    context=context,
                )
                end_time = datetime.now()

                # Graceful degradation: fallback to local if A2A fails
                if not product_result.get("success"):
                    logger.warning(
                        "A2A Product Generator failed, falling back to local agent",
                        error=product_result.get("error"),
                        circuit_open=product_result.get("circuit_open", False),
                    )

                    # Ensure local product generator is initialized
                    if not hasattr(self, 'product_generator'):
                        from src.agents.product_generator_agent import ProductGeneratorAgent
                        self.product_generator = ProductGeneratorAgent()
                        logger.info("Initialized local Product Generator as fallback")

                    # Fallback to local agent
                    start_time = datetime.now()
                    product_result = await self.product_generator.run({
                        "best_image": best_image,
                        "context": context,
                    })
                    end_time = datetime.now()
                    used_fallback = True

                    if product_result.get("success"):
                        logger.info("Local Product Generator fallback succeeded")
                        product_result["used_fallback"] = True
                        product_result["fallback_reason"] = "A2A service unavailable"

                # Log execution (whether A2A or fallback)
                await self._log_agent_execution(
                    agent_name="product_generator",
                    task_id=task_id,
                    shop_domain=shop_domain,
                    agent_input=product_input,
                    agent_output=product_result,
                    start_time=start_time,
                    end_time=end_time,
                )
            else:
                # Use local agent directly
                logger.info("Using local Product Generator (A2A disabled)")
                start_time = datetime.now()
                product_input = {
                    "best_image": best_image,
                    "context": context,
                }
                product_result = await self.product_generator.run(product_input)
                end_time = datetime.now()

                # Log execution                # Print errors immediately if present
                # Log execution                self._print_agent_error_if_present("product_generator", task_id, product_result)

                # Log execution
                await self._log_agent_execution(
                    agent_name="product_generator",
                    task_id=task_id,
                    shop_domain=shop_domain,
                    agent_input=product_input,
                    agent_output=product_result,
                    start_time=start_time,
                    end_time=end_time,
                )

            # Final validation after fallback attempts
            if not product_result or not product_result.get("success"):
                return {
                    "success": False,
                    "error": product_result.get("error") if product_result else "Product generation failed",
                    "stage": "product_generator",
                    "used_fallback": used_fallback,
                    "task_id": task_id,
                }

            # Compile final result
            logger.info(
                "Pipeline complete",
                task_id=task_id,
                num_products=len(product_result.get("products", [])),
            )

            # Update asset record with completed status
            # Note: imageUrl is optional in schema, images are stored as base64 in product results
            try:
                # Image URL not available (images use base64_data), update status to completed
                image_url = best_image.get("url") if best_image else None
                await self.asset_service.update_asset_completed(
                    task_id=task_id,
                    image_url=image_url  # None if no URL available
                )
                logger.info(f"Asset updated to completed: {task_id}, image_url={'set' if image_url else 'null'}")
            except Exception as e:
                logger.error(f"Failed to update asset record: {e}", exc_info=True)
                # Continue execution even if asset update fails

            # Notify real-time subscribers that workflow is complete with full result data
            await notify_workflow_complete(task_id, "completed", {
                "success": True,
                "task_id": task_id,
                "best_image": best_image,
                "products": product_result.get("products", []),
                "user": {
                    "username": context.username,
                    "avatar": context.avatar,
                    "credit_balance": context.credit_balance,
                },
                "credits": {
                    "cost": generation_cost.get("required_credits", 0),
                    "generation_type": generation_cost.get("generation_type", "M"),
                    "balance_before": deduction_result.balance_before,
                    "balance_after": deduction_result.balance_after,
                    "transaction_id": deduction_result.transaction_id,
                    "deduction_successful": deduction_result.success,
                },
                "all_evaluations": evaluation_result.get("evaluations", []),
                "metadata": {
                    "user_id": user_id,
                    "original_prompt": user_prompt,
                    "num_images_generated": len(images),
                    "num_products": len(product_result.get("products", [])),
                },
            })

            # REFACTORED: Credit info now comes from CreditService deduction result
            return {
                "success": True,
                "task_id": task_id,
                # User and credit info with updated balance
                "user": {
                    "username": context.username,
                    "avatar": context.avatar,
                    "credit_balance": context.credit_balance,  # Updated by CreditService
                },
                "credits": {
                    "cost": generation_cost.get("required_credits", 0),
                    "generation_type": generation_cost.get("generation_type", "M"),
                    "balance_before": deduction_result.balance_before,
                    "balance_after": deduction_result.balance_after,
                    "transaction_id": deduction_result.transaction_id,
                    "deduction_successful": deduction_result.success,
                },
                "pipeline_stages": {
                    "interactive": interactive_result if isinstance(interactive_result, dict) else {"success": interactive_result.success},
                    "planner": planner_result,
                    "prompt_manager": prompt_result,
                    "model_selection": model_selection_result,
                    "generation": generation_result.data if generation_result.data else {"success": generation_result.success},
                    "evaluation": evaluation_result,
                    "product_generator": product_result,
                },
                "products": product_result.get("products", []),
                "best_image": best_image,
                "all_evaluations": evaluation_result.get("evaluations", []),
                "selected_models": selected_models,
                "estimated_cost": model_selection_result.get("estimated_cost", 0.0),
                "metadata": {
                    "user_id": user_id,
                    "original_prompt": user_prompt,
                    "num_images_generated": len(images),
                    "num_products": len(product_result.get("products", [])),
                },
            }

        except Exception as e:
            logger.error(
                "Pipeline failed",
                task_id=task_id,
                error=str(e),
                exc_info=True,
            )

            # Update asset record with failed status
            try:
                await self.asset_service.update_asset_failed(
                    task_id=task_id,
                    error_message=str(e)[:500]  # Truncate long error messages
                )
                logger.info(f"Asset updated to failed: {task_id}")
            except Exception as asset_error:
                logger.error(f"Failed to update asset record: {asset_error}", exc_info=True)
                # Continue execution even if asset update fails

            return {
                "success": False,
                "error": f"Pipeline execution failed: {str(e)}",
                "stage": "orchestrator",
                "task_id": task_id,
            }
        finally:
            # Prisma connection is managed by app lifecycle (lifespan manager)
            pass

    async def _refine_prompts_from_feedback(
        self,
        current_prompts: Dict[str, Any],
        evaluations: list[Dict[str, Any]],
        attempt: int,
    ) -> Dict[str, Any]:
        """
        Refine prompts based on evaluation feedback.

        Args:
            current_prompts: Current prompts
            evaluations: Evaluation results from all images
            attempt: Current refinement attempt number

        Returns:
            Refined prompts dictionary
        """
        logger.info("Refining prompts based on evaluation feedback")

        # Analyze feedback from evaluations
        common_issues = []
        avg_scores = {
            "prompt_adherence": 0,
            "aesthetics": 0,
            "product_suitability": 0,
            "safety": 0,
        }

        if evaluations:
            for eval_item in evaluations:
                scores = eval_item.get("scores", {})
                for key in avg_scores:
                    avg_scores[key] += scores.get(key, 0)

            # Calculate averages
            for key in avg_scores:
                avg_scores[key] = avg_scores[key] / len(evaluations)

            # Identify weakest areas
            sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1])
            weakest_area = sorted_scores[0][0]

            logger.info(
                "Evaluation analysis complete",
                avg_scores=avg_scores,
                weakest_area=weakest_area,
            )

        # Build refinement guidance based on weak areas
        refinements = []

        if avg_scores.get("prompt_adherence", 0) < 0.7:
            refinements.append("be more specific and detailed in the description")
            refinements.append("emphasize key elements from the original prompt")

        if avg_scores.get("aesthetics", 0) < 0.7:
            refinements.append("improve visual composition and color harmony")
            refinements.append("add professional quality details")

        if avg_scores.get("product_suitability", 0) < 0.7:
            refinements.append("optimize for product mockup printing")
            refinements.append("ensure clean edges and high contrast")

        # Modify primary prompt with refinements
        base_prompt = current_prompts.get("primary", "")
        refinement_text = ", ".join(refinements) if refinements else "higher quality, more detailed"

        refined_primary = f"{base_prompt}, {refinement_text}"

        # Increase temperature slightly for more variation
        temperature_boost = min(0.1 * attempt, 0.3)  # Max 0.3 boost

        # Return refined prompts
        refined_prompts = {
            "primary": refined_primary,
            "optimized": {
                ImageModel.FLUX.value: f"{refined_primary}, ultra sharp details",
                ImageModel.GEMINI.value: f"{refined_primary}, photorealistic perfection",
            },
            "style": current_prompts.get("style"),
            "refinement_iteration": attempt,
            "temperature_boost": temperature_boost,
        }

        logger.info(
            "Prompts refined",
            iteration=attempt,
            refinements_applied=len(refinements),
        )

        return refined_prompts
