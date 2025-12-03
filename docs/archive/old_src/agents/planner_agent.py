"""
Planner Agent - Task orchestration and execution planning.

Uses Google ADK SequentialAgent for workflow management.
Implements hybrid routing: rule-based fast path for simple requests,
LLM-based planning for complex requests.
"""
from typing import Any, Dict, List
from src.agents.base_agent import BaseAgent, SequentialAgent
from src.connectors.gemini_reasoning import GeminiReasoningEngine
from src.connectors.chatgpt_reasoning import ChatGPTReasoningEngine
from src.config.policy_loader import policy
from src.utils import get_logger
from src.models.schemas import ReasoningModel

logger = get_logger(__name__)


class PlannerAgent(BaseAgent):
    """
    Planner Agent orchestrates the execution workflow.

    Responsibilities:
    - Analyze user prompt
    - Create execution plan with steps
    - Estimate costs and time
    - Coordinate Sequential execution of downstream agents
    """

    def __init__(self, name: str = "PlannerAgent") -> None:
        """Initialize Planner Agent."""
        super().__init__(name=name)

        # Initialize reasoning engines
        self.gemini_engine = GeminiReasoningEngine()
        self.chatgpt_engine = ChatGPTReasoningEngine()

        logger.info("Planner Agent initialized with reasoning engines")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create execution plan for the user request.

        Uses hybrid routing:
        - Rule-based fast path for simple product photo requests (<1ms)
        - LLM-based planning for complex/ambiguous requests (~200ms)

        Args:
            input_data: Context from Interactive Agent containing:
                - prompt: User prompt
                - user_id: User identifier
                - reasoning_model: Model preference
                - image_model: Image model preference

        Returns:
            Execution plan with steps, costs, timeline, and routing metadata
        """
        context = input_data.get("context", {})
        prompt = context.get("prompt", "")
        reasoning_model = context.get("reasoning_model", ReasoningModel.GEMINI)

        # Convert string to enum if needed
        if isinstance(reasoning_model, str):
            reasoning_model = ReasoningModel(reasoning_model)

        logger.info(
            "Planner Agent creating execution plan",
            prompt_length=len(prompt),
            reasoning_model=reasoning_model,
        )

        # Check routing mode from policy
        mode = policy.get_agent_mode("planner")

        # Determine if we should use rule-based or LLM planning
        if mode == "hybrid":
            use_rules = self._should_use_rules(prompt, context)
            if use_rules:
                logger.info("Using rule-based fast path")
                return await self._rule_based_plan(prompt, context, reasoning_model)
            else:
                logger.info("Using LLM-based planning")
                return await self._llm_based_plan(prompt, context, reasoning_model)
        elif mode == "rule":
            logger.info("Forcing rule-based planning (config mode=rule)")
            return await self._rule_based_plan(prompt, context, reasoning_model)
        else:
            logger.info("Forcing LLM-based planning (config mode=llm)")
            return await self._llm_based_plan(prompt, context, reasoning_model)

    async def _llm_based_plan(
        self,
        prompt: str,
        context: Dict[str, Any],
        reasoning_model: ReasoningModel
    ) -> Dict[str, Any]:
        """
        LLM-powered planning for complex requests.

        Args:
            prompt: User prompt
            context: Execution context
            reasoning_model: Reasoning model to use

        Returns:
            Execution plan with metadata
        """
        # Select reasoning engine
        engine = (
            self.gemini_engine
            if reasoning_model == ReasoningModel.GEMINI
            else self.chatgpt_engine
        )

        # Create planning prompt
        planning_prompt = self._create_planning_prompt(prompt)

        # Generate plan using reasoning model
        try:
            llm_response = await engine.generate(
                prompt=planning_prompt,
                system_prompt=self._get_system_prompt(),
                temperature=0.3,  # Lower temperature for structured planning
            )

            # Parse plan into structured steps (extract text from LLMResponse)
            steps = self._parse_plan(llm_response.text)

            # Estimate costs and timing
            total_cost = await self._estimate_total_cost(steps, context)
            total_time = self._estimate_total_time(steps)

            logger.info(
                "Execution plan created (LLM-based)",
                num_steps=len(steps),
                estimated_cost=total_cost,
                estimated_time=total_time,
                tokens_used=llm_response.tokens_used,
                model_name=llm_response.model_name,
            )

            return {
                "success": True,
                "plan": {
                    "steps": steps,
                    "total_estimated_cost": total_cost,
                    "total_estimated_time": total_time,
                    "reasoning_model_used": reasoning_model.value,
                },
                "routing_metadata": {
                    "mode": "llm",
                    "used_llm": True,
                    "confidence": 1.0,
                    "reasoning": llm_response.reasoning or "Complex request requires LLM planning",
                    "tokens": llm_response.tokens_used,
                    "model_name": llm_response.model_name,
                },
                "context": context,  # Pass context forward
            }

        except Exception as e:
            logger.error("Planning failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": f"Planning failed: {str(e)}",
                "routing_metadata": {
                    "mode": "llm",
                    "used_llm": True,
                    "error": str(e)
                },
                "context": context,
            }

    def _should_use_rules(self, prompt: str, context: Dict[str, Any]) -> bool:
        """
        Determine if request qualifies for rule-based fast path.

        Criteria for rule-based path:
        - Prompt has sufficient length (>= min_word_count)
        - Appears to be product-related
        - Mentions background type
        - No composition operators (multi-object scenes)
        - Low novelty score (matches known patterns)

        Args:
            prompt: User prompt
            context: Execution context

        Returns:
            True if should use rule-based planning
        """
        prompt_lower = prompt.lower()

        # Get thresholds from policy
        min_words = policy.get("planner.rule_conditions.min_word_count", 8)
        max_objects = policy.get("planner.rule_conditions.max_objects", 1)
        novelty_threshold = policy.get("planner.rule_conditions.novelty_threshold", 0.35)

        # Check word count
        word_count = len(prompt.split())
        if word_count < min_words:
            logger.debug(f"Word count {word_count} < {min_words}, using LLM")
            return False

        # Check for product keywords
        product_keywords = [
            "product", "photo", "image", "picture", "t-shirt", "tshirt",
            "mug", "cup", "bottle", "bag", "poster", "canvas", "phone case",
            "hoodie", "sweatshirt", "hat", "cap"
        ]
        is_product = any(kw in prompt_lower for kw in product_keywords)

        # Check for background keywords
        background_keywords = [
            "white background", "white bg", "transparent", "isolated",
            "clean background", "studio", "plain background"
        ]
        has_background = any(kw in prompt_lower for kw in background_keywords)

        # Check for composition operators (indicates multi-object/complex scene)
        composition_ops = [
            " and ", " with ", " plus ", " & ", " alongside ",
            "multiple", "several", "combination", "together"
        ]
        has_composition = any(op in prompt_lower for op in composition_ops)

        # Simple novelty check (could be enhanced with embeddings)
        novelty_score = self._calculate_novelty(prompt)

        # Decision logic
        use_rules = (
            is_product and
            has_background and
            not has_composition and
            novelty_score <= novelty_threshold
        )

        logger.debug(
            f"Rule routing decision: {use_rules}",
            word_count=word_count,
            is_product=is_product,
            has_background=has_background,
            has_composition=has_composition,
            novelty=novelty_score
        )

        return use_rules

    def _calculate_novelty(self, prompt: str) -> float:
        """
        Calculate novelty score for prompt.

        Simple version: checks if prompt contains unusual words or patterns.
        Could be enhanced with TF-IDF or embedding distance.

        Args:
            prompt: User prompt

        Returns:
            Novelty score (0.0-1.0), where higher = more novel
        """
        prompt_lower = prompt.lower()

        # Known common patterns
        common_patterns = [
            "product photo", "white background", "professional",
            "high quality", "studio lighting", "commercial",
            "t-shirt", "mug", "poster", "design"
        ]

        # Count matches
        matches = sum(1 for pattern in common_patterns if pattern in prompt_lower)

        # Novelty is inverse of familiarity
        familiarity = min(matches / 5.0, 1.0)  # Normalize to 0-1
        novelty = 1.0 - familiarity

        return novelty

    async def _rule_based_plan(
        self,
        prompt: str,
        context: Dict[str, Any],
        reasoning_model: ReasoningModel
    ) -> Dict[str, Any]:
        """
        Rule-based planning for standard product photo requests.

        Creates fixed execution plan without LLM:
        1. Prompt Manager - refine and select template
        2. Model Selection - choose best AI model
        3. Generator - create images
        4. Evaluation - score quality

        Args:
            prompt: User prompt
            context: Execution context
            reasoning_model: Reasoning model (for metadata only)

        Returns:
            Execution plan with metadata
        """
        # Fixed steps for standard workflow
        steps = [
            {
                "step_number": 1,
                "action": "Refine prompt and select template",
                "description": "Use database templates to enhance prompt",
                "estimated_time": 0.01,  # Database query is fast
                "estimated_cost": 0.0,
                "agent": "prompt_manager"
            },
            {
                "step_number": 2,
                "action": "Select AI model",
                "description": "Choose best model based on requirements",
                "estimated_time": 0.005,  # DB query + priority ranking
                "estimated_cost": 0.0,
                "agent": "model_selection"
            },
            {
                "step_number": 3,
                "action": "Generate images",
                "description": "Create product images using selected model",
                "estimated_time": 3.5,  # Typical generation time
                "estimated_cost": 0.005,  # Model-dependent
                "agent": "generator"
            },
            {
                "step_number": 4,
                "action": "Evaluate quality",
                "description": "Score images on quality metrics",
                "estimated_time": 0.5,  # Objective checks
                "estimated_cost": 0.0,
                "agent": "evaluation"
            }
        ]

        # Calculate totals
        total_cost = await self._estimate_total_cost(steps, context)
        total_time = self._estimate_total_time(steps)

        logger.info(
            "Execution plan created (rule-based)",
            num_steps=len(steps),
            estimated_cost=total_cost,
            estimated_time=total_time,
        )

        return {
            "success": True,
            "plan": {
                "steps": steps,
                "total_estimated_cost": total_cost,
                "total_estimated_time": total_time,
                "reasoning_model_used": reasoning_model.value,  # For metadata
            },
            "routing_metadata": {
                "mode": "rule",
                "used_llm": False,
                "confidence": 1.0,
                "reasoning": "Simple product photo request using rule-based fast path"
            },
            "context": context,
        }

    def _create_planning_prompt(self, user_prompt: str) -> str:
        """
        Create a planning prompt for the reasoning model.

        Args:
            user_prompt: Original user prompt

        Returns:
            Planning prompt
        """
        return f"""
Analyze the following user request and create a detailed execution plan:

USER REQUEST:
{user_prompt}

Create a step-by-step plan that includes:
1. Prompt refinement and style definition
2. Model selection (reasoning and image generation)
3. Image generation with appropriate parameters
4. Quality evaluation and approval
5. Product compositing (if needed)

For each step, specify:
- Action to take
- Expected outcome
- Estimated time
- Risk factors

Format your response as a numbered list with clear, actionable steps.
"""

    def _get_system_prompt(self) -> str:
        """Get system prompt for planner."""
        return """You are an expert AI task planner for image generation workflows.
Your goal is to break down user requests into clear, sequential steps.
Consider image quality, style consistency, and production requirements.
Be specific about model choices and generation parameters."""

    def _parse_plan(self, plan_text: str) -> List[Dict[str, Any]]:
        """
        Parse plan text into structured steps.

        Args:
            plan_text: Raw plan from reasoning model

        Returns:
            List of structured plan steps
        """
        steps = []

        # Split by lines and look for numbered items
        lines = plan_text.strip().split("\n")
        current_step = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for numbered step (e.g., "1.", "1)", "Step 1:")
            if any(
                line.startswith(f"{i}.") or line.startswith(f"{i})") or line.lower().startswith(f"step {i}")
                for i in range(1, 20)
            ):
                if current_step:
                    steps.append(current_step)

                current_step = {
                    "step_number": len(steps) + 1,
                    "action": line,
                    "description": "",
                    "estimated_time": 5.0,  # Default 5 seconds
                    "estimated_cost": 0.0,
                }
            elif current_step:
                # Add detail to current step
                current_step["description"] += " " + line

        # Add final step
        if current_step:
            steps.append(current_step)

        # If no steps parsed, create default workflow
        if not steps:
            steps = [
                {
                    "step_number": 1,
                    "action": "Refine prompt and select models",
                    "description": "Analyze user intent and choose appropriate models",
                    "estimated_time": 2.0,
                    "estimated_cost": 0.001,
                },
                {
                    "step_number": 2,
                    "action": "Generate images",
                    "description": "Create images using selected model",
                    "estimated_time": 10.0,
                    "estimated_cost": 0.055,
                },
                {
                    "step_number": 3,
                    "action": "Evaluate image quality",
                    "description": "Score images on quality and adherence",
                    "estimated_time": 5.0,
                    "estimated_cost": 0.002,
                },
                {
                    "step_number": 4,
                    "action": "Create product mockups",
                    "description": "Composite onto product templates",
                    "estimated_time": 8.0,
                    "estimated_cost": 0.01,
                },
            ]

        return steps

    async def _estimate_total_cost(self, steps: List[Dict[str, Any]], context: Dict[str, Any]) -> float:
        """
        Estimate total cost for plan execution.

        Args:
            steps: Plan steps
            context: Execution context

        Returns:
            Estimated cost in USD
        """
        total = 0.0

        # Get actual number of images from context (not hardcoded)
        num_images = context.get("num_images", 2)

        # Add reasoning cost - more accurate estimation
        # Planning: 1 call, Evaluation: num_images calls (per-image), Refinement: 1 call
        num_reasoning_calls = 2 + num_images  # Dynamic based on images to evaluate

        reasoning_model = context.get("reasoning_model", ReasoningModel.GEMINI)
        if reasoning_model == ReasoningModel.GEMINI:
            total += 0.0001 * num_reasoning_calls  # Gemini 2.0 Flash per call
        else:
            total += 0.005 * num_reasoning_calls  # GPT-4o per call

        # Add image generation cost - use actual num_images from context
        from config import settings
        image_model = context.get("image_model", "flux")
        if image_model == "flux":
            total += settings.flux_pro_cost * num_images  # Dynamic (using Pro tier as default)
        else:
            total += settings.gemini_image_cost * num_images  # Dynamic

        # Add evaluation cost (scales with images)
        total += 0.001 * num_images

        # Add product generation cost (typically constant regardless of num_images)
        total += 0.01

        return round(total, 4)

    def _estimate_total_time(self, steps: List[Dict[str, Any]]) -> float:
        """
        Estimate total execution time.

        Args:
            steps: Plan steps

        Returns:
            Estimated time in seconds
        """
        return sum(step.get("estimated_time", 5.0) for step in steps)
