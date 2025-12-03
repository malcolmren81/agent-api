"""
Unit tests for Planner Agent

Tests hybrid routing logic (rule-based vs LLM-based planning),
novelty detection, and execution plan creation.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.agents.planner_agent import PlannerAgent
from src.models.schemas import ReasoningModel


class TestPlannerHybridRouting:
    """Test hybrid routing decision logic"""

    @pytest.fixture
    def planner(self):
        """Create Planner Agent instance"""
        with patch('src.agents.planner_agent.GeminiReasoningEngine'), \
             patch('src.agents.planner_agent.ChatGPTReasoningEngine'):
            return PlannerAgent()

    @pytest.mark.asyncio
    async def test_simple_product_uses_rule_based(self, planner, sample_prompts, mock_policy_loader):
        """Test that simple product prompts use rule-based fast path"""
        with patch.object(planner, 'policy', mock_policy_loader), \
             patch.object(planner, '_rule_based_plan', new_callable=AsyncMock) as mock_rule:

            mock_rule.return_value = {"success": True, "plan": {}}

            input_data = {
                "context": {
                    "prompt": sample_prompts["simple_product"],
                    "reasoning_model": ReasoningModel.GEMINI
                }
            }

            await planner.run(input_data)

            # Should call rule-based plan
            mock_rule.assert_called_once()

    @pytest.mark.asyncio
    async def test_complex_scene_uses_llm(self, planner, sample_prompts, mock_policy_loader):
        """Test that complex scenes use LLM-based planning"""
        with patch.object(planner, 'policy', mock_policy_loader), \
             patch.object(planner, '_llm_based_plan', new_callable=AsyncMock) as mock_llm:

            mock_llm.return_value = {"success": True, "plan": {}}

            input_data = {
                "context": {
                    "prompt": sample_prompts["complex_scene"],
                    "reasoning_model": ReasoningModel.GEMINI
                }
            }

            await planner.run(input_data)

            # Should call LLM-based plan
            mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_rule_mode(self, planner, sample_prompts):
        """Test forcing rule-based mode via policy"""
        mock_policy = MagicMock()
        mock_policy.get_agent_mode.return_value = "rule"

        with patch('src.agents.planner_agent.policy', mock_policy), \
             patch.object(planner, '_rule_based_plan', new_callable=AsyncMock) as mock_rule:

            mock_rule.return_value = {"success": True}

            input_data = {
                "context": {
                    "prompt": sample_prompts["complex_scene"],  # Even complex uses rule
                    "reasoning_model": ReasoningModel.GEMINI
                }
            }

            await planner.run(input_data)

            mock_rule.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_llm_mode(self, planner, sample_prompts):
        """Test forcing LLM mode via policy"""
        mock_policy = MagicMock()
        mock_policy.get_agent_mode.return_value = "llm"

        with patch('src.agents.planner_agent.policy', mock_policy), \
             patch.object(planner, '_llm_based_plan', new_callable=AsyncMock) as mock_llm:

            mock_llm.return_value = {"success": True}

            input_data = {
                "context": {
                    "prompt": sample_prompts["simple_product"],  # Even simple uses LLM
                    "reasoning_model": ReasoningModel.GEMINI
                }
            }

            await planner.run(input_data)

            mock_llm.assert_called_once()


class TestShouldUseRules:
    """Test routing decision logic"""

    @pytest.fixture
    def planner(self):
        """Create Planner Agent instance"""
        with patch('src.agents.planner_agent.GeminiReasoningEngine'), \
             patch('src.agents.planner_agent.ChatGPTReasoningEngine'):
            return PlannerAgent()

    def test_simple_product_qualifies_for_rules(self, planner, sample_prompts):
        """Test simple product prompts qualify for rule-based routing"""
        with patch('src.agents.planner_agent.policy') as mock_policy:
            mock_policy.get.side_effect = lambda path, default: {
                "planner.rule_conditions.min_word_count": 8,
                "planner.rule_conditions.max_objects": 1,
                "planner.rule_conditions.novelty_threshold": 0.35
            }.get(path, default)

            result = planner._should_use_rules(
                sample_prompts["simple_product"],
                {}
            )

            assert result is True

    def test_complex_scene_requires_llm(self, planner, sample_prompts):
        """Test complex scenes don't qualify for rules"""
        with patch('src.agents.planner_agent.policy') as mock_policy:
            mock_policy.get.side_effect = lambda path, default: {
                "planner.rule_conditions.min_word_count": 8,
                "planner.rule_conditions.max_objects": 1,
                "planner.rule_conditions.novelty_threshold": 0.35
            }.get(path, default)

            result = planner._should_use_rules(
                sample_prompts["complex_scene"],
                {}
            )

            assert result is False

    def test_minimal_prompt_fails_word_count(self, planner, sample_prompts):
        """Test that prompts below min word count use LLM"""
        with patch('src.agents.planner_agent.policy') as mock_policy:
            mock_policy.get.side_effect = lambda path, default: {
                "planner.rule_conditions.min_word_count": 8,
                "planner.rule_conditions.max_objects": 1,
                "planner.rule_conditions.novelty_threshold": 0.35
            }.get(path, default)

            result = planner._should_use_rules(
                sample_prompts["minimal_prompt"],  # "blue shirt"
                {}
            )

            assert result is False

    def test_non_product_requires_llm(self, planner):
        """Test non-product prompts use LLM"""
        with patch('src.agents.planner_agent.policy') as mock_policy:
            mock_policy.get.side_effect = lambda path, default: {
                "planner.rule_conditions.min_word_count": 8,
                "planner.rule_conditions.max_objects": 1,
                "planner.rule_conditions.novelty_threshold": 0.35
            }.get(path, default)

            result = planner._should_use_rules(
                "A beautiful sunset over the ocean with birds flying",
                {}
            )

            # No product keywords, should use LLM
            assert result is False

    def test_composition_operators_require_llm(self, planner):
        """Test prompts with composition operators use LLM"""
        with patch('src.agents.planner_agent.policy') as mock_policy:
            mock_policy.get.side_effect = lambda path, default: {
                "planner.rule_conditions.min_word_count": 8,
                "planner.rule_conditions.max_objects": 1,
                "planner.rule_conditions.novelty_threshold": 0.35
            }.get(path, default)

            result = planner._should_use_rules(
                "Product photo of blue t-shirt and red mug on white background",
                {}
            )

            # Has "and" composition operator
            assert result is False

    def test_high_novelty_requires_llm(self, planner, sample_prompts):
        """Test novel prompts use LLM"""
        with patch('src.agents.planner_agent.policy') as mock_policy:
            mock_policy.get.side_effect = lambda path, default: {
                "planner.rule_conditions.min_word_count": 8,
                "planner.rule_conditions.max_objects": 1,
                "planner.rule_conditions.novelty_threshold": 0.35
            }.get(path, default)

            result = planner._should_use_rules(
                sample_prompts["novel_request"],  # Quantum computing viz
                {}
            )

            # Novel request should have high novelty score
            assert result is False


class TestNoveltyCalculation:
    """Test novelty score calculation"""

    @pytest.fixture
    def planner(self):
        """Create Planner Agent instance"""
        with patch('src.agents.planner_agent.GeminiReasoningEngine'), \
             patch('src.agents.planner_agent.ChatGPTReasoningEngine'):
            return PlannerAgent()

    def test_common_prompt_low_novelty(self, planner):
        """Test common prompts have low novelty score"""
        prompt = "Professional product photo of t-shirt on white background with studio lighting"

        novelty = planner._calculate_novelty(prompt)

        # Should have low novelty (high familiarity)
        assert 0.0 <= novelty <= 0.3

    def test_novel_prompt_high_novelty(self, planner):
        """Test novel prompts have high novelty score"""
        prompt = "Quantum entanglement visualization with holographic nebula"

        novelty = planner._calculate_novelty(prompt)

        # Should have high novelty (low familiarity)
        assert 0.7 <= novelty <= 1.0

    def test_novelty_bounded(self, planner):
        """Test novelty score is always between 0 and 1"""
        prompts = [
            "product photo white background",
            "t-shirt mug poster design",
            "abstract surreal quantum visualization",
            "blue shirt"
        ]

        for prompt in prompts:
            novelty = planner._calculate_novelty(prompt)
            assert 0.0 <= novelty <= 1.0


class TestRuleBasedPlan:
    """Test rule-based planning"""

    @pytest.fixture
    def planner(self):
        """Create Planner Agent instance"""
        with patch('src.agents.planner_agent.GeminiReasoningEngine'), \
             patch('src.agents.planner_agent.ChatGPTReasoningEngine'):
            return PlannerAgent()

    @pytest.mark.asyncio
    async def test_creates_fixed_workflow(self, planner):
        """Test rule-based plan creates fixed 4-step workflow"""
        with patch.object(planner, '_estimate_total_cost', new_callable=AsyncMock) as mock_cost:
            mock_cost.return_value = 0.005

            result = await planner._rule_based_plan(
                "Product photo of blue t-shirt",
                {},
                ReasoningModel.GEMINI
            )

            assert result["success"] is True
            assert len(result["plan"]["steps"]) == 4
            assert result["routing_metadata"]["mode"] == "rule"
            assert result["routing_metadata"]["used_llm"] is False

    @pytest.mark.asyncio
    async def test_step_order_correct(self, planner):
        """Test steps are in correct order"""
        with patch.object(planner, '_estimate_total_cost', new_callable=AsyncMock) as mock_cost:
            mock_cost.return_value = 0.005

            result = await planner._rule_based_plan(
                "Product photo",
                {},
                ReasoningModel.GEMINI
            )

            steps = result["plan"]["steps"]
            assert "prompt" in steps[0]["action"].lower() or "refine" in steps[0]["action"].lower()
            assert "model" in steps[1]["action"].lower() or "select" in steps[1]["action"].lower()
            assert "generate" in steps[2]["action"].lower()
            assert "evaluate" in steps[3]["action"].lower() or "quality" in steps[3]["action"].lower()

    @pytest.mark.asyncio
    async def test_includes_metadata(self, planner):
        """Test includes all required metadata"""
        with patch.object(planner, '_estimate_total_cost', new_callable=AsyncMock) as mock_cost, \
             patch.object(planner, '_estimate_total_time') as mock_time:
            mock_cost.return_value = 0.005
            mock_time.return_value = 4.0

            result = await planner._rule_based_plan(
                "Product photo",
                {"user_id": "test-123"},
                ReasoningModel.GEMINI
            )

            # Check all required fields
            assert "success" in result
            assert "plan" in result
            assert "routing_metadata" in result
            assert "context" in result

            assert "steps" in result["plan"]
            assert "total_estimated_cost" in result["plan"]
            assert "total_estimated_time" in result["plan"]


class TestLLMBasedPlan:
    """Test LLM-based planning"""

    @pytest.fixture
    def planner(self):
        """Create Planner Agent instance"""
        with patch('src.agents.planner_agent.GeminiReasoningEngine') as mock_gemini, \
             patch('src.agents.planner_agent.ChatGPTReasoningEngine') as mock_chatgpt:

            planner = PlannerAgent()

            # Mock engine responses
            planner.gemini_engine.generate = AsyncMock(return_value="1. Step one\n2. Step two\n3. Step three")
            planner.chatgpt_engine.generate = AsyncMock(return_value="1. Step one\n2. Step two\n3. Step three")

            return planner

    @pytest.mark.asyncio
    async def test_uses_gemini_engine(self, planner):
        """Test uses Gemini engine when specified"""
        with patch.object(planner, '_estimate_total_cost', new_callable=AsyncMock) as mock_cost:
            mock_cost.return_value = 0.015

            result = await planner._llm_based_plan(
                "Complex scene",
                {},
                ReasoningModel.GEMINI
            )

            planner.gemini_engine.generate.assert_called_once()
            planner.chatgpt_engine.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_chatgpt_engine(self, planner):
        """Test uses ChatGPT engine when specified"""
        with patch.object(planner, '_estimate_total_cost', new_callable=AsyncMock) as mock_cost:
            mock_cost.return_value = 0.015

            result = await planner._llm_based_plan(
                "Complex scene",
                {},
                ReasoningModel.CHATGPT
            )

            planner.chatgpt_engine.generate.assert_called_once()
            planner.gemini_engine.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_parses_llm_response(self, planner):
        """Test parses LLM response into steps"""
        with patch.object(planner, '_estimate_total_cost', new_callable=AsyncMock) as mock_cost:
            mock_cost.return_value = 0.015

            result = await planner._llm_based_plan(
                "Complex scene",
                {},
                ReasoningModel.GEMINI
            )

            assert result["success"] is True
            assert len(result["plan"]["steps"]) >= 3
            assert result["routing_metadata"]["mode"] == "llm"
            assert result["routing_metadata"]["used_llm"] is True

    @pytest.mark.asyncio
    async def test_handles_generation_error(self, planner):
        """Test handles LLM generation errors gracefully"""
        planner.gemini_engine.generate.side_effect = Exception("API Error")

        result = await planner._llm_based_plan(
            "Complex scene",
            {},
            ReasoningModel.GEMINI
        )

        assert result["success"] is False
        assert "error" in result
        assert "routing_metadata" in result
        assert result["routing_metadata"]["used_llm"] is True


class TestPlanParsing:
    """Test plan text parsing"""

    @pytest.fixture
    def planner(self):
        """Create Planner Agent instance"""
        with patch('src.agents.planner_agent.GeminiReasoningEngine'), \
             patch('src.agents.planner_agent.ChatGPTReasoningEngine'):
            return PlannerAgent()

    def test_parse_numbered_list(self, planner):
        """Test parsing numbered list format"""
        plan_text = """
        1. Refine the prompt
        2. Select best model
        3. Generate images
        4. Evaluate quality
        """

        steps = planner._parse_plan(plan_text)

        assert len(steps) >= 4
        assert all("step_number" in step for step in steps)
        assert all("action" in step for step in steps)

    def test_parse_step_format(self, planner):
        """Test parsing "Step X:" format"""
        plan_text = """
        Step 1: Analyze user request
        Step 2: Choose model
        Step 3: Generate output
        """

        steps = planner._parse_plan(plan_text)

        assert len(steps) >= 3

    def test_parse_empty_returns_default(self, planner):
        """Test empty plan returns default workflow"""
        plan_text = ""

        steps = planner._parse_plan(plan_text)

        # Should return default 4-step workflow
        assert len(steps) == 4

    def test_parse_malformed_returns_default(self, planner):
        """Test malformed plan returns default workflow"""
        plan_text = "This is not a numbered list at all, just random text"

        steps = planner._parse_plan(plan_text)

        # Should return default workflow
        assert len(steps) == 4


class TestCostEstimation:
    """Test cost estimation"""

    @pytest.fixture
    def planner(self):
        """Create Planner Agent instance"""
        with patch('src.agents.planner_agent.GeminiReasoningEngine'), \
             patch('src.agents.planner_agent.ChatGPTReasoningEngine'):
            return PlannerAgent()

    @pytest.mark.asyncio
    async def test_estimates_gemini_cost(self, planner):
        """Test cost estimation for Gemini"""
        steps = [{"estimated_cost": 0.001}]
        context = {
            "reasoning_model": ReasoningModel.GEMINI,
            "image_model": "flux",
            "num_images": 2
        }

        with patch('config.settings') as mock_settings:
            mock_settings.flux_pro_cost = 0.05
            mock_settings.gemini_image_cost = 0.01

            cost = await planner._estimate_total_cost(steps, context)

            assert isinstance(cost, float)
            assert cost > 0

    @pytest.mark.asyncio
    async def test_cost_scales_with_images(self, planner):
        """Test cost scales with number of images"""
        steps = [{"estimated_cost": 0.001}]

        with patch('config.settings') as mock_settings:
            mock_settings.flux_pro_cost = 0.05

            cost_2_images = await planner._estimate_total_cost(steps, {
                "reasoning_model": ReasoningModel.GEMINI,
                "image_model": "flux",
                "num_images": 2
            })

            cost_5_images = await planner._estimate_total_cost(steps, {
                "reasoning_model": ReasoningModel.GEMINI,
                "image_model": "flux",
                "num_images": 5
            })

            assert cost_5_images > cost_2_images

    def test_time_estimation(self, planner):
        """Test time estimation sums step times"""
        steps = [
            {"estimated_time": 1.0},
            {"estimated_time": 2.5},
            {"estimated_time": 3.0},
        ]

        total_time = planner._estimate_total_time(steps)

        assert total_time == 6.5


class TestPromptCreation:
    """Test prompt creation for LLM planning"""

    @pytest.fixture
    def planner(self):
        """Create Planner Agent instance"""
        with patch('src.agents.planner_agent.GeminiReasoningEngine'), \
             patch('src.agents.planner_agent.ChatGPTReasoningEngine'):
            return PlannerAgent()

    def test_includes_user_prompt(self, planner):
        """Test planning prompt includes user request"""
        user_prompt = "Create a surreal landscape"

        planning_prompt = planner._create_planning_prompt(user_prompt)

        assert user_prompt in planning_prompt
        assert "USER REQUEST" in planning_prompt

    def test_includes_instructions(self, planner):
        """Test planning prompt includes instructions"""
        planning_prompt = planner._create_planning_prompt("Test prompt")

        assert "step-by-step" in planning_prompt.lower()
        assert "plan" in planning_prompt.lower()

    def test_system_prompt(self, planner):
        """Test system prompt is appropriate"""
        system_prompt = planner._get_system_prompt()

        assert "planner" in system_prompt.lower()
        assert "image" in system_prompt.lower()
