"""
Integration tests for Phase 3 Hybrid Routing

Tests end-to-end flow through policy, planner, model selection,
and evaluation with hybrid routing enabled.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.agents.planner_agent import PlannerAgent
from src.agents.model_selection_agent import ModelSelectionAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.models.schemas import ReasoningModel, ImageModel


@pytest.mark.integration
class TestHybridRoutingFlow:
    """Test complete hybrid routing pipeline"""

    @pytest.mark.asyncio
    async def test_simple_product_uses_rule_based_path(self):
        """Test simple product request uses rule-based fast path throughout"""
        # Setup: Simple product request
        input_data = {
            "context": {
                "prompt": "Product photo of blue t-shirt on white background",
                "user_id": "test-user",
                "reasoning_model": ReasoningModel.GEMINI,
                "num_images": 2
            }
        }

        # Mock policy to enable hybrid mode
        with patch('src.agents.planner_agent.policy') as mock_policy, \
             patch('src.agents.planner_agent.GeminiReasoningEngine'), \
             patch('src.agents.planner_agent.ChatGPTReasoningEngine'):

            mock_policy.get_agent_mode.return_value = "hybrid"
            mock_policy.get.side_effect = lambda path, default: {
                "planner.rule_conditions.min_word_count": 8,
                "planner.rule_conditions.novelty_threshold": 0.35
            }.get(path, default)

            # Execute Planner Agent
            planner = PlannerAgent()
            planner_result = await planner.run(input_data)

            # Verify planner used rule-based path
            assert planner_result["success"] is True
            assert planner_result["routing_metadata"]["mode"] == "rule"
            assert planner_result["routing_metadata"]["used_llm"] is False
            assert len(planner_result["plan"]["steps"]) == 4

    @pytest.mark.asyncio
    async def test_complex_request_uses_llm_path(self):
        """Test complex request triggers LLM-based planning"""
        input_data = {
            "context": {
                "prompt": "Surreal dreamscape with floating islands, bioluminescent forests, and ethereal lighting",
                "user_id": "test-user",
                "reasoning_model": ReasoningModel.GEMINI
            }
        }

        with patch('src.agents.planner_agent.policy') as mock_policy, \
             patch('src.agents.planner_agent.GeminiReasoningEngine') as mock_gemini, \
             patch('src.agents.planner_agent.ChatGPTReasoningEngine'):

            mock_policy.get_agent_mode.return_value = "hybrid"
            mock_policy.get.side_effect = lambda path, default: {
                "planner.rule_conditions.min_word_count": 8,
                "planner.rule_conditions.novelty_threshold": 0.35
            }.get(path, default)

            planner = PlannerAgent()
            planner.gemini_engine.generate = AsyncMock(
                return_value="1. Analyze surreal requirements\n2. Select advanced model\n3. Generate image"
            )

            planner_result = await planner.run(input_data)

            # Verify planner used LLM path
            assert planner_result["success"] is True
            assert planner_result["routing_metadata"]["mode"] == "llm"
            assert planner_result["routing_metadata"]["used_llm"] is True

    @pytest.mark.asyncio
    async def test_ucb1_model_selection_integration(self, sample_model_stats):
        """Test UCB1 bandit integrates with model selection"""
        with patch('src.agents.model_selection_agent.Prisma'), \
             patch('src.agents.model_selection_agent.policy') as mock_policy:

            mock_policy.get_feature_flag.return_value = True  # Enable bandit

            agent = ModelSelectionAgent()
            agent.prisma = MagicMock()
            agent.prisma.connect = AsyncMock()
            agent.prisma.disconnect = AsyncMock()
            agent.prisma.modelstats.find_many = AsyncMock(return_value=[
                MagicMock(
                    id=stat["id"],
                    modelName=stat["modelName"],
                    bucket=stat["bucket"],
                    impressions=stat["impressions"],
                    rewardMean=stat["rewardMean"],
                    rewardVar=stat["rewardVar"]
                ) for stat in sample_model_stats
            ])

            input_data = {
                "context": {"prompt": "Product photo"},
                "prompts": {"primary": "Product photo on white background"}
            }

            result = await agent.run(input_data)

            # Verify UCB1 was used
            assert result["success"] is True
            assert "selected_models" in result
            assert result["selected_models"]["image"]["model"] in [ImageModel.FLUX, ImageModel.GEMINI]

    @pytest.mark.asyncio
    async def test_evaluation_hybrid_routing(self, sample_image_base64):
        """Test evaluation uses hybrid objective + vision routing"""
        with patch('src.agents.evaluation_agent.GeminiReasoningEngine'), \
             patch('src.agents.evaluation_agent.ChatGPTReasoningEngine'), \
             patch('src.agents.evaluation_agent.policy') as mock_policy:

            mock_policy.get.return_value = "hybrid"
            mock_policy.get_feature_flag.return_value = True

            agent = EvaluationAgent()

            # Mock objective checks to pass
            with patch.object(agent, '_calculate_coverage', return_value=0.75), \
                 patch.object(agent, '_calculate_background_whiteness', return_value=0.95), \
                 patch.object(agent.gemini_engine, 'generate_with_image', new_callable=AsyncMock) as mock_vision:

                mock_vision.return_value = "Aesthetics: 8.5/10\nSuitability: 9/10"

                image_data = {"base64": sample_image_base64}
                result = await agent._evaluate_image(
                    image_data, {}, {}, ReasoningModel.GEMINI
                )

                # Verify hybrid routing
                assert result["routing_metadata"]["objective_passed"] is True
                assert result["routing_metadata"]["used_vision"] is True

    @pytest.mark.asyncio
    async def test_policy_configuration_integration(self, sample_policy_config, tmp_path):
        """Test policy loader integrates with all agents"""
        import yaml
        from src.config.policy_loader import AgentPolicyConfig

        # Create temp policy file
        config_file = tmp_path / "test_policy.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_policy_config, f)

        policy = AgentPolicyConfig(config_path=str(config_file))

        # Verify policy values are accessible
        assert policy.get("planner.mode") == "hybrid"
        assert policy.get("model_selection.strategy") == "ucb1"
        assert policy.get("evaluation.mode") == "hybrid"
        assert policy.get_evaluation_threshold() == 0.75

    @pytest.mark.asyncio
    async def test_end_to_end_simple_flow(self, sample_image_base64):
        """Test simplified end-to-end flow for simple product request"""
        # This test simulates the entire pipeline but with mocked I/O

        # 1. Planner - Rule-based path
        with patch('src.agents.planner_agent.policy') as mock_policy, \
             patch('src.agents.planner_agent.GeminiReasoningEngine'), \
             patch('src.agents.planner_agent.ChatGPTReasoningEngine'):

            mock_policy.get_agent_mode.return_value = "hybrid"
            mock_policy.get.side_effect = lambda path, default: {
                "planner.rule_conditions.min_word_count": 8,
                "planner.rule_conditions.novelty_threshold": 0.35
            }.get(path, default)

            planner = PlannerAgent()
            planner_result = await planner.run({
                "context": {
                    "prompt": "Product photo of red mug on white background",
                    "reasoning_model": ReasoningModel.GEMINI
                }
            })

            assert planner_result["routing_metadata"]["mode"] == "rule"

        # 2. Model Selection - UCB1
        with patch('src.agents.model_selection_agent.Prisma'), \
             patch('src.agents.model_selection_agent.policy') as mock_policy:

            mock_policy.get_feature_flag.return_value = False  # Use rule-based fallback

            model_selector = ModelSelectionAgent()
            selection_result = await model_selector.run({
                "context": planner_result["context"],
                "prompts": {"primary": "Product photo"}
            })

            assert selection_result["success"] is True

        # 3. Evaluation - Hybrid (objective + vision)
        with patch('src.agents.evaluation_agent.GeminiReasoningEngine'), \
             patch('src.agents.evaluation_agent.ChatGPTReasoningEngine'), \
             patch('src.agents.evaluation_agent.policy') as mock_policy:

            mock_policy.get.return_value = "hybrid"
            mock_policy.get_feature_flag.return_value = True
            mock_policy.get_evaluation_threshold.return_value = 0.75

            evaluator = EvaluationAgent()

            with patch.object(evaluator, '_calculate_coverage', return_value=0.80), \
                 patch.object(evaluator, '_calculate_background_whiteness', return_value=0.95), \
                 patch.object(evaluator.gemini_engine, 'generate_with_image', new_callable=AsyncMock) as mock_vision:

                mock_vision.return_value = "Aesthetics: 9/10\nSuitability: 8.5/10"

                eval_result = await evaluator.run({
                    "images": [{"base64": sample_image_base64}],
                    "context": selection_result["context"],
                    "prompts": {"primary": "Product photo"}
                })

                assert eval_result["success"] is True
                assert len(eval_result["evaluations"]) > 0


@pytest.mark.integration
class TestRoutingMetadata:
    """Test routing metadata is tracked throughout pipeline"""

    @pytest.mark.asyncio
    async def test_planner_includes_routing_metadata(self):
        """Test planner includes routing metadata in output"""
        with patch('src.agents.planner_agent.policy') as mock_policy, \
             patch('src.agents.planner_agent.GeminiReasoningEngine'), \
             patch('src.agents.planner_agent.ChatGPTReasoningEngine'):

            mock_policy.get_agent_mode.return_value = "hybrid"
            mock_policy.get.side_effect = lambda path, default: {
                "planner.rule_conditions.min_word_count": 8,
                "planner.rule_conditions.novelty_threshold": 0.35
            }.get(path, default)

            planner = PlannerAgent()
            result = await planner.run({
                "context": {"prompt": "Product photo of blue t-shirt on white background"}
            })

            # Verify routing metadata structure
            assert "routing_metadata" in result
            assert "mode" in result["routing_metadata"]
            assert "used_llm" in result["routing_metadata"]
            assert "confidence" in result["routing_metadata"]
            assert "reasoning" in result["routing_metadata"]

    @pytest.mark.asyncio
    async def test_evaluation_includes_routing_metadata(self, sample_image_base64):
        """Test evaluation includes routing metadata"""
        with patch('src.agents.evaluation_agent.GeminiReasoningEngine'), \
             patch('src.agents.evaluation_agent.ChatGPTReasoningEngine'), \
             patch('src.agents.evaluation_agent.policy') as mock_policy:

            mock_policy.get.return_value = "hybrid"
            mock_policy.get_feature_flag.return_value = False  # Disable vision

            evaluator = EvaluationAgent()

            with patch.object(evaluator, '_calculate_coverage', return_value=0.75), \
                 patch.object(evaluator, '_calculate_background_whiteness', return_value=0.90):

                result = await evaluator._evaluate_image(
                    {"base64": sample_image_base64},
                    {}, {}, ReasoningModel.GEMINI
                )

                # Verify routing metadata
                assert "routing_metadata" in result
                assert "mode" in result["routing_metadata"]
                assert "used_vision" in result["routing_metadata"]
                assert "objective_passed" in result["routing_metadata"]


@pytest.mark.integration
class TestGracefulDegradation:
    """Test graceful degradation when services unavailable"""

    @pytest.mark.asyncio
    async def test_ucb1_falls_back_to_rules_on_db_error(self):
        """Test UCB1 falls back to rule-based on database error"""
        with patch('src.agents.model_selection_agent.Prisma'), \
             patch('src.agents.model_selection_agent.policy') as mock_policy:

            mock_policy.get_feature_flag.return_value = True  # Bandit enabled

            agent = ModelSelectionAgent()
            agent.prisma = MagicMock()
            agent.prisma.connect = AsyncMock(side_effect=Exception("DB Error"))

            result = await agent.run({
                "context": {},
                "prompts": {"primary": "Product photo"}
            })

            # Should succeed with fallback
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_vision_falls_back_to_objective_on_error(self, sample_image_base64):
        """Test vision evaluation falls back to objective on LLM error"""
        with patch('src.agents.evaluation_agent.GeminiReasoningEngine'), \
             patch('src.agents.evaluation_agent.ChatGPTReasoningEngine'), \
             patch('src.agents.evaluation_agent.policy') as mock_policy:

            mock_policy.get.return_value = "hybrid"
            mock_policy.get_feature_flag.return_value = True

            evaluator = EvaluationAgent()
            evaluator.gemini_engine.generate_with_image = AsyncMock(
                side_effect=Exception("Vision API Error")
            )

            with patch.object(evaluator, '_calculate_coverage', return_value=0.75), \
                 patch.object(evaluator, '_calculate_background_whiteness', return_value=0.90):

                result = await evaluator._evaluate_image(
                    {"base64": sample_image_base64},
                    {}, {}, ReasoningModel.GEMINI
                )

                # Should succeed with objective only
                assert "overall_score" in result
                assert result["routing_metadata"]["used_vision"] is True
                assert "vision_error" in result["routing_metadata"]
