"""
Unit tests for Evaluation Agent

Tests hybrid evaluation (objective + vision LLM), score combination,
and routing logic.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.agents.evaluation_agent import EvaluationAgent
from src.models.schemas import ReasoningModel


class TestHybridEvaluation:
    """Test hybrid evaluation routing"""

    @pytest.fixture
    def agent(self):
        """Create Evaluation Agent"""
        with patch('src.agents.evaluation_agent.GeminiReasoningEngine'), \
             patch('src.agents.evaluation_agent.ChatGPTReasoningEngine'):
            return EvaluationAgent()

    @pytest.mark.asyncio
    async def test_hybrid_uses_vision_when_objective_passes(self, agent, sample_image_base64):
        """Test hybrid mode uses vision when objective checks pass"""
        with patch.object(agent, '_objective_checks', new_callable=AsyncMock) as mock_obj, \
             patch.object(agent, '_vision_evaluation', new_callable=AsyncMock) as mock_vision, \
             patch('src.agents.evaluation_agent.policy') as mock_policy:

            mock_policy.get.return_value = "hybrid"
            mock_policy.get_feature_flag.return_value = True

            mock_obj.return_value = ({"coverage": 0.80}, True)  # Passed
            mock_vision.return_value = ({"aesthetics": 0.85}, "Good image")

            image_data = {"base64": sample_image_base64}
            result = await agent._evaluate_image(
                image_data, {}, {}, ReasoningModel.GEMINI
            )

            # Should call vision evaluation
            mock_vision.assert_called_once()
            assert result["routing_metadata"]["used_vision"] is True

    @pytest.mark.asyncio
    async def test_hybrid_skips_vision_when_objective_fails(self, agent, sample_image_base64):
        """Test hybrid mode skips vision when objective checks fail"""
        with patch.object(agent, '_objective_checks', new_callable=AsyncMock) as mock_obj, \
             patch.object(agent, '_vision_evaluation', new_callable=AsyncMock) as mock_vision, \
             patch('src.agents.evaluation_agent.policy') as mock_policy:

            mock_policy.get.return_value = "hybrid"
            mock_policy.get_feature_flag.return_value = True

            mock_obj.return_value = ({"coverage": 0.02}, False)  # Failed

            image_data = {"base64": sample_image_base64}
            result = await agent._evaluate_image(
                image_data, {}, {}, ReasoningModel.GEMINI
            )

            # Should NOT call vision evaluation
            mock_vision.assert_not_called()
            assert result["routing_metadata"]["used_vision"] is False

    @pytest.mark.asyncio
    async def test_objective_only_mode(self, agent, sample_image_base64):
        """Test objective-only mode skips vision"""
        with patch.object(agent, '_objective_checks', new_callable=AsyncMock) as mock_obj, \
             patch.object(agent, '_vision_evaluation', new_callable=AsyncMock) as mock_vision, \
             patch('src.agents.evaluation_agent.policy') as mock_policy:

            mock_policy.get.return_value = "objective"

            mock_obj.return_value = ({"coverage": 0.80}, True)

            image_data = {"base64": sample_image_base64}
            await agent._evaluate_image(
                image_data, {}, {}, ReasoningModel.GEMINI
            )

            # Should NOT call vision
            mock_vision.assert_not_called()

    @pytest.mark.asyncio
    async def test_vision_mode_forces_vision(self, agent, sample_image_base64):
        """Test vision mode forces vision even if objective fails"""
        with patch.object(agent, '_objective_checks', new_callable=AsyncMock) as mock_obj, \
             patch.object(agent, '_vision_evaluation', new_callable=AsyncMock) as mock_vision, \
             patch('src.agents.evaluation_agent.policy') as mock_policy:

            mock_policy.get.return_value = "vision"
            mock_policy.get_feature_flag.return_value = True

            mock_obj.return_value = ({"coverage": 0.02}, False)  # Failed
            mock_vision.return_value = ({}, "Feedback")

            image_data = {"base64": sample_image_base64}
            await agent._evaluate_image(
                image_data, {}, {}, ReasoningModel.GEMINI
            )

            # Should call vision anyway
            mock_vision.assert_called_once()


class TestObjectiveChecks:
    """Test objective image checks"""

    @pytest.fixture
    def agent(self):
        """Create Evaluation Agent"""
        with patch('src.agents.evaluation_agent.GeminiReasoningEngine'), \
             patch('src.agents.evaluation_agent.ChatGPTReasoningEngine'):
            return EvaluationAgent()

    @pytest.mark.asyncio
    async def test_calculates_coverage(self, agent, sample_image_base64):
        """Test coverage calculation"""
        with patch.object(agent, '_calculate_coverage') as mock_coverage:
            mock_coverage.return_value = 0.75

            image_data = {"base64": sample_image_base64}
            scores, passed = await agent._objective_checks(image_data)

            assert "coverage" in scores
            mock_coverage.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculates_whiteness(self, agent, sample_image_base64):
        """Test background whiteness calculation"""
        with patch.object(agent, '_calculate_background_whiteness') as mock_white:
            mock_white.return_value = 0.95

            image_data = {"base64": sample_image_base64}
            scores, passed = await agent._objective_checks(image_data)

            assert "background_whiteness" in scores
            mock_white.assert_called_once()

    @pytest.mark.asyncio
    async def test_pass_fail_logic(self, agent, sample_image_base64):
        """Test objective pass/fail logic"""
        with patch.object(agent, '_calculate_coverage') as mock_cov, \
             patch.object(agent, '_calculate_background_whiteness') as mock_white, \
             patch('src.agents.evaluation_agent.policy') as mock_policy:

            mock_policy.get.side_effect = lambda path, default: {
                "evaluation.objective_checks.coverage_min": 0.05,
                "evaluation.objective_checks.coverage_max": 0.90,
                "evaluation.objective_checks.whiteness_min": 0.85,
            }.get(path, default)

            # Good values
            mock_cov.return_value = 0.70
            mock_white.return_value = 0.95

            image_data = {"base64": sample_image_base64}
            scores, passed = await agent._objective_checks(image_data)

            assert passed is True

    @pytest.mark.asyncio
    async def test_fails_on_low_coverage(self, agent, sample_image_base64):
        """Test fails when coverage too low"""
        with patch.object(agent, '_calculate_coverage') as mock_cov, \
             patch.object(agent, '_calculate_background_whiteness') as mock_white, \
             patch('src.agents.evaluation_agent.policy') as mock_policy:

            mock_policy.get.side_effect = lambda path, default: {
                "evaluation.objective_checks.coverage_min": 0.05,
            }.get(path, default)

            mock_cov.return_value = 0.02  # Too low
            mock_white.return_value = 0.95

            image_data = {"base64": sample_image_base64}
            scores, passed = await agent._objective_checks(image_data)

            assert passed is False


class TestVisionEvaluation:
    """Test vision LLM evaluation"""

    @pytest.fixture
    def agent(self):
        """Create Evaluation Agent"""
        with patch('src.agents.evaluation_agent.GeminiReasoningEngine') as mock_gemini, \
             patch('src.agents.evaluation_agent.ChatGPTReasoningEngine'):

            agent = EvaluationAgent()
            agent.gemini_engine.generate_with_image = AsyncMock(
                return_value="Aesthetics: 8.5/10\nSuitability: 9.0/10"
            )
            return agent

    @pytest.mark.asyncio
    async def test_calls_vision_llm(self, agent, sample_image_base64):
        """Test calls vision LLM with image"""
        image_data = {"base64": sample_image_base64}

        scores, feedback = await agent._vision_evaluation(
            image_data, {"primary": "Test prompt"}, ReasoningModel.GEMINI
        )

        # Should call vision LLM
        agent.gemini_engine.generate_with_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_parses_vision_scores(self, agent, sample_image_base64):
        """Test parses scores from vision LLM response"""
        agent.gemini_engine.generate_with_image.return_value = (
            "Aesthetics: 8.5/10\nSuitability: 9.0/10"
        )

        image_data = {"base64": sample_image_base64}
        scores, feedback = await agent._vision_evaluation(
            image_data, {}, ReasoningModel.GEMINI
        )

        assert isinstance(scores, dict)
        assert isinstance(feedback, str)


class TestScoreCombination:
    """Test score combination logic"""

    @pytest.fixture
    def agent(self):
        """Create Evaluation Agent"""
        with patch('src.agents.evaluation_agent.GeminiReasoningEngine'), \
             patch('src.agents.evaluation_agent.ChatGPTReasoningEngine'):
            return EvaluationAgent()

    def test_combines_objective_and_subjective(self, agent):
        """Test combines objective and subjective scores"""
        objective = {"coverage": 0.80, "background_whiteness": 0.95}
        subjective = {"aesthetics": 0.85, "suitability": 0.90}

        combined = agent._combine_scores(objective, subjective)

        assert "coverage" in combined
        assert "aesthetics" in combined

    def test_handles_empty_subjective(self, agent):
        """Test handles missing subjective scores"""
        objective = {"coverage": 0.80}
        subjective = {}

        combined = agent._combine_scores(objective, subjective)

        # Should still have objective scores
        assert "coverage" in combined

    def test_calculates_weighted_overall(self, agent):
        """Test calculates weighted overall score"""
        with patch('src.agents.evaluation_agent.policy') as mock_policy:
            mock_policy.get.return_value = {
                "coverage": 0.35,
                "aesthetics": 0.40,
                "suitability": 0.25
            }

            scores = {
                "coverage": 0.80,
                "aesthetics": 0.90,
                "suitability": 0.85
            }

            overall = agent._calculate_overall_score_weighted(scores)

            # Weighted: 0.35*0.80 + 0.40*0.90 + 0.25*0.85 = 0.8525
            assert abs(overall - 0.8525) < 0.01


class TestApprovalLogic:
    """Test image approval logic"""

    @pytest.fixture
    def agent(self):
        """Create Evaluation Agent"""
        with patch('src.agents.evaluation_agent.GeminiReasoningEngine'), \
             patch('src.agents.evaluation_agent.ChatGPTReasoningEngine'):
            return EvaluationAgent()

    def test_approves_high_scores(self, agent):
        """Test approves images above threshold"""
        with patch('src.agents.evaluation_agent.policy') as mock_policy:
            mock_policy.get_evaluation_threshold.return_value = 0.75

            approved = agent._is_approved(0.85)

            assert approved is True

    def test_rejects_low_scores(self, agent):
        """Test rejects images below threshold"""
        with patch('src.agents.evaluation_agent.policy') as mock_policy:
            mock_policy.get_evaluation_threshold.return_value = 0.75

            approved = agent._is_approved(0.60)

            assert approved is False

    def test_selects_best_image(self, agent):
        """Test selects image with highest score"""
        evaluations = [
            {"image_id": "img-1", "overall_score": 0.75},
            {"image_id": "img-2", "overall_score": 0.90},
            {"image_id": "img-3", "overall_score": 0.82},
        ]

        best = agent._select_best_image(evaluations)

        assert best["image_id"] == "img-2"
        assert best["overall_score"] == 0.90


class TestMainRun:
    """Test main run() method"""

    @pytest.fixture
    def agent(self):
        """Create Evaluation Agent"""
        with patch('src.agents.evaluation_agent.GeminiReasoningEngine'), \
             patch('src.agents.evaluation_agent.ChatGPTReasoningEngine'):
            return EvaluationAgent()

    @pytest.mark.asyncio
    async def test_evaluates_all_images(self, agent, sample_image_base64):
        """Test evaluates all provided images"""
        with patch.object(agent, '_evaluate_image', new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = {
                "image_id": "img-1",
                "overall_score": 0.85,
                "approved": True
            }

            images = [
                {"base64": sample_image_base64},
                {"base64": sample_image_base64},
                {"base64": sample_image_base64},
            ]

            result = await agent.run({
                "images": images,
                "context": {},
                "prompts": {}
            })

            # Should evaluate all 3 images
            assert mock_eval.call_count == 3
            assert len(result["evaluations"]) == 3

    @pytest.mark.asyncio
    async def test_returns_best_image(self, agent, sample_image_base64):
        """Test returns best scoring image"""
        with patch.object(agent, '_evaluate_image', new_callable=AsyncMock) as mock_eval:
            mock_eval.side_effect = [
                {"image_id": "img-1", "overall_score": 0.70, "approved": True},
                {"image_id": "img-2", "overall_score": 0.90, "approved": True},
            ]

            images = [
                {"base64": sample_image_base64},
                {"base64": sample_image_base64},
            ]

            result = await agent.run({
                "images": images,
                "context": {},
                "prompts": {}
            })

            assert result["best_image"]["image_id"] == "img-2"

    @pytest.mark.asyncio
    async def test_returns_approved_images(self, agent, sample_image_base64):
        """Test returns list of approved images"""
        with patch.object(agent, '_evaluate_image', new_callable=AsyncMock) as mock_eval:
            mock_eval.side_effect = [
                {"image_id": "img-1", "overall_score": 0.85, "approved": True},
                {"image_id": "img-2", "overall_score": 0.60, "approved": False},
                {"image_id": "img-3", "overall_score": 0.78, "approved": True},
            ]

            images = [
                {"base64": sample_image_base64},
                {"base64": sample_image_base64},
                {"base64": sample_image_base64},
            ]

            result = await agent.run({
                "images": images,
                "context": {},
                "prompts": {}
            })

            # Should have 2 approved images
            assert len(result["approved_images"]) == 2

    @pytest.mark.asyncio
    async def test_handles_errors(self, agent, sample_image_base64):
        """Test handles evaluation errors gracefully"""
        with patch.object(agent, '_evaluate_image', new_callable=AsyncMock) as mock_eval:
            mock_eval.side_effect = Exception("Evaluation error")

            result = await agent.run({
                "images": [{"base64": sample_image_base64}],
                "context": {},
                "prompts": {}
            })

            assert result["success"] is False
            assert "error" in result
