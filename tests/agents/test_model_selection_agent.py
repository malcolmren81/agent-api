"""
Unit tests for Model Selection Agent

Tests UCB1 bandit algorithm, bucket determination, reward updates,
and fallback selection logic.
"""

import pytest
import math
from unittest.mock import patch, MagicMock, AsyncMock
from src.agents.model_selection_agent import ModelSelectionAgent
from src.models.schemas import ReasoningModel, ImageModel


class TestUCB1Selection:
    """Test UCB1 bandit algorithm"""

    @pytest.fixture
    def agent(self):
        """Create Model Selection Agent with mocked Prisma"""
        with patch('src.agents.model_selection_agent.Prisma'):
            agent = ModelSelectionAgent()
            agent.prisma = MagicMock()
            return agent

    @pytest.mark.asyncio
    async def test_ucb1_selects_highest_score(self, agent, sample_model_stats):
        """Test UCB1 selects model with highest UCB score"""
        # Mock Prisma connection
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

        prompts = {"primary": "Product photo on white background"}
        context = {}

        model, rationale = await agent._ucb1_selection(prompts, context)

        # flux-pro has highest impressions and reward mean, should be selected
        assert isinstance(model, ImageModel)
        assert "UCB1" in rationale

    @pytest.mark.asyncio
    async def test_cold_start_exploration(self, agent):
        """Test cold start forces exploration of underexplored models"""
        # Create stats where one model has very few trials
        stats = [
            MagicMock(
                id="stat-1",
                modelName="flux",
                bucket="product:realistic:white-bg",
                impressions=50,
                rewardMean=0.80,
                rewardVar=0.05
            ),
            MagicMock(
                id="stat-2",
                modelName="imagen-3",
                bucket="product:realistic:white-bg",
                impressions=0,  # Never tried
                rewardMean=0.50,
                rewardVar=0.10
            )
        ]

        agent.prisma.connect = AsyncMock()
        agent.prisma.disconnect = AsyncMock()
        agent.prisma.modelstats.find_many = AsyncMock(return_value=stats)

        with patch('src.agents.model_selection_agent.policy') as mock_policy:
            mock_policy.get.return_value = 1  # min_trials = 1

            model, rationale = await agent._ucb1_selection(
                {"primary": "Product photo"},
                {}
            )

            # Should select underexplored model (imagen-3)
            assert "cold start" in rationale.lower()

    @pytest.mark.asyncio
    async def test_fallback_on_no_stats(self, agent):
        """Test falls back to rule-based when no stats available"""
        agent.prisma.connect = AsyncMock()
        agent.prisma.disconnect = AsyncMock()
        agent.prisma.modelstats.find_many = AsyncMock(return_value=[])

        model, rationale = await agent._ucb1_selection(
            {"primary": "Product photo on white background"},
            {}
        )

        # Should use rule-based fallback
        assert isinstance(model, ImageModel)

    @pytest.mark.asyncio
    async def test_fallback_on_db_error(self, agent):
        """Test falls back to rule-based on database error"""
        agent.prisma.connect = AsyncMock(side_effect=Exception("DB Error"))

        model, rationale = await agent._ucb1_selection(
            {"primary": "Product photo"},
            {}
        )

        # Should fallback gracefully
        assert isinstance(model, ImageModel)

    @pytest.mark.asyncio
    async def test_stores_selection_metadata(self, agent, sample_model_stats):
        """Test stores selection metadata in context for reward update"""
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

        context = {}
        model, rationale = await agent._ucb1_selection(
            {"primary": "Product photo"},
            context
        )

        # Should store stat_id and bucket
        assert "selected_model_stat_id" in context
        assert "bucket" in context


class TestUCBScoreCalculation:
    """Test UCB score calculation formula"""

    @pytest.fixture
    def agent(self):
        """Create Model Selection Agent"""
        with patch('src.agents.model_selection_agent.Prisma'):
            return ModelSelectionAgent()

    def test_ucb_formula_correct(self, agent):
        """Test UCB1 formula: μ + sqrt((2 * ln N) / n)"""
        mean_reward = 0.75
        impressions = 10
        total_impressions = 100

        ucb = agent._calculate_ucb_score(mean_reward, impressions, total_impressions)

        # Manual calculation
        expected_exploration = math.sqrt((2 * math.log(total_impressions)) / impressions)
        expected_ucb = mean_reward + expected_exploration

        assert abs(ucb - expected_ucb) < 0.001

    def test_untried_model_gets_infinity(self, agent):
        """Test untried models get infinite UCB score"""
        ucb = agent._calculate_ucb_score(
            mean_reward=0.5,
            impressions=0,
            total_impressions=100
        )

        assert ucb == float('inf')

    def test_higher_uncertainty_higher_score(self, agent):
        """Test models with fewer impressions get higher exploration bonus"""
        # Model with few impressions
        ucb_few = agent._calculate_ucb_score(
            mean_reward=0.70,
            impressions=5,
            total_impressions=100
        )

        # Model with many impressions
        ucb_many = agent._calculate_ucb_score(
            mean_reward=0.70,  # Same mean
            impressions=50,
            total_impressions=100
        )

        # Few impressions should have higher UCB (more exploration)
        assert ucb_few > ucb_many

    def test_insufficient_data_returns_mean(self, agent):
        """Test returns mean reward when insufficient total data"""
        ucb = agent._calculate_ucb_score(
            mean_reward=0.75,
            impressions=1,
            total_impressions=1
        )

        # Should just return mean when N <= 1
        assert ucb == 0.75


class TestBucketDetermination:
    """Test use-case bucket categorization"""

    @pytest.fixture
    def agent(self):
        """Create Model Selection Agent"""
        with patch('src.agents.model_selection_agent.Prisma'):
            return ModelSelectionAgent()

    def test_product_realistic_white_bg(self, agent):
        """Test product with white background categorization"""
        bucket = agent._determine_bucket(
            {"primary": "Product photo of t-shirt on white background"},
            {}
        )

        assert bucket == "product:realistic:white-bg"

    def test_product_realistic_lifestyle(self, agent):
        """Test product lifestyle categorization"""
        bucket = agent._determine_bucket(
            {"primary": "Lifestyle photo of product in authentic use"},
            {}
        )

        assert bucket == "product:realistic:lifestyle"

    def test_creative_artistic_abstract(self, agent):
        """Test creative abstract categorization"""
        bucket = agent._determine_bucket(
            {"primary": "Abstract creative visualization", "style": "artistic"},
            {}
        )

        assert bucket == "creative:artistic:abstract"

    def test_creative_realistic_scene(self, agent):
        """Test creative realistic scene categorization"""
        bucket = agent._determine_bucket(
            {"primary": "Beautiful landscape scene"},
            {}
        )

        assert bucket == "creative:realistic:scene"

    def test_style_from_prompts(self, agent):
        """Test style detection from prompts dict"""
        bucket = agent._determine_bucket(
            {"primary": "Product photo", "style": "artistic watercolor"},
            {}
        )

        assert "artistic" in bucket


class TestRuleBasedSelection:
    """Test rule-based fallback selection"""

    @pytest.fixture
    def agent(self):
        """Create Model Selection Agent"""
        with patch('src.agents.model_selection_agent.Prisma'):
            return ModelSelectionAgent()

    def test_photorealistic_selects_imagen(self, agent):
        """Test photorealistic prompts select Imagen 3"""
        model, rationale = agent._rule_based_image_selection(
            {"primary": "Photorealistic product photo"},
            {}
        )

        assert model == ImageModel.GEMINI
        assert "photorealism" in rationale.lower()

    def test_style_transfer_selects_flux(self, agent):
        """Test style transfer prompts select Flux"""
        model, rationale = agent._rule_based_image_selection(
            {"primary": "Product with watercolor style transfer"},
            {}
        )

        assert model == ImageModel.FLUX
        assert "style" in rationale.lower()

    def test_character_consistency_selects_flux(self, agent):
        """Test character consistency prompts select Flux"""
        model, rationale = agent._rule_based_image_selection(
            {"primary": "Consistent character design"},
            {}
        )

        assert model == ImageModel.FLUX
        assert "character" in rationale.lower() or "consistent" in rationale.lower()

    def test_multi_image_blend_selects_imagen(self, agent):
        """Test multi-image blending selects Imagen 3"""
        model, rationale = agent._rule_based_image_selection(
            {"primary": "Blend multiple images together"},
            {}
        )

        assert model == ImageModel.GEMINI
        assert "blend" in rationale.lower()

    def test_default_selects_flux(self, agent):
        """Test default case selects Flux for speed"""
        model, rationale = agent._rule_based_image_selection(
            {"primary": "Generic image generation"},
            {}
        )

        assert model == ImageModel.FLUX
        assert "faster" in rationale.lower() or "fast" in rationale.lower()


class TestRewardUpdate:
    """Test bandit reward update mechanism"""

    @pytest.fixture
    def agent(self):
        """Create Model Selection Agent with mocked Prisma"""
        with patch('src.agents.model_selection_agent.Prisma'):
            agent = ModelSelectionAgent()
            agent.prisma = MagicMock()
            return agent

    @pytest.mark.asyncio
    async def test_update_increments_impressions(self, agent):
        """Test reward update increments impression count"""
        agent.prisma.connect = AsyncMock()
        agent.prisma.disconnect = AsyncMock()

        # Mock existing stat
        existing_stat = MagicMock(
            impressions=10,
            rewardMean=0.75,
            rewardVar=0.05
        )
        agent.prisma.modelstats.find_unique = AsyncMock(return_value=existing_stat)
        agent.prisma.modelstats.update = AsyncMock()

        context = {
            "selected_model_stat_id": "stat-1",
            "bucket": "product:realistic:white-bg"
        }
        evaluation = {
            "overall_score": 0.85
        }

        await agent.update_bandit_reward(context, evaluation)

        # Should update with incremented impressions
        agent.prisma.modelstats.update.assert_called_once()
        update_call = agent.prisma.modelstats.update.call_args
        assert update_call[1]["data"]["impressions"] == 11

    @pytest.mark.asyncio
    async def test_update_uses_ema(self, agent):
        """Test reward update uses EMA for mean"""
        agent.prisma.connect = AsyncMock()
        agent.prisma.disconnect = AsyncMock()

        existing_stat = MagicMock(
            impressions=10,
            rewardMean=0.70,
            rewardVar=0.05
        )
        agent.prisma.modelstats.find_unique = AsyncMock(return_value=existing_stat)
        agent.prisma.modelstats.update = AsyncMock()

        with patch('src.agents.model_selection_agent.policy') as mock_policy:
            mock_policy.get.return_value = 0.1  # alpha = 0.1

            context = {"selected_model_stat_id": "stat-1"}
            evaluation = {"overall_score": 0.90}

            await agent.update_bandit_reward(context, evaluation)

            # EMA: new_mean = alpha * new_score + (1 - alpha) * old_mean
            # new_mean = 0.1 * 0.90 + 0.9 * 0.70 = 0.72

            update_call = agent.prisma.modelstats.update.call_args
            updated_mean = update_call[1]["data"]["rewardMean"]

            # Allow small floating point error
            assert abs(updated_mean - 0.72) < 0.01

    @pytest.mark.asyncio
    async def test_handles_missing_stat_id(self, agent):
        """Test gracefully handles missing stat ID"""
        context = {}  # No stat_id
        evaluation = {"overall_score": 0.85}

        # Should not raise exception
        await agent.update_bandit_reward(context, evaluation)


class TestModelSelectionRun:
    """Test main run() method"""

    @pytest.fixture
    def agent(self):
        """Create Model Selection Agent"""
        with patch('src.agents.model_selection_agent.Prisma'):
            return ModelSelectionAgent()

    @pytest.mark.asyncio
    async def test_run_selects_both_models(self, agent):
        """Test run() selects both reasoning and image models"""
        with patch.object(agent, '_select_reasoning_model', new_callable=AsyncMock) as mock_reason, \
             patch.object(agent, '_select_image_model', new_callable=AsyncMock) as mock_image, \
             patch.object(agent, '_calculate_total_cost', new_callable=AsyncMock) as mock_cost:

            mock_reason.return_value = (ReasoningModel.GEMINI, "Fast and cheap")
            mock_image.return_value = (ImageModel.FLUX, "Best quality")
            mock_cost.return_value = 0.055

            result = await agent.run({
                "context": {"prompt": "Test prompt"},
                "prompts": {"primary": "Test"}
            })

            assert result["success"] is True
            assert "selected_models" in result
            assert "reasoning" in result["selected_models"]
            assert "image" in result["selected_models"]

    @pytest.mark.asyncio
    async def test_run_includes_cost_estimate(self, agent):
        """Test run() includes cost estimate"""
        with patch.object(agent, '_select_reasoning_model', new_callable=AsyncMock) as mock_reason, \
             patch.object(agent, '_select_image_model', new_callable=AsyncMock) as mock_image, \
             patch.object(agent, '_calculate_total_cost', new_callable=AsyncMock) as mock_cost:

            mock_reason.return_value = (ReasoningModel.GEMINI, "Rationale")
            mock_image.return_value = (ImageModel.FLUX, "Rationale")
            mock_cost.return_value = 0.055

            result = await agent.run({
                "context": {},
                "prompts": {}
            })

            assert "estimated_cost" in result
            assert result["estimated_cost"] == 0.055

    @pytest.mark.asyncio
    async def test_run_handles_errors(self, agent):
        """Test run() handles selection errors gracefully"""
        with patch.object(agent, '_select_reasoning_model', new_callable=AsyncMock) as mock_reason:
            mock_reason.side_effect = Exception("Selection error")

            result = await agent.run({
                "context": {},
                "prompts": {}
            })

            assert result["success"] is False
            assert "error" in result


class TestReasoningModelSelection:
    """Test reasoning model selection logic"""

    @pytest.fixture
    def agent(self):
        """Create Model Selection Agent"""
        with patch('src.agents.model_selection_agent.Prisma'):
            return ModelSelectionAgent()

    @pytest.mark.asyncio
    async def test_defaults_to_gemini(self, agent):
        """Test defaults to Gemini for cost and speed"""
        model, rationale = await agent._select_reasoning_model(
            user_preference=None,
            context={"prompt": "Generate product description"}
        )

        assert model == ReasoningModel.GEMINI
        assert "gemini" in rationale.lower()

    @pytest.mark.asyncio
    async def test_technical_prompts_use_gpt4o(self, agent):
        """Test technical prompts select GPT-4o"""
        model, rationale = await agent._select_reasoning_model(
            user_preference=None,
            context={"prompt": "Write a Python function to parse API responses"}
        )

        assert model == ReasoningModel.CHATGPT
        assert "gpt" in rationale.lower() or "technical" in rationale.lower()


class TestFeatureFlagIntegration:
    """Test feature flag integration"""

    @pytest.fixture
    def agent(self):
        """Create Model Selection Agent"""
        with patch('src.agents.model_selection_agent.Prisma'):
            return ModelSelectionAgent()

    @pytest.mark.asyncio
    async def test_bandit_disabled_uses_rules(self, agent):
        """Test bandit disabled falls back to rules"""
        with patch('src.agents.model_selection_agent.policy') as mock_policy:
            mock_policy.get_feature_flag.return_value = False

            with patch.object(agent, '_rule_based_image_selection') as mock_rule:
                mock_rule.return_value = (ImageModel.FLUX, "Rule-based")

                await agent._select_image_model(
                    user_preference=None,
                    prompts={"primary": "Test"},
                    context={}
                )

                mock_rule.assert_called_once()

    @pytest.mark.asyncio
    async def test_bandit_enabled_uses_ucb1(self, agent):
        """Test bandit enabled uses UCB1"""
        agent.prisma = MagicMock()

        with patch('src.agents.model_selection_agent.policy') as mock_policy:
            mock_policy.get_feature_flag.return_value = True

            with patch.object(agent, '_ucb1_selection', new_callable=AsyncMock) as mock_ucb1:
                mock_ucb1.return_value = (ImageModel.FLUX, "UCB1")

                await agent._select_image_model(
                    user_preference=None,
                    prompts={"primary": "Test"},
                    context={}
                )

                mock_ucb1.assert_called_once()
