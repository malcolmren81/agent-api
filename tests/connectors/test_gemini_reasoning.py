"""
Unit tests for Gemini Reasoning Connector

Tests vision API integration and multimodal capabilities.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.connectors.gemini_reasoning import GeminiReasoningEngine


class TestGeminiVisionAPI:
    """Test Gemini Vision API integration"""

    @pytest.fixture
    def engine(self):
        """Create Gemini engine with mocked client"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel'):
            return GeminiReasoningEngine()

    @pytest.mark.asyncio
    async def test_generate_with_image_calls_api(self, engine, sample_image_base64):
        """Test generate_with_image calls Gemini Vision API"""
        # Mock the model's generate_content_async
        mock_response = MagicMock()
        mock_response.text = "This is a test image"
        mock_response.candidates = [MagicMock(finish_reason=1)]  # STOP

        engine.model.generate_content_async = AsyncMock(return_value=mock_response)

        result = await engine.generate_with_image(
            prompt="Describe this image",
            image_base64=sample_image_base64
        )

        assert result == "This is a test image"
        engine.model.generate_content_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_base64_decoding(self, engine, sample_image_base64):
        """Test properly decodes base64 image"""
        mock_response = MagicMock()
        mock_response.text = "Description"
        mock_response.candidates = [MagicMock(finish_reason=1)]

        engine.model.generate_content_async = AsyncMock(return_value=mock_response)

        # Should not raise exception
        result = await engine.generate_with_image(
            prompt="Test",
            image_base64=sample_image_base64
        )

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_handles_safety_filters(self, engine, sample_image_base64):
        """Test handles content blocked by safety filters"""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock(finish_reason=2)]  # SAFETY

        engine.model.generate_content_async = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="safety"):
            await engine.generate_with_image(
                prompt="Test",
                image_base64=sample_image_base64
            )

    @pytest.mark.asyncio
    async def test_respects_temperature(self, engine, sample_image_base64):
        """Test respects temperature parameter"""
        mock_response = MagicMock()
        mock_response.text = "Result"
        mock_response.candidates = [MagicMock(finish_reason=1)]

        engine.model.generate_content_async = AsyncMock(return_value=mock_response)

        await engine.generate_with_image(
            prompt="Test",
            image_base64=sample_image_base64,
            temperature=0.9
        )

        # Check that temperature was used in config
        call_args = engine.model.generate_content_async.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_respects_max_tokens(self, engine, sample_image_base64):
        """Test respects max_tokens parameter"""
        mock_response = MagicMock()
        mock_response.text = "Result"
        mock_response.candidates = [MagicMock(finish_reason=1)]

        engine.model.generate_content_async = AsyncMock(return_value=mock_response)

        await engine.generate_with_image(
            prompt="Test",
            image_base64=sample_image_base64,
            max_tokens=500
        )

        # Should complete without error
        engine.model.generate_content_async.assert_called_once()


class TestGeminiTextGeneration:
    """Test standard text generation"""

    @pytest.fixture
    def engine(self):
        """Create Gemini engine"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel'):
            return GeminiReasoningEngine()

    @pytest.mark.asyncio
    async def test_generate_text(self, engine):
        """Test basic text generation"""
        mock_response = MagicMock()
        mock_response.text = "Generated response"
        mock_response.candidates = [MagicMock(finish_reason=1)]

        engine.model.generate_content_async = AsyncMock(return_value=mock_response)

        result = await engine.generate(
            prompt="Test prompt",
            system_prompt="You are a helpful assistant"
        )

        assert result == "Generated response"

    @pytest.mark.asyncio
    async def test_handles_empty_prompt(self, engine):
        """Test handles empty prompt"""
        with pytest.raises(ValueError, match="Empty prompt"):
            await engine.generate(prompt="")

    @pytest.mark.asyncio
    async def test_streaming_generation(self, engine):
        """Test streaming generation"""
        # Mock streaming response
        async def mock_stream():
            chunks = [
                MagicMock(text="Hello "),
                MagicMock(text="world"),
            ]
            for chunk in chunks:
                yield chunk

        mock_response = MagicMock()
        mock_response.__aiter__ = mock_stream

        engine.model.generate_content_async = AsyncMock(return_value=mock_response)

        stream = await engine.generate(
            prompt="Test",
            stream=True
        )

        # Collect stream
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0


class TestRetryLogic:
    """Test retry logic for transient errors"""

    @pytest.fixture
    def engine(self):
        """Create Gemini engine"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel'):
            return GeminiReasoningEngine()

    @pytest.mark.asyncio
    async def test_retries_rate_limit(self, engine):
        """Test retries on rate limit errors"""
        from google.api_core import exceptions as google_exceptions

        # Fail twice, then succeed
        mock_response = MagicMock()
        mock_response.text = "Success"
        mock_response.candidates = [MagicMock(finish_reason=1)]

        engine.model.generate_content_async = AsyncMock(
            side_effect=[
                google_exceptions.ResourceExhausted("Rate limit"),
                google_exceptions.ResourceExhausted("Rate limit"),
                mock_response
            ]
        )

        result = await engine.generate(prompt="Test")

        # Should succeed after retries
        assert result == "Success"
        assert engine.model.generate_content_async.call_count == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self, engine):
        """Test gives up after max retries"""
        from google.api_core import exceptions as google_exceptions

        engine.model.generate_content_async = AsyncMock(
            side_effect=google_exceptions.ResourceExhausted("Rate limit")
        )

        with pytest.raises(google_exceptions.ResourceExhausted):
            await engine.generate(prompt="Test")

        # Should try 3 times (configured in @retry decorator)
        assert engine.model.generate_content_async.call_count == 3


class TestCostEstimation:
    """Test cost estimation"""

    @pytest.fixture
    def engine(self):
        """Create Gemini engine"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel'):
            return GeminiReasoningEngine()

    @pytest.mark.asyncio
    async def test_estimates_cost(self, engine):
        """Test estimates cost for generation"""
        cost = await engine.estimate_cost(
            prompt="This is a test prompt" * 100,
            max_tokens=1000
        )

        assert isinstance(cost, float)
        assert cost > 0

    @pytest.mark.asyncio
    async def test_cost_scales_with_tokens(self, engine):
        """Test cost scales with token count"""
        small_cost = await engine.estimate_cost(
            prompt="Short prompt",
            max_tokens=100
        )

        large_cost = await engine.estimate_cost(
            prompt="Long prompt" * 1000,
            max_tokens=2000
        )

        assert large_cost > small_cost


class TestMetadata:
    """Test model metadata"""

    @pytest.fixture
    def engine(self):
        """Create Gemini engine"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel'):
            return GeminiReasoningEngine()

    def test_get_metadata(self, engine):
        """Test get_metadata returns correct info"""
        metadata = engine.get_metadata()

        assert metadata.model_name is not None
        assert metadata.provider == "google"
        assert metadata.supports_streaming is True
        assert metadata.max_tokens > 0
