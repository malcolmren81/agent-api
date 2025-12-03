"""Tests for palet8_agents.models.planning module."""

import pytest
from palet8_agents.models.planning import (
    PlanningTask,
    ContextSummary,
    PromptPlan,
)


class TestPlanningTask:
    """Tests for PlanningTask dataclass."""

    def test_init_minimal(self):
        """Test minimal initialization."""
        task = PlanningTask(
            job_id="job-123",
            user_id="user-456",
            phase="initial",
            requirements={"subject": "a cat"},
            complexity="standard",
            product_type="poster",
        )
        assert task.job_id == "job-123"
        assert task.user_id == "user-456"
        assert task.phase == "initial"
        assert task.requirements == {"subject": "a cat"}
        assert task.complexity == "standard"
        assert task.product_type == "poster"
        assert task.print_method is None
        assert task.previous_plan is None
        assert task.evaluation_feedback is None
        assert task.edit_instructions is None
        assert task.metadata == {}

    def test_init_full(self):
        """Test full initialization."""
        task = PlanningTask(
            job_id="job-789",
            user_id="user-abc",
            phase="fix_plan",
            requirements={"subject": "a dog", "style": "cartoon"},
            complexity="complex",
            product_type="t-shirt",
            print_method="screen_print",
            previous_plan={"prompt": "A cartoon dog", "quality_score": 0.6},
            evaluation_feedback={"issues": ["Clarity too low"]},
            metadata={"revision": 2},
        )
        assert task.phase == "fix_plan"
        assert task.print_method == "screen_print"
        assert task.previous_plan["prompt"] == "A cartoon dog"
        assert task.evaluation_feedback["issues"] == ["Clarity too low"]

    def test_init_edit_phase(self):
        """Test initialization for edit phase."""
        task = PlanningTask(
            job_id="job-edit",
            user_id="user-edit",
            phase="edit",
            requirements={"subject": "sunset"},
            complexity="simple",
            product_type="mug",
            previous_plan={"prompt": "A beautiful sunset"},
            edit_instructions="Make the sky more orange",
        )
        assert task.phase == "edit"
        assert task.edit_instructions == "Make the sky more orange"

    def test_is_initial(self):
        """Test is_initial property."""
        task = PlanningTask(
            job_id="job-1",
            user_id="user-1",
            phase="initial",
            requirements={},
            complexity="standard",
            product_type="poster",
        )
        assert task.is_initial is True
        assert task.is_fix is False
        assert task.is_edit is False

    def test_is_fix(self):
        """Test is_fix property."""
        task = PlanningTask(
            job_id="job-2",
            user_id="user-2",
            phase="fix_plan",
            requirements={},
            complexity="standard",
            product_type="poster",
        )
        assert task.is_initial is False
        assert task.is_fix is True
        assert task.is_edit is False

    def test_is_edit(self):
        """Test is_edit property."""
        task = PlanningTask(
            job_id="job-3",
            user_id="user-3",
            phase="edit",
            requirements={},
            complexity="standard",
            product_type="poster",
        )
        assert task.is_initial is False
        assert task.is_fix is False
        assert task.is_edit is True

    def test_to_dict(self):
        """Test to_dict serialization."""
        task = PlanningTask(
            job_id="job-dict",
            user_id="user-dict",
            phase="initial",
            requirements={"subject": "mountain"},
            complexity="complex",
            product_type="canvas",
            print_method="giclée",
            metadata={"source": "web"},
        )
        data = task.to_dict()

        assert data["job_id"] == "job-dict"
        assert data["user_id"] == "user-dict"
        assert data["phase"] == "initial"
        assert data["requirements"]["subject"] == "mountain"
        assert data["complexity"] == "complex"
        assert data["product_type"] == "canvas"
        assert data["print_method"] == "giclée"
        assert data["previous_plan"] is None
        assert data["evaluation_feedback"] is None
        assert data["edit_instructions"] is None
        assert data["metadata"]["source"] == "web"

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "job_id": "job-from",
            "user_id": "user-from",
            "phase": "fix_plan",
            "requirements": {"subject": "ocean", "colors": ["blue", "white"]},
            "complexity": "standard",
            "product_type": "poster",
            "print_method": None,
            "previous_plan": {"prompt": "An ocean scene"},
            "evaluation_feedback": {"passed": False, "issues": ["Too dark"]},
            "edit_instructions": None,
            "metadata": {},
        }
        task = PlanningTask.from_dict(data)

        assert task.job_id == "job-from"
        assert task.phase == "fix_plan"
        assert task.requirements["colors"] == ["blue", "white"]
        assert task.previous_plan["prompt"] == "An ocean scene"
        assert task.evaluation_feedback["passed"] is False

    def test_from_dict_defaults(self):
        """Test from_dict with minimal data uses defaults."""
        data = {
            "job_id": "job-min",
            "user_id": "user-min",
        }
        task = PlanningTask.from_dict(data)

        assert task.phase == "initial"
        assert task.requirements == {}
        assert task.complexity == "standard"
        assert task.product_type == "general"

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = PlanningTask(
            job_id="job-roundtrip",
            user_id="user-roundtrip",
            phase="edit",
            requirements={"subject": "flower", "style": "watercolor"},
            complexity="simple",
            product_type="greeting_card",
            print_method="offset",
            previous_plan={"prompt": "A watercolor flower"},
            edit_instructions="Add a bee",
            metadata={"version": 3},
        )

        data = original.to_dict()
        restored = PlanningTask.from_dict(data)

        assert restored.job_id == original.job_id
        assert restored.phase == original.phase
        assert restored.requirements == original.requirements
        assert restored.complexity == original.complexity
        assert restored.edit_instructions == original.edit_instructions


class TestContextSummary:
    """Tests for ContextSummary dataclass."""

    def test_init_defaults(self):
        """Test default initialization."""
        summary = ContextSummary()
        assert summary.user_history_count == 0
        assert summary.art_references_count == 0
        assert summary.web_search_count == 0
        assert summary.rag_sources == []
        assert summary.reference_images == []
        assert summary.metadata == {}

    def test_init_full(self):
        """Test full initialization."""
        summary = ContextSummary(
            user_history_count=5,
            art_references_count=3,
            web_search_count=2,
            rag_sources=["doc1.txt", "doc2.txt"],
            reference_images=["img1.png", "img2.jpg"],
            metadata={"search_query": "vintage poster"},
        )
        assert summary.user_history_count == 5
        assert summary.art_references_count == 3
        assert summary.web_search_count == 2
        assert len(summary.rag_sources) == 2
        assert len(summary.reference_images) == 2
        assert summary.metadata["search_query"] == "vintage poster"

    def test_to_dict(self):
        """Test to_dict serialization."""
        summary = ContextSummary(
            user_history_count=10,
            art_references_count=5,
            rag_sources=["source.md"],
        )
        data = summary.to_dict()

        assert data["user_history_count"] == 10
        assert data["art_references_count"] == 5
        assert data["web_search_count"] == 0
        assert data["rag_sources"] == ["source.md"]
        assert data["reference_images"] == []

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "user_history_count": 3,
            "art_references_count": 7,
            "web_search_count": 1,
            "rag_sources": ["a.txt", "b.txt"],
            "reference_images": ["ref.png"],
            "metadata": {"key": "value"},
        }
        summary = ContextSummary.from_dict(data)

        assert summary.user_history_count == 3
        assert summary.art_references_count == 7
        assert summary.web_search_count == 1
        assert summary.rag_sources == ["a.txt", "b.txt"]
        assert summary.reference_images == ["ref.png"]
        assert summary.metadata["key"] == "value"

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = ContextSummary(
            user_history_count=4,
            art_references_count=2,
            web_search_count=3,
            rag_sources=["doc.txt"],
            reference_images=["photo.jpg"],
            metadata={"test": True},
        )

        data = original.to_dict()
        restored = ContextSummary.from_dict(data)

        assert restored.user_history_count == original.user_history_count
        assert restored.art_references_count == original.art_references_count
        assert restored.web_search_count == original.web_search_count
        assert restored.rag_sources == original.rag_sources


class TestPromptPlan:
    """Tests for PromptPlan dataclass."""

    def test_init_minimal(self):
        """Test minimal initialization."""
        plan = PromptPlan(prompt="A beautiful landscape")
        assert plan.prompt == "A beautiful landscape"
        assert plan.negative_prompt == ""
        assert plan.dimensions == {}
        assert plan.provider_params == {}
        assert plan.quality_score == 0.0
        assert plan.quality_acceptable is False
        assert plan.quality_feedback == []
        assert plan.failed_dimensions == []
        assert plan.revision_count == 0
        assert plan.revision_history == []
        assert isinstance(plan.context_summary, ContextSummary)
        assert plan.mode == "STANDARD"
        assert plan.metadata == {}

    def test_init_full(self):
        """Test full initialization."""
        context = ContextSummary(user_history_count=3, art_references_count=2)
        plan = PromptPlan(
            prompt="A serene mountain lake at sunset",
            negative_prompt="blurry, low quality, distorted",
            dimensions={"subject": "mountain lake", "time": "sunset", "mood": "serene"},
            provider_params={"steps": 40, "guidance_scale": 8.0, "scheduler": "dpm_2m"},
            quality_score=0.85,
            quality_acceptable=True,
            quality_feedback=["Good coverage", "Clear subject"],
            failed_dimensions=[],
            revision_count=1,
            revision_history=["Added lighting details"],
            context_summary=context,
            mode="COMPLEX",
            metadata={"iteration": 2},
        )
        assert plan.prompt == "A serene mountain lake at sunset"
        assert plan.quality_score == 0.85
        assert plan.quality_acceptable is True
        assert plan.revision_count == 1
        assert plan.mode == "COMPLEX"
        assert plan.context_summary.user_history_count == 3
        assert plan.provider_params["steps"] == 40
        assert plan.provider_params["scheduler"] == "dpm_2m"

    def test_is_acceptable_true(self):
        """Test is_acceptable when plan passes quality check."""
        plan = PromptPlan(
            prompt="Test prompt",
            quality_score=0.85,
            quality_acceptable=True,
        )
        assert plan.is_acceptable is True

    def test_is_acceptable_false_low_score(self):
        """Test is_acceptable when score is below 0.7."""
        plan = PromptPlan(
            prompt="Test prompt",
            quality_score=0.65,
            quality_acceptable=True,
        )
        assert plan.is_acceptable is False

    def test_is_acceptable_false_not_acceptable(self):
        """Test is_acceptable when quality_acceptable is False."""
        plan = PromptPlan(
            prompt="Test prompt",
            quality_score=0.85,
            quality_acceptable=False,
        )
        assert plan.is_acceptable is False

    def test_to_dict(self):
        """Test to_dict serialization."""
        context = ContextSummary(user_history_count=2)
        plan = PromptPlan(
            prompt="A cat sleeping on a sofa",
            negative_prompt="dog, outdoors",
            dimensions={"subject": "cat", "action": "sleeping", "location": "sofa"},
            provider_params={"steps": 35, "seed": 12345},
            quality_score=0.9,
            quality_acceptable=True,
            quality_feedback=["Excellent subject clarity"],
            revision_count=0,
            context_summary=context,
            mode="STANDARD",
        )
        data = plan.to_dict()

        assert data["prompt"] == "A cat sleeping on a sofa"
        assert data["provider_params"] == {"steps": 35, "seed": 12345}
        assert data["negative_prompt"] == "dog, outdoors"
        assert data["dimensions"]["subject"] == "cat"
        assert data["quality_score"] == 0.9
        assert data["quality_acceptable"] is True
        assert data["quality_feedback"] == ["Excellent subject clarity"]
        assert data["revision_count"] == 0
        assert data["context_summary"]["user_history_count"] == 2
        assert data["mode"] == "STANDARD"
        assert data["is_acceptable"] is True

    def test_to_dict_with_context_dict(self):
        """Test to_dict when context_summary is already a dict."""
        plan = PromptPlan(prompt="Test")
        # Manually set context_summary to a dict (edge case)
        plan.context_summary = {"user_history_count": 5}  # type: ignore
        data = plan.to_dict()

        assert data["context_summary"] == {"user_history_count": 5}

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "prompt": "A vintage car",
            "negative_prompt": "modern, new",
            "dimensions": {"subject": "car", "style": "vintage"},
            "quality_score": 0.78,
            "quality_acceptable": True,
            "quality_feedback": [],
            "failed_dimensions": ["lighting"],
            "revision_count": 2,
            "revision_history": ["First revision", "Second revision"],
            "context_summary": {
                "user_history_count": 4,
                "art_references_count": 1,
                "web_search_count": 0,
                "rag_sources": [],
                "reference_images": [],
                "metadata": {},
            },
            "mode": "RELAX",
            "metadata": {"note": "test"},
        }
        plan = PromptPlan.from_dict(data)

        assert plan.prompt == "A vintage car"
        assert plan.negative_prompt == "modern, new"
        assert plan.dimensions["style"] == "vintage"
        assert plan.quality_score == 0.78
        assert plan.failed_dimensions == ["lighting"]
        assert plan.revision_count == 2
        assert len(plan.revision_history) == 2
        assert plan.context_summary.user_history_count == 4
        assert plan.mode == "RELAX"

    def test_from_dict_defaults(self):
        """Test from_dict with minimal data uses defaults."""
        data = {}
        plan = PromptPlan.from_dict(data)

        assert plan.prompt == ""
        assert plan.negative_prompt == ""
        assert plan.dimensions == {}
        assert plan.quality_score == 0.0
        assert plan.quality_acceptable is False
        assert plan.mode == "STANDARD"

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        context = ContextSummary(
            user_history_count=5,
            art_references_count=3,
            web_search_count=1,
            rag_sources=["context.txt"],
        )
        original = PromptPlan(
            prompt="A detailed cityscape at night",
            negative_prompt="daytime, empty",
            dimensions={
                "subject": "cityscape",
                "time": "night",
                "style": "photorealistic",
            },
            quality_score=0.88,
            quality_acceptable=True,
            quality_feedback=["Good detail", "Nice composition"],
            failed_dimensions=[],
            revision_count=1,
            revision_history=["Added lighting"],
            context_summary=context,
            mode="COMPLEX",
            metadata={"version": 2},
        )

        data = original.to_dict()
        restored = PromptPlan.from_dict(data)

        assert restored.prompt == original.prompt
        assert restored.negative_prompt == original.negative_prompt
        assert restored.dimensions == original.dimensions
        assert restored.quality_score == original.quality_score
        assert restored.quality_acceptable == original.quality_acceptable
        assert restored.quality_feedback == original.quality_feedback
        assert restored.revision_count == original.revision_count
        assert restored.revision_history == original.revision_history
        assert restored.context_summary.user_history_count == original.context_summary.user_history_count
        assert restored.mode == original.mode
        assert restored.is_acceptable == original.is_acceptable
