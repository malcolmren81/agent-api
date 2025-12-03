"""Tests for palet8_agents.models.enums module."""

import pytest
from palet8_agents.models.enums import PlannerPhase, EvaluationPhase


class TestPlannerPhase:
    """Tests for PlannerPhase enum."""

    def test_values(self):
        """Test enum values exist."""
        assert PlannerPhase.INITIAL.value == "initial"
        assert PlannerPhase.FIX_PLAN.value == "fix_plan"
        assert PlannerPhase.CLARIFY.value == "clarify"

    def test_from_string(self):
        """Test creating from string value."""
        assert PlannerPhase("initial") == PlannerPhase.INITIAL
        assert PlannerPhase("fix_plan") == PlannerPhase.FIX_PLAN
        assert PlannerPhase("clarify") == PlannerPhase.CLARIFY

    def test_string_representation(self):
        """Test string representation."""
        assert str(PlannerPhase.INITIAL) == "PlannerPhase.INITIAL"
        assert PlannerPhase.INITIAL.name == "INITIAL"

    def test_iteration(self):
        """Test iterating over enum members."""
        phases = list(PlannerPhase)
        assert len(phases) == 3
        assert PlannerPhase.INITIAL in phases
        assert PlannerPhase.FIX_PLAN in phases
        assert PlannerPhase.CLARIFY in phases


class TestEvaluationPhase:
    """Tests for EvaluationPhase enum."""

    def test_values(self):
        """Test enum values exist."""
        assert EvaluationPhase.CREATE_PLAN.value == "create_plan"
        assert EvaluationPhase.EXECUTE.value == "execute"

    def test_from_string(self):
        """Test creating from string value."""
        assert EvaluationPhase("create_plan") == EvaluationPhase.CREATE_PLAN
        assert EvaluationPhase("execute") == EvaluationPhase.EXECUTE

    def test_string_representation(self):
        """Test string representation."""
        assert str(EvaluationPhase.CREATE_PLAN) == "EvaluationPhase.CREATE_PLAN"
        assert EvaluationPhase.CREATE_PLAN.name == "CREATE_PLAN"

    def test_iteration(self):
        """Test iterating over enum members."""
        phases = list(EvaluationPhase)
        assert len(phases) == 2
        assert EvaluationPhase.CREATE_PLAN in phases
        assert EvaluationPhase.EXECUTE in phases

    def test_comparison(self):
        """Test enum comparison."""
        assert EvaluationPhase.CREATE_PLAN != EvaluationPhase.EXECUTE
        assert EvaluationPhase.CREATE_PLAN == EvaluationPhase.CREATE_PLAN

    def test_hash(self):
        """Test enum is hashable for use in sets/dicts."""
        phase_set = {EvaluationPhase.CREATE_PLAN, EvaluationPhase.EXECUTE}
        assert len(phase_set) == 2

        phase_dict = {EvaluationPhase.CREATE_PLAN: "Phase 1", EvaluationPhase.EXECUTE: "Phase 2"}
        assert phase_dict[EvaluationPhase.CREATE_PLAN] == "Phase 1"
