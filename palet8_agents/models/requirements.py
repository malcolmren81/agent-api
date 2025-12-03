"""
Requirements models for design request gathering.

Used by PaliAgent for tracking user input completeness.
"""

from typing import Any, Dict, List, Optional


class RequirementsStatus:
    """Status of requirements gathering."""

    def __init__(self):
        self.subject: Optional[str] = None
        self.style: Optional[str] = None
        self.colors: List[str] = []
        self.mood: Optional[str] = None
        self.composition: Optional[str] = None
        self.include_elements: List[str] = []
        self.avoid_elements: List[str] = []
        self.reference_image: Optional[str] = None
        self.additional_notes: Optional[str] = None

    @property
    def completeness_score(self) -> float:
        """
        Calculate how complete the requirements are (0.0 to 1.0).

        Note: This is a basic score for UI feedback only.
        Planner Agent does the thorough "Enough Context?" check.
        """
        score = 0.0
        if self.subject:
            score += 0.5
        if self.style:
            score += 0.2
        if self.colors:
            score += 0.15
        if self.mood:
            score += 0.15
        return min(1.0, score)

    @property
    def is_complete(self) -> bool:
        """
        Check if minimum requirements are met to pass to Planner.

        Pali only checks: Does user have a subject/intent?
        Planner does the thorough "Enough Context?" evaluation.
        """
        return self.subject is not None

    @property
    def missing_fields(self) -> List[str]:
        """Get list of missing required/recommended fields."""
        missing = []
        if not self.subject:
            missing.append("subject")
        if not self.style:
            missing.append("style")
        if not self.colors:
            missing.append("colors")
        return missing

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject": self.subject,
            "style": self.style,
            "colors": self.colors,
            "mood": self.mood,
            "composition": self.composition,
            "include_elements": self.include_elements,
            "avoid_elements": self.avoid_elements,
            "reference_image": self.reference_image,
            "additional_notes": self.additional_notes,
            "completeness_score": self.completeness_score,
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequirementsStatus":
        """Create from dictionary."""
        instance = cls()
        instance.subject = data.get("subject")
        instance.style = data.get("style")
        instance.colors = data.get("colors", [])
        instance.mood = data.get("mood")
        instance.composition = data.get("composition")
        instance.include_elements = data.get("include_elements", [])
        instance.avoid_elements = data.get("avoid_elements", [])
        instance.reference_image = data.get("reference_image")
        instance.additional_notes = data.get("additional_notes")
        return instance
