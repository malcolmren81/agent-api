"""
Prompt Template Service - Template and rule keeper for prompt generation.

This service is the RULE KEEPER - it loads and provides access to:
- Mode rules (RELAX/STANDARD/COMPLEX dimension requirements)
- Dimension definitions and examples
- Print method constraints
- Product specifications
- Style examples (few-shot library)
- Conflict resolution rules
- Scenario templates

It does NOT make decisions or write prompts.
- Decision making: Planner Agent
- Prompt writing: PromptComposerService

Documentation Reference: Section 5.3
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import yaml

logger = logging.getLogger(__name__)


class PromptMode(Enum):
    """Prompt generation modes."""
    RELAX = "RELAX"
    STANDARD = "STANDARD"
    COMPLEX = "COMPLEX"


class Scenario(Enum):
    """Generation scenarios."""
    NEW_IMAGE = "new_image"
    REFERENCE_BASED = "reference_based"
    PARTIAL_EDIT = "partial_edit"


class PromptTemplateServiceError(Exception):
    """Base exception for Prompt Template Service errors."""
    pass


class LibraryLoadError(PromptTemplateServiceError):
    """Error loading the few-shot library."""
    pass


class PromptTemplateService:
    """
    Template and rule keeper for prompt generation.

    This is a DATA PROVIDER - it does NOT make decisions or write prompts.
    - Decisions are made by the Planner Agent
    - Prompts are written by the PromptComposerService

    Provides:
        - Mode rules (what dimensions required/optional/excluded per mode)
        - Dimension definitions (what each dimension means)
        - Print method constraints (gradients, colors, technical specs)
        - Product specifications (print area, default method)
        - Style examples (few-shot examples with dimensions)
        - Conflict rules (what conflicts exist and how to resolve)
        - Scenario templates (new_image, reference_based, partial_edit)

    Usage:
        service = PromptTemplateService()

        # Get rules for Planner to reason about
        mode_rules = service.get_mode_rules()
        constraints = service.get_print_constraints("screen_print")
        examples = service.get_style_example("streetwear_tiger")
    """

    # Library path - same directory as this service
    DEFAULT_LIBRARY_PATH = Path(__file__).parent / "prompt_fewshot_library.yaml"

    def __init__(
        self,
        library_path: Optional[Path] = None,
    ):
        """
        Initialize the Prompt Template Service and load library.

        Args:
            library_path: Optional path to library YAML. Uses default if not provided.
        """
        self._library_path = library_path or self.DEFAULT_LIBRARY_PATH
        self._library: Dict[str, Any] = {}
        self._loaded = False

        # Load library on init
        self._load_library()

    def _load_library(self) -> None:
        """Load the few-shot library from YAML."""
        try:
            if not self._library_path.exists():
                raise LibraryLoadError(f"Library file not found: {self._library_path}")

            with open(self._library_path, "r", encoding="utf-8") as f:
                self._library = yaml.safe_load(f)

            self._loaded = True
            logger.info(f"Loaded prompt library from {self._library_path}")
            logger.info(f"Library version: {self._library.get('metadata', {}).get('version', 'unknown')}")

        except yaml.YAMLError as e:
            raise LibraryLoadError(f"Failed to parse library YAML: {e}")
        except Exception as e:
            raise LibraryLoadError(f"Failed to load library: {e}")

    def reload_library(self) -> None:
        """Reload the library from disk (hot reload)."""
        self._load_library()

    @property
    def is_loaded(self) -> bool:
        """Check if library is loaded."""
        return self._loaded

    @property
    def library_version(self) -> str:
        """Get library version."""
        return self._library.get("metadata", {}).get("version", "unknown")

    # =========================================================================
    # MODE RULES
    # =========================================================================

    def get_mode_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all mode rules for Planner to reason about.

        Returns:
            Dict with RELAX, STANDARD, COMPLEX rules including:
            - required: dimensions that must be included
            - optional: dimensions that can be included
            - exclude: dimensions that should not be included
            - token_range: expected token count range
        """
        return self._library.get("mode_dimensions", {})

    def get_mode_rule(self, mode: str) -> Dict[str, Any]:
        """
        Get rules for a specific mode.

        Args:
            mode: "RELAX", "STANDARD", or "COMPLEX"

        Returns:
            Mode configuration with required/optional/exclude dimensions
        """
        return self._library.get("mode_dimensions", {}).get(mode, {})

    # =========================================================================
    # DIMENSION DEFINITIONS
    # =========================================================================

    def get_all_dimensions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all dimension definitions.

        Returns:
            Dict of dimension_name -> {description, priority, required_for, examples}
        """
        return self._library.get("dimensions", {})

    def get_dimension(self, dimension: str) -> Dict[str, Any]:
        """
        Get definition for a specific dimension.

        Args:
            dimension: Dimension name (subject, aesthetic, color, etc.)

        Returns:
            Dimension definition with description, priority, examples
        """
        return self._library.get("dimensions", {}).get(dimension, {})

    def get_dimension_examples(self, dimension: str) -> List[str]:
        """
        Get example values for a dimension.

        Args:
            dimension: Dimension name

        Returns:
            List of example values
        """
        dim = self.get_dimension(dimension)
        return dim.get("examples", [])

    # =========================================================================
    # PRINT METHOD CONSTRAINTS
    # =========================================================================

    def get_all_print_methods(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all print method constraints.

        Returns:
            Dict of print_method -> constraints
        """
        return self._library.get("print_methods", {})

    def get_print_constraints(self, print_method: str) -> Dict[str, Any]:
        """
        Get constraints for a specific print method.

        Args:
            print_method: screen_print, embroidery, sublimation, etc.

        Returns:
            Constraints including:
            - max_colors: color limit or null
            - supports_gradients: bool
            - technical_template: template for technical specs
        """
        return self._library.get("print_methods", {}).get(print_method, {})

    def get_technical_template(self, print_method: str) -> Dict[str, str]:
        """
        Get technical specification template for a print method.

        Args:
            print_method: The print method

        Returns:
            Template dict with placeholders (e.g., {n}, {color_list})
        """
        method = self.get_print_constraints(print_method)
        return method.get("technical_template", {})

    # =========================================================================
    # PRODUCT SPECIFICATIONS
    # =========================================================================

    def get_all_products(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all product specifications.

        Returns:
            Dict of product_id -> specifications
        """
        return self._library.get("products", {})

    def get_product(self, product: str) -> Dict[str, Any]:
        """
        Get specifications for a specific product.

        Args:
            product: Product ID (mens_tshirt, phone_case, etc.)

        Returns:
            Product specs including:
            - print_area: {width, height, dpi}
            - default_method: default print method
            - safe_zone: optional safe zone dimensions
            - default_mode: recommended mode
        """
        return self._library.get("products", {}).get(product, {})

    def list_products(self) -> List[str]:
        """List all available product IDs."""
        return list(self._library.get("products", {}).keys())

    # =========================================================================
    # STYLE EXAMPLES (FEW-SHOT)
    # =========================================================================

    def get_all_examples(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all style examples.

        Returns:
            Dict of style_id -> example data
        """
        return self._library.get("examples", {})

    def get_style_example(self, style: str) -> Dict[str, Any]:
        """
        Get a specific style example.

        Args:
            style: Style ID (streetwear_tiger, cozy_cat, etc.)

        Returns:
            Example including:
            - product_types: applicable products
            - print_method: intended print method
            - dimensions: all dimension values
            - technical: technical specifications
        """
        return self._library.get("examples", {}).get(style, {})

    def get_example_dimensions(self, style: str) -> Dict[str, str]:
        """
        Get just the dimension values from an example.

        Args:
            style: Style ID

        Returns:
            Dict of dimension -> value
        """
        example = self.get_style_example(style)
        return example.get("dimensions", {})

    def get_example_technical(self, style: str) -> Dict[str, str]:
        """
        Get technical specs from an example.

        Args:
            style: Style ID

        Returns:
            Dict of technical spec -> value
        """
        example = self.get_style_example(style)
        return example.get("technical", {})

    def list_styles(self) -> List[str]:
        """List all available style example IDs."""
        return list(self._library.get("examples", {}).keys())

    def find_examples_for_product(self, product: str) -> List[str]:
        """
        Find style examples applicable to a product.

        Args:
            product: Product ID

        Returns:
            List of applicable style IDs
        """
        examples = self.get_all_examples()
        applicable = []
        for style_id, example in examples.items():
            if product in example.get("product_types", []):
                applicable.append(style_id)
        return applicable

    # =========================================================================
    # CONFLICT RULES
    # =========================================================================

    def get_all_conflicts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all conflict resolution rules.

        Returns:
            Dict of conflict_id -> {detect, resolve}
        """
        return self._library.get("conflicts", {})

    def get_conflict_rule(self, conflict: str) -> Dict[str, Any]:
        """
        Get a specific conflict resolution rule.

        Args:
            conflict: Conflict ID (gradient_on_screen_print, etc.)

        Returns:
            Rule with detect condition and resolve action
        """
        return self._library.get("conflicts", {}).get(conflict, {})

    # =========================================================================
    # SCENARIO TEMPLATES
    # =========================================================================

    def get_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all scenario templates.

        Returns:
            Dict of scenario -> template info
        """
        return self._library.get("scenarios", {})

    def get_scenario(self, scenario: str) -> Dict[str, Any]:
        """
        Get template for a specific scenario.

        Args:
            scenario: new_image, reference_based, or partial_edit

        Returns:
            Scenario template with fields and requirements
        """
        return self._library.get("scenarios", {}).get(scenario, {})

    # =========================================================================
    # SAFETY DEFAULTS
    # =========================================================================

    def get_safety_defaults(self) -> Dict[str, Any]:
        """
        Get safety defaults for content restrictions.

        Returns:
            Safety configuration including restrictions
        """
        return self._library.get("safety_defaults", {})

    # =========================================================================
    # FULL CONTEXT FOR PLANNER
    # =========================================================================

    def get_planning_context(
        self,
        product: Optional[str] = None,
        print_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get full context for Planner Agent to reason about.

        This bundles all relevant rules and constraints for a given
        product/print_method combination so the Planner can make decisions.

        Args:
            product: Optional product to get specific specs
            print_method: Optional print method to get specific constraints

        Returns:
            Dict with:
            - mode_rules: all mode configurations
            - dimensions: all dimension definitions
            - product_spec: product specifications (if provided)
            - print_constraints: print method constraints (if provided)
            - conflicts: relevant conflict rules
            - safety: safety defaults
        """
        context = {
            "mode_rules": self.get_mode_rules(),
            "dimensions": self.get_all_dimensions(),
            "safety": self.get_safety_defaults(),
        }

        if product:
            context["product_spec"] = self.get_product(product)
            # Get default print method from product if not specified
            if not print_method:
                print_method = context["product_spec"].get("default_method")
            # Find applicable examples
            context["applicable_examples"] = self.find_examples_for_product(product)

        if print_method:
            context["print_constraints"] = self.get_print_constraints(print_method)
            context["technical_template"] = self.get_technical_template(print_method)

        # Add relevant conflicts based on print method
        if print_method:
            conflicts = {}
            all_conflicts = self.get_all_conflicts()
            for conflict_id, rule in all_conflicts.items():
                if print_method in conflict_id:
                    conflicts[conflict_id] = rule
            context["conflicts"] = conflicts

        return context

    # =========================================================================
    # COMPOSITION CONTEXT FOR COMPOSER
    # =========================================================================

    def get_composition_context(
        self,
        mode: str,
        print_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get context for PromptComposerService to compose prompts.

        Args:
            mode: The selected mode (RELAX, STANDARD, COMPLEX)
            print_method: Optional print method for technical templates

        Returns:
            Dict with:
            - mode_rule: specific mode configuration
            - dimensions: dimension definitions
            - technical_template: print method template (if applicable)
            - conflicts: relevant conflict rules
        """
        context = {
            "mode_rule": self.get_mode_rule(mode),
            "dimensions": self.get_all_dimensions(),
        }

        if print_method:
            context["technical_template"] = self.get_technical_template(print_method)
            context["print_constraints"] = self.get_print_constraints(print_method)
            # Add relevant conflicts
            conflicts = {}
            all_conflicts = self.get_all_conflicts()
            for conflict_id, rule in all_conflicts.items():
                if print_method in conflict_id:
                    conflicts[conflict_id] = rule
            context["conflicts"] = conflicts

        return context

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def close(self) -> None:
        """Clean up resources."""
        pass

    async def __aenter__(self) -> "PromptTemplateService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# Backward compatibility alias
PromptWriterService = PromptTemplateService
PromptWriterServiceError = PromptTemplateServiceError
