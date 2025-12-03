"""
Prompt Manager Agent - Template retrieval and prompt assembly.

Uses Google ADK BaseAgent.
Implements hybrid routing: database templates first, LLM fallback for creative requests.
"""
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
from src.agents.base_agent import BaseAgent
from src.config.policy_loader import policy
from src.utils import get_logger

logger = get_logger(__name__)

# Import Prisma client for database access
# Note: This will be available after admin-frontend runs `prisma generate`
try:
    import sys
    from pathlib import Path
    admin_frontend_path = Path(__file__).parent.parent.parent.parent / "admin-frontend"
    sys.path.insert(0, str(admin_frontend_path))
    from prisma import Prisma
    PRISMA_AVAILABLE = True
except ImportError:
    logger.warning("Prisma client not available, using hardcoded templates")
    PRISMA_AVAILABLE = False
    Prisma = None


class PromptManagerAgent(BaseAgent):
    """
    Prompt Manager Agent handles prompt templates and refinement.

    Responsibilities:
    - Retrieve prompt templates
    - Assemble prompts with style guidance
    - Add product-specific requirements
    - Optimize prompts for different models
    """

    def __init__(self, name: str = "PromptManagerAgent") -> None:
        """Initialize Prompt Manager Agent."""
        super().__init__(name=name)

        # Initialize Prisma client if available
        self.prisma = Prisma() if PRISMA_AVAILABLE else None
        self.db_connected = False

        # Load hardcoded templates as fallback
        self.fallback_templates = self._load_templates()

        logger.info(
            "Prompt Manager Agent initialized",
            db_available=PRISMA_AVAILABLE,
            fallback_templates=len(self.fallback_templates)
        )

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve and assemble prompts for image generation.

        Uses hybrid routing:
        - Database templates (primary, ~5-10ms)
        - LLM fallback for low confidence or creative requests (~250ms)

        Args:
            input_data: Contains:
                - context: User context with original prompt
                - plan: Execution plan from Planner

        Returns:
            Refined prompts for image generation with routing metadata
        """
        context = input_data.get("context", {})
        user_prompt = context.get("prompt", "")

        logger.info("Prompt Manager processing request", prompt_length=len(user_prompt))

        # Connect to database if not already connected
        if self.prisma and not self.db_connected:
            try:
                await self.prisma.connect()
                self.db_connected = True
                logger.info("Connected to database")
            except Exception as e:
                logger.warning(f"Failed to connect to database: {e}, using fallback templates")
                self.prisma = None

        try:
            # Extract style and content
            style, content = self._parse_prompt(user_prompt)

            # Try database template first
            if self.prisma and self.db_connected:
                template, confidence, template_id = await self._find_template_db(user_prompt, style)
            else:
                # Fallback to hardcoded templates
                template = self._select_template(content)
                confidence = 0.8  # Assume reasonable confidence for fallback
                template_id = None

            # Get confidence threshold from policy
            threshold = policy.get("prompt_manager.template_confidence_threshold", 0.80)

            # Determine if we should use LLM fallback
            use_llm_fallback = False
            llm_enabled = policy.get_feature_flag("prompt_mgr_llm_enabled")

            if confidence < threshold and llm_enabled:
                logger.info(f"Confidence {confidence} < {threshold}, using LLM fallback")
                use_llm_fallback = True
                # TODO: Add LLM fallback in Phase 3
                # For now, proceed with low-confidence template

            # Assemble final prompt
            final_prompt = self._assemble_prompt_from_template(template, style, user_prompt)

            # Generate variations for testing
            variations = self._generate_variations(final_prompt, num_variations=3)

            # Add model-specific optimizations
            from src.models.schemas import ImageModel
            optimized_prompts = {
                ImageModel.FLUX.value: self._optimize_for_flux(final_prompt),
                ImageModel.GEMINI.value: self._optimize_for_imagen(final_prompt),
            }

            # Update template usage stats if using DB
            if template_id and self.prisma and self.db_connected:
                await self._update_template_usage(template_id)

            logger.info(
                "Prompts assembled",
                confidence=confidence,
                template=template.get("name", "fallback"),
                variations_count=len(variations),
                used_llm=use_llm_fallback
            )

            # DIAGNOSTIC: Print assembled prompts
            print("=" * 80)
            print("🔍 DIAGNOSTIC - Prompt Manager")
            print(f"  User Prompt: '{user_prompt}'")
            print(f"  Template: {template.get('name', 'fallback')}")
            print(f"  Final Prompt: '{final_prompt}'")
            print(f"  Optimized for Flux: '{optimized_prompts.get(ImageModel.FLUX.value, '')}'")
            print(f"  Optimized for Gemini: '{optimized_prompts.get(ImageModel.GEMINI.value, '')}'")
            print(f"  Style: '{style or template.get('style', 'realistic')}'")
            print("=" * 80)

            return {
                "success": True,
                "prompts": {
                    "primary": final_prompt,
                    "variations": variations,
                    "optimized": optimized_prompts,
                    "style": style or template.get("style", "realistic"),
                    "content_type": template.get("category", "general"),
                },
                "routing_metadata": {
                    "mode": "llm" if use_llm_fallback else "database",
                    "used_llm": use_llm_fallback,
                    "confidence": confidence,
                    "template_id": template_id,
                    "template_name": template.get("name", "fallback"),
                    "fallback_used": not self.db_connected
                },
                "context": context,
            }

        except Exception as e:
            logger.error("Prompt assembly failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": f"Prompt assembly failed: {str(e)}",
                "routing_metadata": {
                    "mode": "error",
                    "error": str(e)
                },
                "context": context,
            }
        finally:
            # Disconnect from database (will reconnect on next run)
            if self.prisma and self.db_connected:
                await self.prisma.disconnect()
                self.db_connected = False

    async def _find_template_db(
        self,
        user_prompt: str,
        style: Optional[str]
    ) -> Tuple[Dict[str, Any], float, Optional[str]]:
        """
        Query database for best matching template.

        Args:
            user_prompt: User prompt
            style: Detected style

        Returns:
            Tuple of (template_dict, confidence_score, template_id)
        """
        prompt_lower = user_prompt.lower()

        # Determine category from prompt
        category = self._detect_category(prompt_lower)

        # Query templates from database
        templates = await self.prisma.template.find_many(
            where={
                "isActive": True,
                "category": category,
                "style": style or "realistic"
            },
            order={"usageCount": "desc"}  # Prefer commonly used templates
        )

        if not templates:
            # No templates found, use fallback
            logger.warning(f"No templates found for category={category}, style={style}")
            fallback_template = self.fallback_templates.get("general", {})
            return fallback_template, 0.5, None

        # Score each template based on keyword match
        best_template = None
        best_score = 0.0
        best_id = None

        for template in templates:
            score = self._score_template_match(prompt_lower, template)
            if score > best_score:
                best_score = score
                best_template = template
                best_id = template.id

        # Convert Prisma model to dict
        template_dict = {
            "name": best_template.name,
            "category": best_template.category,
            "promptText": best_template.promptText,
            "style": best_template.style,
            "tags": best_template.tags,
            "language": best_template.language
        }

        logger.debug(
            f"Selected template",
            template=best_template.name,
            confidence=best_score,
            category=category
        )

        return template_dict, best_score, best_id

    def _detect_category(self, prompt_lower: str) -> str:
        """
        Detect category from prompt keywords.

        Args:
            prompt_lower: Lowercased prompt

        Returns:
            Category string
        """
        if any(kw in prompt_lower for kw in ["product", "commercial", "merchandise"]):
            return "product"
        elif any(kw in prompt_lower for kw in ["lifestyle", "in-use", "authentic"]):
            return "lifestyle"
        elif any(kw in prompt_lower for kw in ["creative", "artistic", "abstract"]):
            return "creative"
        return "product"  # Default to product

    def _score_template_match(self, prompt_lower: str, template) -> float:
        """
        Score how well a template matches the prompt.

        Args:
            prompt_lower: Lowercased prompt
            template: Template object from database

        Returns:
            Confidence score (0.0-1.0)
        """
        # Count tag matches
        tag_matches = sum(1 for tag in template.tags if tag.lower() in prompt_lower)
        max_tags = len(template.tags) if template.tags else 1

        # Keyword coverage
        coverage = tag_matches / max_tags

        # Bonus for high usage count (proven templates)
        usage_bonus = min(template.usageCount / 100.0, 0.2)  # Max 0.2 bonus

        # Bonus for high acceptance rate
        accept_bonus = (template.acceptRate or 0.5) * 0.1  # Max 0.1 bonus

        total_score = coverage + usage_bonus + accept_bonus

        return min(total_score, 1.0)  # Cap at 1.0

    def _assemble_prompt_from_template(
        self,
        template: Dict[str, Any],
        style: Optional[str],
        user_prompt: str
    ) -> str:
        """
        Assemble final prompt from database template.

        Args:
            template: Template dictionary
            style: Style override
            user_prompt: Original user prompt

        Returns:
            Assembled prompt
        """
        # DB templates use {product} placeholder
        if "promptText" in template:
            # Extract product/subject from user prompt (simplified)
            product = self._extract_product(user_prompt)

            # DIAGNOSTIC: Log template replacement process
            print("=" * 80)
            print("🔍 DIAGNOSTIC - Template Replacement")
            print(f"  Raw Template promptText: '{template['promptText']}'")
            print(f"  User Prompt: '{user_prompt}'")
            print(f"  Extracted Product: '{product}'")
            print(f"  Has {{product}} placeholder: {'{product}' in template['promptText']}")

            # Replace placeholder
            prompt = template["promptText"].replace("{product}", product)

            print(f"  After Replacement: '{prompt}'")

            # Add style override if provided
            if style:
                prompt = f"{style} style, {prompt}"
                print(f"  After Style Addition: '{prompt}'")

            print("=" * 80)

            return prompt
        else:
            # Fallback template format
            print("=" * 80)
            print("🔍 DIAGNOSTIC - Using Fallback Template Format")
            print(f"  Template: {template}")
            print(f"  User Prompt: '{user_prompt}'")
            print("=" * 80)
            return self._assemble_prompt(template, style, user_prompt)

    def _extract_product(self, prompt: str) -> str:
        """
        Extract product name from prompt.

        Simple heuristic: remove common keywords and return remaining text.

        Args:
            prompt: User prompt

        Returns:
            Extracted product name
        """
        # Remove common keywords
        remove_keywords = [
            "professional", "high quality", "photo", "image", "picture",
            "on white background", "white background", "white bg",
            "studio lighting", "commercial"
        ]

        product = prompt.lower()
        for keyword in remove_keywords:
            product = product.replace(keyword, "")

        # Clean up whitespace
        product = " ".join(product.split())

        return product.strip() or "product"

    async def _update_template_usage(self, template_id: str) -> None:
        """
        Update template usage statistics.

        Args:
            template_id: Template ID to update
        """
        try:
            await self.prisma.template.update(
                where={"id": template_id},
                data={
                    "usageCount": {"increment": 1},
                    "lastUsed": datetime.now()
                }
            )
            logger.debug(f"Updated usage for template {template_id}")
        except Exception as e:
            logger.warning(f"Failed to update template usage: {e}")

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Load prompt templates for different use cases.

        Returns:
            Dictionary of templates
        """
        return {
            "product_design": {
                "type": "product",
                "prefix": "High-quality product design:",
                "suffix": "Professional photography, studio lighting, white background",
                "negative": "low quality, blurry, distorted, watermark",
            },
            "tshirt": {
                "type": "apparel",
                "prefix": "T-shirt design:",
                "suffix": "Flat lay, centered, isolated on white background",
                "negative": "wrinkled, people wearing, low resolution",
            },
            "mug": {
                "type": "drinkware",
                "prefix": "Coffee mug design:",
                "suffix": "Ceramic material, 11oz standard size, 360-degree wrap design",
                "negative": "cracked, chipped, poor print quality",
            },
            "abstract_art": {
                "type": "art",
                "prefix": "Abstract artistic design:",
                "suffix": "High resolution, vibrant colors, modern aesthetic",
                "negative": "realistic, photographic, low contrast",
            },
            "character": {
                "type": "character",
                "prefix": "Character illustration:",
                "suffix": "Consistent style, clear details, professional quality",
                "negative": "inconsistent features, malformed anatomy",
            },
            "general": {
                "type": "general",
                "prefix": "",
                "suffix": "High quality, detailed, professional",
                "negative": "low quality, blurry, distorted",
            },
        }

    def _parse_prompt(self, prompt: str) -> tuple[Optional[str], str]:
        """
        Parse prompt into style and content.

        Args:
            prompt: User prompt

        Returns:
            Tuple of (style, content)
        """
        # Look for style indicators
        style_keywords = [
            "minimalist", "modern", "vintage", "retro", "abstract",
            "photorealistic", "cartoon", "anime", "watercolor", "oil painting"
        ]

        detected_style = None
        prompt_lower = prompt.lower()

        for style in style_keywords:
            if style in prompt_lower:
                detected_style = style
                break

        return detected_style, prompt

    def _select_template(self, content: str) -> Dict[str, Any]:
        """
        Select appropriate template based on content.

        Args:
            content: Prompt content

        Returns:
            Selected template
        """
        content_lower = content.lower()

        # Check for specific product types
        if "t-shirt" in content_lower or "tshirt" in content_lower:
            return self.fallback_templates["tshirt"]
        elif "mug" in content_lower or "cup" in content_lower:
            return self.fallback_templates["mug"]
        elif "character" in content_lower or "mascot" in content_lower:
            return self.fallback_templates["character"]
        elif "abstract" in content_lower:
            return self.fallback_templates["abstract_art"]
        elif any(word in content_lower for word in ["product", "merchandise", "design"]):
            return self.fallback_templates["product_design"]

        return self.fallback_templates["general"]

    def _assemble_prompt(
        self, template: Dict[str, Any], style: Optional[str], content: str
    ) -> str:
        """
        Assemble final prompt from components.

        Args:
            template: Prompt template
            style: Style description
            content: Content description

        Returns:
            Assembled prompt
        """
        parts = []

        # Add template prefix
        if template.get("prefix"):
            parts.append(template["prefix"])

        # Add style if detected
        if style:
            parts.append(f"{style} style,")

        # Add main content
        parts.append(content)

        # Add template suffix
        if template.get("suffix"):
            parts.append(f", {template['suffix']}")

        return " ".join(parts)

    def _generate_variations(self, base_prompt: str, num_variations: int = 3) -> list[str]:
        """
        Generate prompt variations for testing.

        Args:
            base_prompt: Base prompt
            num_variations: Number of variations

        Returns:
            List of prompt variations
        """
        variations = []

        # Quality modifiers
        quality_mods = [
            "ultra high quality, 8K resolution",
            "professional quality, sharp details",
            "masterpiece quality, perfect composition",
        ]

        for i in range(num_variations):
            if i < len(quality_mods):
                variation = f"{base_prompt}, {quality_mods[i]}"
                variations.append(variation)

        return variations

    def _optimize_for_flux(self, prompt: str) -> str:
        """
        Optimize prompt for Flux 1 Kontext.

        Args:
            prompt: Base prompt

        Returns:
            Optimized prompt
        """
        # Flux excels at style transfer and character consistency
        optimizations = [
            "consistent style",
            "coherent composition",
            "8x fast inference quality",
        ]

        return f"{prompt}, {', '.join(optimizations)}"

    def _optimize_for_imagen(self, prompt: str) -> str:
        """
        Optimize prompt for Imagen 3.

        Args:
            prompt: Base prompt

        Returns:
            Optimized prompt
        """
        # Imagen 3 excels at photorealism and natural language
        optimizations = [
            "photorealistic details",
            "natural lighting",
            "world-class quality",
        ]

        return f"{prompt}, {', '.join(optimizations)}"
