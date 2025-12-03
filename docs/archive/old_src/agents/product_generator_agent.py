"""
Product Generator Agent - Composites images onto product mockups.

Uses Google ADK BaseAgent.
"""
from typing import Any, Dict, List, Optional
from src.agents.base_agent import BaseAgent
from PIL import Image
import io
import base64
from pathlib import Path
from src.utils import get_logger
from src.models.schemas import ProductType
from src.services.product_template_service import product_template_service, ProductTemplateNotFoundError
from src.services.character_service import character_service, CharacterNotFoundError
# from src.services.aesthetic_service import aesthetic_service, AestheticNotFoundError  # Temporarily disabled for debugging

logger = get_logger(__name__)

# Path to product template files
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "assets" / "templates"


class ProductGeneratorAgent(BaseAgent):
    """
    Product Generator Agent composites images onto product templates.

    Responsibilities:
    - Load product templates (T-shirts, mugs, etc.)
    - Composite approved images
    - Maintain image quality
    - Generate multiple product variations
    """

    def __init__(self, name: str = "ProductGeneratorAgent") -> None:
        """Initialize Product Generator Agent."""
        super().__init__(name=name)

        # Load product templates
        self.templates = self._load_templates()

        logger.info("Product Generator Agent initialized")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate product mockups from approved images.

        Args:
            input_data: Contains:
                - best_image: Approved image from Evaluation Agent
                - context: User context
                - product_types: Types of products to generate

        Returns:
            Generated product mockups
        """
        import time

        best_image = input_data.get("best_image", {})
        context = input_data.get("context", {})

        # Phase 7.1.2: Fetch product template if template_id is provided
        template_id = context.get("template_id")
        product_template = None

        if template_id:
            logger.info(
                "Product template selected for generation",
                template_id=template_id,
                category=context.get("template_category")
            )

            try:
                # Fetch product template from admin-api
                product_template = await product_template_service.get_template(template_id)

                logger.info(
                    "✓ Product template fetched successfully",
                    template_name=product_template.get("name"),
                    category=product_template.get("category"),
                    design_area=product_template.get("designArea")
                )

                # Store in context for Phase 7.2 (product creation)
                context["product_template"] = product_template
                context["design_area"] = product_template.get("designArea")

            except ProductTemplateNotFoundError:
                logger.warning(
                    "Product template not found",
                    template_id=template_id
                )
                # Don't fail the workflow, just log and continue without template
                context["template_error"] = f"Template {template_id} not found"

            except Exception as e:
                logger.error(
                    "Failed to fetch product template",
                    template_id=template_id,
                    error=str(e),
                    exc_info=True
                )
                # Don't fail the workflow, just log and continue
                context["template_error"] = f"Template fetch failed: {str(e)}"
        else:
            logger.debug("No product template selected, using default mockup generation")

        # Phase 7.4: Fetch character reference if character_id is provided
        character_id = context.get("character_id")
        character_reference = None

        if character_id:
            logger.info(
                "Character reference selected for generation",
                character_id=character_id
            )

            try:
                # Fetch character reference from admin-api
                character_reference = await character_service.get_character(character_id)

                logger.info(
                    "✓ Character reference fetched successfully",
                    character_name=character_reference.get("name"),
                    character_display_id=character_reference.get("displayId"),
                    category=character_reference.get("categoryRel", {}).get("name"),
                    creative_assets_count=len(character_reference.get("creativeAssets", []))
                )

                # Store in context for use in prompt enhancement and image generation
                context["character_reference"] = character_reference
                context["character_name"] = character_reference.get("name")
                context["character_main_image"] = character_reference.get("mainImage")
                context["character_creative_assets"] = character_reference.get("creativeAssets", [])

            except CharacterNotFoundError:
                logger.warning(
                    "Character reference not found",
                    character_id=character_id
                )
                # Don't fail the workflow, just log and continue without character
                context["character_error"] = f"Character {character_id} not found"

            except Exception as e:
                logger.error(
                    "Failed to fetch character reference",
                    character_id=character_id,
                    error=str(e),
                    exc_info=True
                )
                # Don't fail the workflow, just log and continue
                context["character_error"] = f"Character fetch failed: {str(e)}"
        else:
            logger.debug("No character reference selected, proceeding without character")

        # Phase 7.3: Fetch aesthetic reference if aesthetic_id is provided
        # Temporarily disabled for debugging
        # aesthetic_id = context.get("aesthetic_id")
        # aesthetic_reference = None

        # if aesthetic_id:
        #     logger.info(
        #         "Aesthetic reference selected for generation",
        #         aesthetic_id=aesthetic_id
        #     )

        #     try:
        #         # Fetch aesthetic reference from admin-api
        #         aesthetic_reference = await aesthetic_service.get_aesthetic(aesthetic_id)

        #         logger.info(
        #             "✓ Aesthetic reference fetched successfully",
        #             aesthetic_name=aesthetic_reference.get("name"),
        #             aesthetic_display_id=aesthetic_reference.get("displayId"),
        #             mood_board_count=len(aesthetic_reference.get("moodBoardImages", []))
        #         )

        #         # Store in context for use in prompt enhancement and image generation
        #         context["aesthetic_reference"] = aesthetic_reference
        #         context["aesthetic_name"] = aesthetic_reference.get("name")
        #         context["aesthetic_main_image"] = aesthetic_reference.get("mainImage")
        #         context["aesthetic_mood_board"] = aesthetic_reference.get("moodBoardImages", [])

        #     except AestheticNotFoundError:
        #         logger.warning(
        #             "Aesthetic reference not found",
        #             aesthetic_id=aesthetic_id
        #         )
        #         # Don't fail the workflow, just log and continue without aesthetic
        #         context["aesthetic_error"] = f"Aesthetic {aesthetic_id} not found"

        #     except Exception as e:
        #         logger.error(
        #             "Failed to fetch aesthetic reference",
        #             aesthetic_id=aesthetic_id,
        #             error=str(e),
        #             exc_info=True
        #         )
        #         # Don't fail the workflow, just log and continue
        #         context["aesthetic_error"] = f"Aesthetic fetch failed: {str(e)}"
        # else:
        #     logger.debug("No aesthetic reference selected, proceeding with default style")

        raw_product_types = context.get("product_types", [ProductType.TSHIRT, ProductType.MUG])

        # Convert string product types to ProductType enums
        product_types = []
        for pt in raw_product_types:
            if isinstance(pt, ProductType):
                # Already an enum
                product_types.append(pt)
            elif isinstance(pt, str):
                # Convert string to enum - handle "t-shirt" → "tshirt" format differences
                normalized = pt.lower().replace('-', '').replace('_', '').replace(' ', '')
                mapping = {
                    'tshirt': ProductType.TSHIRT,
                    'mug': ProductType.MUG,
                    'poster': ProductType.POSTER,
                    'phonecase': ProductType.PHONE_CASE,
                }
                if normalized in mapping:
                    product_types.append(mapping[normalized])
                else:
                    logger.warning(f"Unknown product type '{pt}', skipping")

        # Fallback to defaults if no valid types
        if not product_types:
            product_types = [ProductType.TSHIRT, ProductType.MUG]

        if not best_image:
            logger.warning("No approved image provided for product generation")
            return {
                "success": False,
                "error": "No approved image available",
                "context": context,
            }

        logger.info(
            "Product Generator starting",
            image_id=best_image.get("image_id"),
            num_product_types=len(product_types),
        )

        try:
            start_time = time.time()

            # Get image data
            image_base64 = best_image.get("base64_data", "")
            if not image_base64:
                raise ValueError("No image data available")

            # Decode image
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))

            # Generate products for each type
            products = []

            for product_type in product_types:
                product_image = await self._composite_on_product(
                    image, product_type
                )

                # Convert back to base64
                buffer = io.BytesIO()
                product_image.save(buffer, format="PNG")
                product_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                products.append({
                    "product_id": f"{best_image.get('image_id')}-{product_type.value}",
                    "product_type": product_type.value,
                    "base64_data": product_base64,
                    "width": product_image.width,
                    "height": product_image.height,
                    "metadata": {
                        "width": product_image.width,
                        "height": product_image.height,
                        "original_image_id": best_image.get('image_id'),
                        "product_type": product_type.value,
                        "format": "PNG",
                    }
                })

            generation_time = time.time() - start_time

            logger.info(
                "Product generation complete",
                num_products=len(products),
                generation_time=generation_time,
            )

            return {
                "success": True,
                "products": products,
                "product_metadata": {
                    "num_products": len(products),
                    "generation_time": generation_time,
                    "source_image_id": best_image.get("image_id"),
                },
                "context": context,
            }

        except Exception as e:
            logger.error("Product generation failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": f"Product generation failed: {str(e)}",
                "context": context,
            }

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Load product templates with file paths.

        Returns:
            Dictionary of product templates
        """
        return {
            ProductType.TSHIRT.value: {
                "name": "T-Shirt",
                "design_area": (400, 400),  # Design area on shirt
                "position": (300, 450),  # Position on template (matches template file)
                "template_size": (1000, 1200),
                "template_file": TEMPLATES_DIR / "tshirt_template.png",
            },
            ProductType.MUG.value: {
                "name": "Coffee Mug",
                "design_area": (350, 350),
                "position": (325, 300),  # Position on template (matches template file)
                "template_size": (1000, 800),
                "template_file": TEMPLATES_DIR / "mug_template.png",
            },
            ProductType.POSTER.value: {
                "name": "Poster",
                "design_area": (600, 800),
                "position": (200, 200),  # Position on template (matches template file)
                "template_size": (1000, 1200),
                "template_file": TEMPLATES_DIR / "poster_template.png",
            },
            ProductType.PHONE_CASE.value: {
                "name": "Phone Case",
                "design_area": (300, 450),  # Updated to match template
                "position": (350, 250),  # Position on template (matches template file)
                "template_size": (1000, 800),
                "template_file": TEMPLATES_DIR / "phone_case_template.png",
            },
        }

    async def _composite_on_product(
        self, design_image: Image.Image, product_type: ProductType
    ) -> Image.Image:
        """
        Composite design onto product template.

        Args:
            design_image: Design image to composite
            product_type: Type of product

        Returns:
            Composited product image
        """
        template_info = self.templates.get(product_type.value, {})

        if not template_info:
            raise ValueError(f"Unknown product type: {product_type}")

        # Get template dimensions
        design_area = template_info["design_area"]
        position = template_info["position"]
        template_file = template_info.get("template_file")

        # Resize design to fit design area
        design_resized = design_image.resize(design_area, Image.Resampling.LANCZOS)

        # Load product template from file
        if template_file and template_file.exists():
            product_image = Image.open(template_file).convert("RGBA")
        else:
            # Fallback: create blank template if file not found
            logger.warning(f"Template file not found: {template_file}, using blank template")
            template_size = template_info["template_size"]
            product_image = Image.new("RGBA", template_size, (255, 255, 255, 255))

        # Composite design onto product template
        # Use alpha channel for blending if design has transparency
        if design_resized.mode == "RGBA":
            product_image.paste(design_resized, position, design_resized)
        else:
            product_image.paste(design_resized, position)

        # Add product-specific effects
        product_image = self._apply_product_effects(product_image, product_type)

        return product_image

    def _apply_product_effects(
        self, product_image: Image.Image, product_type: ProductType
    ) -> Image.Image:
        """
        Apply product-specific visual effects.

        Args:
            product_image: Product image
            product_type: Product type

        Returns:
            Image with effects applied
        """
        # In production, apply realistic effects like:
        # - Fabric texture for t-shirts
        # - Mug curve/wrap for mugs
        # - Frame shadows for posters
        # - Phone case edges

        # For now, return as-is
        # TODO: Implement realistic product effects

        return product_image
