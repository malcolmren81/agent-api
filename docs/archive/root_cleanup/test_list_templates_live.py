#!/usr/bin/env python3
"""
Smoke Test: List Templates (Live API Call)
Tests that the ProductTemplateService can fetch templates from admin-api.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.product_template_service import product_template_service


async def test_list_templates():
    """Test fetching templates from admin-api"""
    try:
        print("="*60)
        print("Smoke Test: List Templates (Live API)")
        print("="*60)

        # Fetch apparel templates
        print("\n[1] Fetching apparel templates...")
        apparel_templates = await product_template_service.list_templates(category="apparel")

        print(f"✓ Fetched {len(apparel_templates)} apparel templates")

        if apparel_templates:
            template = apparel_templates[0]
            print(f"\nSample Template:")
            print(f"  Name: {template.get('name')}")
            print(f"  Category: {template.get('category')}")
            print(f"  Price: ${template.get('price')}")
            print(f"  Active: {template.get('isActive')}")

        # Fetch all templates
        print("\n[2] Fetching all templates...")
        all_templates = await product_template_service.list_templates()

        print(f"✓ Fetched {len(all_templates)} total templates")

        # Test different categories
        print("\n[3] Testing category filters...")
        categories = ["apparel", "drinkware", "wall-art", "accessories"]

        for category in categories:
            templates = await product_template_service.list_templates(category=category)
            print(f"  {category:15} {len(templates)} templates")

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)

        await product_template_service.close()
        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        await product_template_service.close()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_list_templates())
    sys.exit(0 if result else 1)
