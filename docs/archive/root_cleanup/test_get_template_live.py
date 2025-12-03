#!/usr/bin/env python3
"""
Smoke Test: Get Single Template (Live API Call)
Tests fetching a single template by ID from admin-api.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.product_template_service import product_template_service


async def test_get_template():
    """Test fetching single template by ID"""
    try:
        print("="*60)
        print("Smoke Test: Get Single Template (Live API)")
        print("="*60)

        # First, get list of templates to find a valid ID
        print("\n[1] Fetching template list to get valid ID...")
        templates = await product_template_service.list_templates()

        if not templates:
            print("⚠ WARNING: No templates in database")
            print("Cannot test get_template() without existing templates")
            print("This is not a failure - database may be empty")
            await product_template_service.close()
            return True

        template_id = templates[0]["id"]
        print(f"✓ Found {len(templates)} templates")
        print(f"  Using template ID: {template_id}")

        # Fetch single template
        print(f"\n[2] Fetching template by ID...")
        template = await product_template_service.get_template(template_id)

        print(f"✓ Successfully fetched template")

        # Display template details
        print(f"\nTemplate Details:")
        print(f"  ID: {template['id']}")
        print(f"  Name: {template['name']}")
        print(f"  Category: {template['category']}")
        print(f"  Price: ${template['price']}")
        print(f"  Cost: ${template.get('cost', 'N/A')}")
        print(f"  Active: {template['isActive']}")
        print(f"  Suspended: {template.get('isSuspended', False)}")

        # Check images
        if template.get('images'):
            print(f"  Images: {len(template['images'])}")
            main_image = next((img for img in template['images'] if img.get('isMain')), None)
            if main_image:
                print(f"    Main: {main_image['url'][:60]}...")

        # Check design area
        if template.get('designArea'):
            da = template['designArea']
            print(f"  Design Area:")
            print(f"    Position: ({da.get('x')}, {da.get('y')})")
            print(f"    Size: {da.get('width')}x{da.get('height')} {da.get('unit', 'pixels')}")

        # Verify required fields
        print(f"\n[3] Verifying required fields...")
        required_fields = ["id", "name", "category", "images", "price"]
        for field in required_fields:
            if field not in template:
                raise AssertionError(f"Missing required field: {field}")
            print(f"  ✓ {field}")

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
    result = asyncio.run(test_get_template())
    sys.exit(0 if result else 1)
