"""
Template API routes.

Provides endpoints for managing prompt templates.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from src.database import prisma
from src.utils import get_logger
from src.models.schemas import (
    CreateTemplateRequest,
    UpdateTemplateRequest,
    TemplateResponse,
    TemplateListResponse,
    TemplateStatsResponse,
    TemplateCategory,
)

logger = get_logger(__name__)
router = APIRouter()



def template_to_response(template) -> TemplateResponse:
    """Convert Prisma Template model to TemplateResponse."""
    return TemplateResponse(
        id=template.id,
        name=template.name,
        category=template.category,
        promptText=template.promptText,
        style=template.style,
        tags=template.tags or [],
        language=template.language,
        isActive=template.isActive,
        createdBy=template.createdBy,
        usageCount=template.usageCount,
        acceptRate=template.acceptRate,
        avgScore=template.avgScore,
        lastUsed=template.lastUsed,
        source=template.source,
        createdAt=template.createdAt,
        updatedAt=template.updatedAt,
    )


@router.get("/templates", response_model=TemplateListResponse)
async def list_templates(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    category: Optional[TemplateCategory] = None,
    is_active: Optional[bool] = None,
    search: Optional[str] = None,
    language: Optional[str] = None,
) -> TemplateListResponse:
    """
    List templates with filtering and pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of results per page
        category: Filter by category (product/lifestyle/creative)
        is_active: Filter by active status
        search: Search in name and tags
        language: Filter by language code

    Returns:
        Paginated list of templates
    """
    try:

        # Build where clause
        where = {}
        if category:
            where["category"] = category.value
        if is_active is not None:
            where["isActive"] = is_active
        if language:
            where["language"] = language
        if search:
            # Search in name and tags
            where["OR"] = [
                {"name": {"contains": search, "mode": "insensitive"}},
                {"tags": {"has": search}},
            ]

        # Get total count
        total = await prisma.template.count(where=where)

        # Get paginated results
        skip = (page - 1) * page_size
        templates = await prisma.template.find_many(
            where=where,
            skip=skip,
            take=page_size,
            order={"usageCount": "desc"},  # Most used first
        )

        # Convert to response models
        template_responses = [template_to_response(t) for t in templates]

        has_more = (skip + page_size) < total


        return TemplateListResponse(
            templates=template_responses,
            total=total,
            page=page,
            pageSize=page_size,
            hasMore=has_more,
        )

    except Exception as e:
        logger.error("Error fetching templates", error_detail=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/{template_id}", response_model=TemplateResponse)
async def get_template(template_id: str) -> TemplateResponse:
    """
    Get a single template by ID.

    Args:
        template_id: Template ID

    Returns:
        Template details
    """
    try:

        template = await prisma.template.find_unique(where={"id": template_id})

        if not template:
            raise HTTPException(
                status_code=404, detail=f"Template {template_id} not found"
            )

        response = template_to_response(template)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error fetching template", template_id=template_id, error_detail=str(e), exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates", response_model=TemplateResponse, status_code=201)
async def create_template(request: CreateTemplateRequest) -> TemplateResponse:
    """
    Create a new template.

    Args:
        request: Template creation request

    Returns:
        Created template
    """
    try:

        # Check if name already exists
        existing = await prisma.template.find_unique(where={"name": request.name})
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Template with name '{request.name}' already exists",
            )

        # Create template
        template = await prisma.template.create(
            data={
                "name": request.name,
                "category": request.category.value,
                "promptText": request.promptText,
                "style": request.style,
                "tags": request.tags,
                "language": request.language,
                "createdBy": request.createdBy,
                "source": request.source.value,
            }
        )

        response = template_to_response(template)


        logger.info(
            "Template created",
            template_id=template.id,
            name=template.name,
            category=template.category,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating template", error_detail=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/templates/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: str, request: UpdateTemplateRequest
) -> TemplateResponse:
    """
    Update an existing template.

    Args:
        template_id: Template ID
        request: Template update request

    Returns:
        Updated template
    """
    try:

        # Check if template exists
        template = await prisma.template.find_unique(where={"id": template_id})
        if not template:
            raise HTTPException(
                status_code=404, detail=f"Template {template_id} not found"
            )

        # Check if updating name conflicts with existing template
        if request.name and request.name != template.name:
            existing = await prisma.template.find_unique(where={"name": request.name})
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Template with name '{request.name}' already exists",
                )

        # Build update data (only include non-None fields)
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.category is not None:
            update_data["category"] = request.category.value
        if request.promptText is not None:
            update_data["promptText"] = request.promptText
        if request.style is not None:
            update_data["style"] = request.style
        if request.tags is not None:
            update_data["tags"] = request.tags
        if request.language is not None:
            update_data["language"] = request.language
        if request.isActive is not None:
            update_data["isActive"] = request.isActive

        # Update template
        updated_template = await prisma.template.update(
            where={"id": template_id}, data=update_data
        )

        response = template_to_response(updated_template)


        logger.info(
            "Template updated",
            template_id=template_id,
            updates=list(update_data.keys()),
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error updating template",
            template_id=template_id,
            error_detail=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/templates/{template_id}")
async def delete_template(template_id: str, hard_delete: bool = Query(False)):
    """
    Delete a template.

    Args:
        template_id: Template ID
        hard_delete: If True, permanently delete. If False, soft delete (set isActive=False)

    Returns:
        Success message
    """
    try:

        # Check if template exists
        template = await prisma.template.find_unique(where={"id": template_id})
        if not template:
            raise HTTPException(
                status_code=404, detail=f"Template {template_id} not found"
            )

        if hard_delete:
            # Permanently delete
            await prisma.template.delete(where={"id": template_id})
            message = f"Template {template_id} permanently deleted"
        else:
            # Soft delete (set isActive=False)
            await prisma.template.update(
                where={"id": template_id}, data={"isActive": False}
            )
            message = f"Template {template_id} deactivated"


        logger.info(
            "Template deleted",
            template_id=template_id,
            hard_delete=hard_delete,
        )

        return {"message": message, "template_id": template_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error deleting template",
            template_id=template_id,
            error_detail=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/category/{category}", response_model=List[TemplateResponse])
async def list_templates_by_category(category: TemplateCategory) -> List[TemplateResponse]:
    """
    Get all active templates for a specific category.

    Args:
        category: Template category (product/lifestyle/creative)

    Returns:
        List of templates in the category, ordered by usage count
    """
    try:

        templates = await prisma.template.find_many(
            where={"category": category.value, "isActive": True},
            order={"usageCount": "desc"},
        )

        template_responses = [template_to_response(t) for t in templates]


        return template_responses

    except Exception as e:
        logger.error(
            "Error fetching templates by category",
            category=category.value,
            error_detail=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/stats", response_model=TemplateStatsResponse)
async def get_template_stats() -> TemplateStatsResponse:
    """
    Get template usage statistics.

    Returns:
        Template statistics including usage counts, accept rates, etc.
    """
    try:

        # Get all templates
        all_templates = await prisma.template.find_many()

        if not all_templates:
            return TemplateStatsResponse(
                totalTemplates=0,
                activeTemplates=0,
                totalUsages=0,
                averageAcceptRate=0.0,
                averageScore=0.0,
                mostUsedTemplate=None,
                categoryBreakdown={},
            )

        # Calculate statistics
        total_templates = len(all_templates)
        active_templates = sum(1 for t in all_templates if t.isActive)
        total_usages = sum(t.usageCount for t in all_templates)

        # Calculate average accept rate
        accept_rates = [t.acceptRate for t in all_templates if t.acceptRate is not None]
        avg_accept_rate = (
            sum(accept_rates) / len(accept_rates) if accept_rates else 0.0
        )

        # Calculate average score
        scores = [t.avgScore for t in all_templates if t.avgScore is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Find most used template
        most_used = max(all_templates, key=lambda t: t.usageCount, default=None)

        # Category breakdown
        category_breakdown = {}
        for template in all_templates:
            category = template.category
            category_breakdown[category] = category_breakdown.get(category, 0) + 1


        return TemplateStatsResponse(
            totalTemplates=total_templates,
            activeTemplates=active_templates,
            totalUsages=total_usages,
            averageAcceptRate=avg_accept_rate,
            averageScore=avg_score,
            mostUsedTemplate=template_to_response(most_used) if most_used else None,
            categoryBreakdown=category_breakdown,
        )

    except Exception as e:
        logger.error("Error calculating template stats", error_detail=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
