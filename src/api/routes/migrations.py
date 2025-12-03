"""
Database migrations endpoint - temporary for applying schema changes.
"""
from fastapi import APIRouter, HTTPException
from src.database import prisma
from src.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/migrations", tags=["migrations"])


@router.post("/add-model-name-column")
async def add_model_name_column():
    """
    Add modelName column to AgentLog table.
    This is a temporary endpoint for applying the migration.
    """
    try:
        logger.info("Attempting to add modelName column to AgentLog table")

        # Execute raw SQL to add column
        await prisma.execute_raw(
            'ALTER TABLE "AgentLog" ADD COLUMN IF NOT EXISTS "modelName" TEXT;'
        )

        logger.info("Successfully added modelName column")

        # Verify column exists
        result = await prisma.query_raw(
            '''
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'AgentLog' AND column_name = 'modelName';
            '''
        )

        return {
            "success": True,
            "message": "Migration completed successfully",
            "verification": result
        }

    except Exception as e:
        logger.error(f"Migration failed: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Migration failed: {str(e)}"
        )
