"""
Database client module.

Provides centralized Prisma client with proper connection lifecycle management.
"""
import asyncio
from prisma import Prisma
from src.utils import get_logger

logger = get_logger(__name__)

# Global Prisma client instance
prisma = Prisma()


async def connect_db() -> None:
    """
    Connect to the database.

    Should be called once during application startup.
    Retries connection with exponential backoff.
    """
    print("=" * 80)
    print("DATABASE CONNECTION: Starting connection process")
    print("=" * 80)

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            print(f"[Attempt {attempt + 1}/{max_retries}] Connecting to database...")
            logger.info(f"Connecting to database (attempt {attempt + 1}/{max_retries})...")

            await prisma.connect()

            print(f"✅ Database connected successfully on attempt {attempt + 1}")
            logger.info("Database connected successfully")

            # Verify connection is actually working
            print("Verifying database connection...")
            if prisma.is_connected():
                print("✅ Database connection verified - client reports connected")
            else:
                print("⚠️ WARNING: Prisma client reports NOT connected despite successful connect()")

            print("=" * 80)
            return

        except Exception as e:
            error_msg = f"Database connection attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}"
            print(f"❌ {error_msg}")
            logger.warning(error_msg)

            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            else:
                # All retries exhausted - log warning and continue
                print("=" * 80)
                print("⚠️ WARNING: Failed to connect to database after all retries")
                print(f"Last error: {type(e).__name__}: {str(e)}")
                print("Application will start but database operations will fail")
                print("=" * 80)
                logger.error("Failed to connect to database after all retries", exc_info=True)
                logger.warning("Application starting without database connection")


async def disconnect_db() -> None:
    """
    Disconnect from the database.

    Should be called once during application shutdown.
    """
    try:
        print("=" * 80)
        print("DATABASE DISCONNECTION: Starting disconnect process")
        logger.info("Disconnecting from database...")

        await prisma.disconnect()

        print("✅ Database disconnected successfully")
        logger.info("Database disconnected successfully")
        print("=" * 80)
    except Exception as e:
        print(f"❌ Failed to disconnect from database: {type(e).__name__}: {str(e)}")
        logger.error(f"Failed to disconnect from database: {e}", exc_info=True)
        raise


async def get_db_client() -> Prisma:
    """
    Get the database client instance.

    This is a FastAPI dependency that returns the global Prisma client.
    The client should already be connected via the lifespan manager.

    Returns:
        Prisma: The connected Prisma client
    """
    return prisma
