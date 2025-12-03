"""
Session Service

Manages Shopify session lifecycle including creation, retrieval,
and cleanup of expired sessions.
"""

from prisma import Prisma
from prisma.models import Session
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SessionServiceError(Exception):
    """Base exception for SessionService errors"""
    pass


class SessionService:
    """
    Service for managing Shopify Session operations.

    Handles session storage for Shopify authentication,
    including automatic cleanup of expired sessions.
    """

    def __init__(self, prisma: Prisma):
        """
        Initialize SessionService with Prisma client.

        Args:
            prisma: Connected Prisma client instance
        """
        self.prisma = prisma

    async def create_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update a session.

        Args:
            session_data: Session data from Shopify

        Returns:
            Dictionary representation of created/updated session

        Raises:
            SessionServiceError: If session creation/update fails
        """
        try:
            logger.info(f"Creating session for shop={session_data.get('shop')}")

            # Upsert session (create or update if exists)
            session = await self.prisma.session.upsert(
                where={"id": session_data["id"]},
                data={
                    "create": session_data,
                    "update": session_data
                }
            )

            logger.info(f"Session created/updated: id={session.id}")
            return session.model_dump()

        except Exception as e:
            logger.error(f"Failed to create session: {e}", exc_info=True)
            raise SessionServiceError(f"Failed to create session: {str(e)}") from e

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session dictionary or None if not found/expired
        """
        try:
            session = await self.prisma.session.find_unique(
                where={"id": session_id}
            )

            if not session:
                return None

            # Check if expired
            if session.expires and session.expires < datetime.now():
                logger.info(f"Session expired: id={session_id}")
                return None

            return session.model_dump()

        except Exception as e:
            logger.error(f"Failed to get session: {e}", exc_info=True)
            raise SessionServiceError(f"Failed to retrieve session: {str(e)}") from e

    async def get_session_by_shop(self, shop: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve most recent session for a shop.

        Args:
            shop: Shop domain

        Returns:
            Session dictionary or None if not found
        """
        try:
            session = await self.prisma.session.find_first(
                where={"shop": shop},
                order_by={"id": "desc"}  # Get most recent
            )

            if not session:
                return None

            # Check if expired
            if session.expires and session.expires < datetime.now():
                logger.info(f"Session expired for shop: {shop}")
                return None

            return session.model_dump()

        except Exception as e:
            logger.error(f"Failed to get session by shop: {e}", exc_info=True)
            raise SessionServiceError(f"Failed to retrieve session: {str(e)}") from e

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a specific session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully, False if not found
        """
        try:
            logger.info(f"Deleting session: id={session_id}")

            await self.prisma.session.delete(
                where={"id": session_id}
            )

            logger.info(f"Session deleted: id={session_id}")
            return True

        except Exception as e:
            if "Record to delete does not exist" in str(e):
                logger.warning(f"Session not found for deletion: id={session_id}")
                return False

            logger.error(f"Failed to delete session: {e}", exc_info=True)
            raise SessionServiceError(f"Failed to delete session: {str(e)}") from e

    async def delete_expired_sessions(self) -> int:
        """
        Delete all expired sessions from database.

        This should be run periodically (e.g., daily cron job) to clean up
        expired sessions and prevent database bloat.

        Returns:
            Number of sessions deleted
        """
        try:
            now = datetime.now()

            logger.info("Deleting expired sessions")

            result = await self.prisma.session.delete_many(
                where={
                    "expires": {
                        "lt": now
                    }
                }
            )

            deleted_count = result
            logger.info(f"Deleted {deleted_count} expired sessions")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete expired sessions: {e}", exc_info=True)
            raise SessionServiceError(f"Failed to delete expired sessions: {str(e)}") from e

    async def delete_sessions_for_shop(self, shop: str) -> int:
        """
        Delete all sessions for a specific shop.

        Useful when a shop uninstalls the app or revokes access.

        Args:
            shop: Shop domain

        Returns:
            Number of sessions deleted
        """
        try:
            logger.info(f"Deleting all sessions for shop: {shop}")

            result = await self.prisma.session.delete_many(
                where={"shop": shop}
            )

            deleted_count = result
            logger.info(f"Deleted {deleted_count} sessions for shop: {shop}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete sessions for shop: {e}", exc_info=True)
            raise SessionServiceError(f"Failed to delete sessions: {str(e)}") from e


# Example usage
"""
from services.session_service import SessionService

# Initialize service
session_service = SessionService(prisma)

# Create/update session
session = await session_service.create_session({
    "id": "offline_myshop.myshopify.com",
    "shop": "myshop.myshopify.com",
    "state": "abc123",
    "isOnline": False,
    "scope": "read_products,write_products",
    "expires": None,  # Offline sessions don't expire
    "accessToken": "shpat_xxxx",
    "userId": None,
    "accountOwner": True
})

# Get session
session = await session_service.get_session("offline_myshop.myshopify.com")
if session:
    print(f"Found session for {session['shop']}")

# Get session by shop
session = await session_service.get_session_by_shop("myshop.myshopify.com")

# Clean up expired sessions (run daily)
deleted = await session_service.delete_expired_sessions()
print(f"Deleted {deleted} expired sessions")

# Delete all sessions for a shop (on uninstall)
deleted = await session_service.delete_sessions_for_shop("myshop.myshopify.com")
"""
