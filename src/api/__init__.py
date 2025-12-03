"""API module."""
# Removed: from .main import app
# This was causing circular import when src.api.clients was imported
# App should only be imported directly by uvicorn: uvicorn src.api.main:app

__all__ = []
