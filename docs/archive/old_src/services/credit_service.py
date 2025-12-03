"""
Credit Service - Centralized credit management and billing.

This service is the single source of truth for all credit-related operations.
Agents should NOT directly interact with credit APIs - they only calculate costs.

Responsibilities:
- Fetch user profile and credit balance
- Pre-generation credit validation
- Post-generation credit deduction
- Transaction tracking and error handling

Architecture:
- Wraps Palet8APIClient for external API calls
- Called by orchestrator, NOT by individual agents
- Agents return cost calculations; service handles billing
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass
from src.api.clients import (
    Palet8APIClient,
    GenerationType,
    InsufficientCreditsError,
    Palet8APIError,
    UnifiedCustomerData,
    Transaction
)
from src.agents.base_agent import AgentContext
from src.utils import get_logger

logger = get_logger(__name__)

# Minimum credits required for any generation (S = 10 credits)
MIN_CREDITS_REQUIRED = 10


@dataclass
class CreditCheckResult:
    """Result of pre-generation credit check"""
    sufficient: bool
    current_balance: int
    required_credits: int
    shortfall: Optional[int] = None
    username: Optional[str] = None
    avatar: Optional[str] = None


@dataclass
class CreditDeductionResult:
    """Result of post-generation credit deduction"""
    success: bool
    transaction_id: Optional[str] = None
    balance_before: int = 0
    balance_after: int = 0
    amount_deducted: int = 0
    error: Optional[str] = None


class CreditService:
    """
    Centralized service for all credit operations.

    This service handles:
    1. Fetching user profile + credit balance
    2. Pre-generation validation (do they have enough credits?)
    3. Post-generation deduction (charge them after success)
    4. Error handling and logging
    """

    def __init__(self, api_client: Optional[Palet8APIClient] = None):
        """
        Initialize Credit Service.

        Args:
            api_client: Palet8 API client (defaults to new instance)
        """
        self.api_client = api_client or Palet8APIClient()
        logger.info("CreditService initialized")

    async def fetch_and_populate_user_data(self, context: AgentContext) -> bool:
        """
        Fetch user profile and credit balance, populate AgentContext.

        This is called ONCE at the beginning of the pipeline to populate
        the context with user data from the external API.

        Args:
            context: AgentContext to populate

        Returns:
            bool: True if successful, False otherwise

        Side effects:
            Populates context.username, context.avatar, context.credit_balance
        """
        try:
            logger.info(
                "Fetching customer data from external API",
                customer_id=context.customer_id
                # SINGLE-TENANT: shop_domain removed
            )

            # SINGLE-TENANT: Force shop_domain=None to use legacy NULL accounts
            customer_data = await self.api_client.get_customer_data(
                customer_id=context.customer_id,
                shop_domain=None
            )

            # Populate context
            context.username = customer_data.profile.username
            context.avatar = customer_data.profile.avatar
            context.credit_balance = customer_data.credits.balance

            logger.info(
                "Customer data fetched successfully",
                username=context.username,
                credit_balance=context.credit_balance
            )

            return True

        except Palet8APIError as e:
            logger.error(
                "Failed to fetch customer data",
                error=str(e),
                status_code=e.status_code,
                customer_id=context.customer_id
            )
            return False

    async def check_sufficient_credits(
        self,
        context: AgentContext,
        required_credits: int
    ) -> CreditCheckResult:
        """
        Check if user has sufficient credits for generation.

        Args:
            context: AgentContext with user data (must be populated first)
            required_credits: Number of credits required

        Returns:
            CreditCheckResult with validation details
        """
        current_balance = context.credit_balance
        sufficient = current_balance >= required_credits

        result = CreditCheckResult(
            sufficient=sufficient,
            current_balance=current_balance,
            required_credits=required_credits,
            shortfall=None if sufficient else (required_credits - current_balance),
            username=context.username,
            avatar=context.avatar
        )

        if not sufficient:
            logger.warning(
                "Insufficient credits",
                current_balance=current_balance,
                required=required_credits,
                shortfall=result.shortfall,
                customer_id=context.customer_id
            )

        return result

    async def deduct_credits_for_generation(
        self,
        context: AgentContext,
        generation_type: GenerationType,
        num_images: int,
        prompt: str,
        model: str
    ) -> CreditDeductionResult:
        """
        Deduct credits after successful image generation.

        This is called AFTER images are successfully generated.
        If deduction fails, images are still returned (customer experience priority)
        but the failure is logged for manual reconciliation.

        Args:
            context: AgentContext with customer data
            generation_type: Size/type of generation (S/M/L/XL)
            num_images: Number of images generated
            prompt: Generation prompt (first 100 chars stored)
            model: Image model used (flux/imagen3)

        Returns:
            CreditDeductionResult with transaction details

        Side effects:
            Updates context.credit_balance with new balance
        """
        balance_before = context.credit_balance

        try:
            logger.info(
                "Deducting credits for generation",
                customer_id=context.customer_id,
                generation_type=generation_type.value,
                num_images=num_images
            )

            # SINGLE-TENANT: Force shop_domain=None to use legacy NULL accounts
            transaction = await self.api_client.deduct_for_generation(
                customer_id=context.customer_id,
                generation_type=generation_type,
                shop_domain=None,
                metadata={
                    "task_id": context.task_id,
                    "prompt": prompt[:100],
                    "model": model,
                    "num_images": num_images,
                }
            )

            # Update context with new balance
            context.credit_balance = transaction.balance_after

            logger.info(
                "Credits deducted successfully",
                transaction_id=transaction.transaction_id,
                amount_deducted=transaction.amount,
                balance_before=balance_before,
                balance_after=transaction.balance_after
            )

            return CreditDeductionResult(
                success=True,
                transaction_id=transaction.transaction_id,
                balance_before=balance_before,
                balance_after=transaction.balance_after,
                amount_deducted=transaction.amount
            )

        except InsufficientCreditsError as e:
            # This shouldn't happen after pre-check, but handle it gracefully
            logger.error(
                "Insufficient credits during deduction",
                current_balance=e.current_balance,
                required=e.required,
                customer_id=context.customer_id
            )

            # Log for manual reconciliation
            logger.critical(
                "COMPENSATION REQUIRED: Generation succeeded but credit deduction failed",
                task_id=context.task_id,
                customer_id=context.customer_id,
                generation_type=generation_type.value,
                error="insufficient_credits"
            )

            return CreditDeductionResult(
                success=False,
                balance_before=balance_before,
                balance_after=balance_before,  # No change
                amount_deducted=0,
                error=f"Insufficient credits: {e.message}"
            )

        except Palet8APIError as e:
            logger.error(
                "Credit deduction API error",
                error=str(e),
                status_code=e.status_code,
                customer_id=context.customer_id
            )

            # Log for manual reconciliation
            logger.critical(
                "COMPENSATION REQUIRED: Generation succeeded but credit deduction failed",
                task_id=context.task_id,
                customer_id=context.customer_id,
                generation_type=generation_type.value,
                error=str(e)
            )

            return CreditDeductionResult(
                success=False,
                balance_before=balance_before,
                balance_after=balance_before,  # No change
                amount_deducted=0,
                error=f"API error: {e.message}"
            )

    def get_minimum_required_credits(self) -> int:
        """
        Get minimum credits required for any generation.

        Returns:
            int: Minimum credits (currently 10 for size S)
        """
        return MIN_CREDITS_REQUIRED
