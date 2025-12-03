"""
Unified Palet8 API Client
Handles all external API calls to https://api.palet8.biz for:
- User Profile operations
- Credit System operations
- Unified customer data retrieval
"""

import os
import requests
import asyncio
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GenerationType(str, Enum):
    """Image generation size types with corresponding credit costs"""
    S = "S"    # 512x512 = 10 credits
    M = "M"    # 1024x1024 = 25 credits
    L = "L"    # 1536x1536 = 50 credits
    XL = "XL"  # 2048x2048 = 100 credits


@dataclass
class ProfileData:
    """User profile data from external API"""
    customer_id: str
    email: str
    username: Optional[str] = None
    bio: Optional[str] = None
    avatar: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    social_links: Optional[Dict[str, str]] = None


@dataclass
class BalanceData:
    """Credit balance data"""
    customer_id: str
    shop_domain: Optional[str]
    balance: int
    currency: str = "credits"
    last_updated: Optional[str] = None


@dataclass
class Transaction:
    """Credit transaction record"""
    transaction_id: str
    customer_id: str
    type: str  # "credit" or "debit"
    amount: int
    reason: str
    balance_before: int
    balance_after: int
    shop_domain: Optional[str]
    metadata: Optional[Dict[str, Any]]
    timestamp: str


@dataclass
class BonusResult:
    """Daily bonus claim result"""
    bonus_awarded: bool
    amount_added: int
    new_balance: int
    streak_count: int
    message: str
    next_available: Optional[str] = None


@dataclass
class UnifiedCustomerData:
    """Combined profile + credits data"""
    profile: ProfileData
    credits: BalanceData


class Palet8APIError(Exception):
    """Base exception for Palet8 API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(self.message)


class InsufficientCreditsError(Palet8APIError):
    """Raised when user has insufficient credits (402 status)"""
    def __init__(self, current_balance: int, required: int, shortfall: int):
        self.current_balance = current_balance
        self.required = required
        self.shortfall = shortfall
        super().__init__(
            f"Insufficient credits: have {current_balance}, need {required} (short {shortfall})",
            status_code=402
        )


class Palet8APIClient:
    """
    Unified client for all Palet8 external API operations.

    Handles:
    - Profile management (get, update, avatar)
    - Credit operations (balance, add, deduct, generate)
    - Daily bonus claims
    - Transaction history
    - Unified customer data

    Features:
    - Automatic retry with exponential backoff
    - Proper error handling for all status codes
    - Type-safe responses with Pydantic models
    - Connection pooling
    - Request/response logging
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: int = 3
    ):
        self.base_url = base_url or os.getenv("PALET8_API_URL", "https://api.palet8.biz")
        self.api_key = api_key or os.getenv("PALET8_API_KEY")
        # Timeout from env var (milliseconds) or parameter (seconds)
        timeout_ms = int(os.getenv("PALET8_API_TIMEOUT", "30000"))
        self.timeout = timeout if timeout is not None else (timeout_ms / 1000)
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("PALET8_API_KEY environment variable is required")

        # Create HTTP session with connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "Palet8-Agents/1.0"
        })

        logger.info(f"Initialized Palet8APIClient: {self.base_url}")

    def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> requests.Response:
        """Synchronous request method."""
        return self.session.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            timeout=self.timeout
        )

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and error handling.

        Retry logic:
        - 401 Unauthorized: No retry (auth issue)
        - 402 Payment Required: No retry (insufficient credits)
        - 404 Not Found: No retry (resource doesn't exist)
        - 429 Rate Limit: Retry with backoff
        - 500+ Server Error: Retry with backoff
        - Network errors: Retry with backoff
        """
        try:
            # Construct full URL
            url = f"{self.base_url}{path}"

            # Log request details
            logger.info(f"[Palet8API] {method} {url} - params={params}")
            logger.info(f"[Palet8API] Headers: X-API-Key={self.api_key[:10]}..., User-Agent={self.session.headers.get('User-Agent')}")

            # Run sync request in thread pool
            response = await asyncio.to_thread(
                self._make_request,
                method=method,
                url=url,
                params=params,
                json_data=json
            )

            # Log response
            logger.info(f"[Palet8API] Response: {response.status_code} - {response.text[:500]}")

            # Handle specific status codes
            if response.status_code == 200:
                return response.json()

            elif response.status_code == 401:
                raise Palet8APIError(
                    "Authentication failed - invalid API key",
                    status_code=401
                )

            elif response.status_code == 402:
                # Insufficient credits - parse error details
                data = response.json()
                error = data.get("error", {})
                raise InsufficientCreditsError(
                    current_balance=error.get("current_balance", 0),
                    required=error.get("required", 0),
                    shortfall=error.get("shortfall", 0)
                )

            elif response.status_code == 404:
                raise Palet8APIError(
                    f"Resource not found: {path}",
                    status_code=404,
                    response_data=response.json() if response.text else None
                )

            elif response.status_code == 429:
                # Rate limit - retry with backoff
                if retry_count < self.max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Rate limited. Retrying in {wait_time}s (attempt {retry_count + 1}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                    return await self._request(method, path, params, json, retry_count + 1)
                else:
                    raise Palet8APIError(
                        "Rate limit exceeded - max retries reached",
                        status_code=429
                    )

            elif response.status_code >= 500:
                # Server error - retry with backoff
                if retry_count < self.max_retries:
                    wait_time = 2 ** retry_count
                    logger.warning(f"Server error {response.status_code}. Retrying in {wait_time}s (attempt {retry_count + 1}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                    return await self._request(method, path, params, json, retry_count + 1)
                else:
                    raise Palet8APIError(
                        f"Server error {response.status_code} - max retries reached",
                        status_code=response.status_code,
                        response_data=response.json() if response.text else None
                    )

            else:
                # Other error
                raise Palet8APIError(
                    f"API error: {response.status_code}",
                    status_code=response.status_code,
                    response_data=response.json() if response.text else None
                )

        except requests.exceptions.ConnectionError as e:
            # Network error - retry with backoff
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                logger.warning(f"Network error: {e}. Retrying in {wait_time}s (attempt {retry_count + 1}/{self.max_retries})")
                await asyncio.sleep(wait_time)
                return await self._request(method, path, params, json, retry_count + 1)
            else:
                raise Palet8APIError(
                    f"Network error - max retries reached: {e}",
                    status_code=None
                )

        except requests.exceptions.Timeout:
            raise Palet8APIError(
                f"Request timeout after {self.timeout}s",
                status_code=None
            )

    # ============================================
    # Profile Operations
    # ============================================

    async def get_profile(self, customer_id: str, email: str) -> ProfileData:
        """
        Get customer profile by customer ID and email.

        GET /api/profile/{customer_id}/{email}
        """
        data = await self._request("GET", f"/api/profile/{customer_id}/{email}")
        return ProfileData(
            customer_id=data.get("customer_id"),
            email=data.get("email"),
            username=data.get("username"),
            bio=data.get("bio"),
            avatar=data.get("avatar"),
            location=data.get("location"),
            website=data.get("website"),
            social_links=data.get("social_links")
        )

    async def update_profile(self, profile_data: Dict[str, Any]) -> ProfileData:
        """
        Create or update customer profile.

        POST /api/profile
        """
        data = await self._request("POST", "/api/profile", json=profile_data)
        return ProfileData(
            customer_id=data.get("customer_id"),
            email=data.get("email"),
            username=data.get("username"),
            bio=data.get("bio"),
            avatar=data.get("avatar"),
            location=data.get("location"),
            website=data.get("website"),
            social_links=data.get("social_links")
        )

    # ============================================
    # Credit Operations
    # ============================================

    async def get_balance(self, customer_id: str, shop_domain: Optional[str] = None) -> BalanceData:
        """
        Get credit balance for a customer.

        GET /api/v1/credits/balance/{customer_id}
        Optional: ?shop_domain=store.myshopify.com
        """
        # SINGLE-TENANT: Don't send shop_domain parameter, use legacy NULL accounts
        data = await self._request("GET", f"/api/v1/credits/balance/{customer_id}")

        return BalanceData(
            customer_id=data["customer_id"],
            shop_domain=data.get("shop_domain"),
            balance=data["balance"],
            currency="credits",
            last_updated=data.get("updated_at")
        )

    async def add_credits(
        self,
        customer_id: str,
        amount: int,
        reason: str,
        shop_domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Transaction:
        """
        Add credits to customer account (bonus, purchase, etc).

        POST /api/v1/credits/add
        """
        payload = {
            "customer_id": customer_id,
            "email": "placeholder@example.com",  # Required by API
            "shop_domain": None,  # SINGLE-TENANT: Always use legacy NULL accounts
            "amount": amount,
            "transaction_type": "addition",
            "reason": reason,
            "metadata": metadata or {}
        }

        data = await self._request("POST", "/api/v1/credits/add", json=payload)

        return Transaction(
            transaction_id=str(data["transaction_id"]),
            customer_id=customer_id,
            type="credit",
            amount=amount,
            reason=reason,
            balance_before=data["previous_balance"],
            balance_after=data["new_balance"],
            shop_domain=shop_domain,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )

    async def deduct_credits(
        self,
        customer_id: str,
        amount: int,
        reason: str,
        shop_domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Transaction:
        """
        Deduct credits from customer account.

        POST /api/v1/credits/deduct

        Raises InsufficientCreditsError if balance < amount
        """
        payload = {
            "customer_id": customer_id,
            "email": "placeholder@example.com",  # Required by API
            "shop_domain": None,  # SINGLE-TENANT: Always use legacy NULL accounts
            "amount": amount,
            "transaction_type": "deduction",
            "reason": reason,
            "metadata": metadata or {}
        }

        data = await self._request("POST", "/api/v1/credits/deduct", json=payload)

        return Transaction(
            transaction_id=str(data["transaction_id"]),
            customer_id=customer_id,
            type="debit",
            amount=amount,
            reason=reason,
            balance_before=data["previous_balance"],
            balance_after=data["new_balance"],
            shop_domain=shop_domain,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )

    async def deduct_for_generation(
        self,
        customer_id: str,
        generation_type: GenerationType,
        shop_domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Transaction:
        """
        Deduct credits for image generation (uses generation_type pricing).

        POST /api/v1/credits/generate

        Generation types and costs:
        - S (512x512): 10 credits
        - M (1024x1024): 25 credits
        - L (1536x1536): 50 credits
        - XL (2048x2048): 100 credits

        Raises InsufficientCreditsError if balance < required cost
        """
        payload = {
            "customer_id": customer_id,
            "generation_type": generation_type.value,
            "shop_domain": None,  # SINGLE-TENANT: Always use legacy NULL accounts
            "metadata": metadata or {}
        }

        data = await self._request("POST", "/api/v1/credits/generate", json=payload)

        # Get generation cost from metadata or estimate based on type
        cost_map = {"S": 10, "M": 25, "L": 50, "XL": 100}
        amount = cost_map.get(generation_type.value, 10)

        return Transaction(
            transaction_id=str(data["transaction_id"]),
            customer_id=customer_id,
            type="debit",
            amount=amount,
            reason=f"image_generation_{generation_type.value}",
            balance_before=data["previous_balance"],
            balance_after=data["new_balance"],
            shop_domain=shop_domain,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )

    async def get_transaction_history(
        self,
        customer_id: str,
        shop_domain: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        transaction_type: Optional[str] = None
    ) -> List[Transaction]:
        """
        Get transaction history for a customer.

        GET /api/v1/transactions/history/{customer_id}
        Optional: ?shop_domain=...&limit=50&offset=0&type=deduction
        """
        params = {
            "shop_domain": None,  # SINGLE-TENANT: Always use legacy NULL accounts
            "limit": limit,
            "offset": offset
        }
        if transaction_type:
            params["type"] = transaction_type

        data = await self._request("GET", f"/api/v1/transactions/history/{customer_id}", params=params)

        transactions = []
        for txn in data.get("transactions", []):
            transactions.append(Transaction(
                transaction_id=str(txn["id"]),
                customer_id=txn["customer_id"],
                type=txn["transaction_type"],
                amount=txn["amount"],
                reason=txn["reason"],
                balance_before=0,  # Not provided in response
                balance_after=txn["balance_after"],
                shop_domain=txn.get("shop_domain"),
                metadata=txn.get("metadata"),
                timestamp=txn["created_at"]
            ))

        return transactions

    async def claim_daily_bonus(self, customer_id: str, shop_domain: Optional[str] = None) -> BonusResult:
        """
        Claim daily login bonus (idempotent - can only claim once per day).

        POST /api/v1/credits/daily-bonus/{customer_id}
        Optional: ?shop_domain=store.myshopify.com

        Returns bonus_awarded=False if already claimed today
        """
        # SINGLE-TENANT: Don't send shop_domain parameter, use legacy NULL accounts
        data = await self._request("POST", f"/api/v1/credits/daily-bonus/{customer_id}")

        return BonusResult(
            bonus_awarded=data["bonus_awarded"],
            amount_added=data["data"].get("amount_added", 0),
            new_balance=data["data"].get("new_balance", 0),
            streak_count=data["data"].get("streak_count", 0),
            message=data["data"]["message"],
            next_available=data["data"].get("next_available")
        )

    async def get_generation_costs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current generation pricing.

        GET /api/v1/credits/generation-costs

        Returns:
        {
          "S": {"size": "512x512", "credit_cost": 10},
          "M": {"size": "1024x1024", "credit_cost": 25},
          "L": {"size": "1536x1536", "credit_cost": 50},
          "XL": {"size": "2048x2048", "credit_cost": 100}
        }
        """
        data = await self._request("GET", "/api/v1/credits/generation-costs")
        return data["data"]

    # ============================================
    # Unified Operations
    # ============================================

    async def get_customer_data(
        self,
        customer_id: str,
        shop_domain: Optional[str] = None
    ) -> UnifiedCustomerData:
        """
        Get customer identity and credit balance for validation.

        Agents API only needs to verify customer exists and check credits.
        Avatar/profile data is already fetched by customer app for UI display.

        Uses GET /api/v1/credits/balance/{customer_id}
        Optional: ?shop_domain=store.myshopify.com
        """
        # SINGLE-TENANT: Don't send shop_domain parameter, use legacy NULL accounts
        data = await self._request("GET", f"/api/v1/credits/balance/{customer_id}")

        # Return minimal data needed for validation (email for identity, balance for credit check)
        return UnifiedCustomerData(
            profile=ProfileData(
                customer_id=data["customer_id"],
                email=data["email"],
                username=data.get("username"),  # May be included in balance response
                bio=None,  # Not needed for agents API
                avatar=None,  # Not needed - customer app already has this
                location=None,
                website=None,
                social_links=None
            ),
            credits=BalanceData(
                customer_id=data["customer_id"],
                shop_domain=data.get("shop_domain"),
                balance=data["balance"],
                currency=data.get("currency", "credits"),
                last_updated=data.get("updated_at")
            )
        )

    # ============================================
    # Utility Methods
    # ============================================

    async def health_check(self) -> bool:
        """Check if the external API is accessible."""
        try:
            await self._request("GET", "/api/health")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client and cleanup resources."""
        await self.client.aclose()
        logger.info("Palet8APIClient closed")

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
        except Exception:
            pass
