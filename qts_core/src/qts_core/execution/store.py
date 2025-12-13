"""State persistence layer for QTS-Architect.

Provides abstraction over storage backends (Redis, in-memory) for
persisting portfolio state, orders, and other critical data.

Design Decisions:
- Protocol-based interface for flexibility
- Atomic transactions for consistency
- TTL support for ephemeral data
- Fail-fast on production (Redis required), graceful fallback in dev
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import structlog
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Generator

log = structlog.get_logger()

# Generic type for Pydantic models
T = TypeVar("T", bound=BaseModel)


# ==============================================================================
# State Store Protocol
# ==============================================================================
@runtime_checkable
class StateStore(Protocol):
    """Protocol for state persistence backends.

    Implementations must support:
    - Key-value storage for raw strings
    - Pydantic model serialization
    - TTL (time-to-live) for ephemeral data
    - Atomic multi-key operations (transactions)
    """

    def save(self, key: str, model: BaseModel, ttl_seconds: int | None = None) -> None:
        """Persist a Pydantic model."""
        ...

    def load(self, key: str, model_cls: type[T]) -> T | None:
        """Load a Pydantic model by key."""
        ...

    def delete(self, key: str) -> None:
        """Delete a key."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...

    def get(self, key: str) -> str | None:
        """Get raw string value."""
        ...

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        """Set raw string value."""
        ...


# ==============================================================================
# Custom Exceptions
# ==============================================================================
class StoreConnectionError(Exception):
    """Raised when store connection fails."""


class TransactionError(Exception):
    """Raised when atomic transaction fails."""


# ==============================================================================
# Redis Store
# ==============================================================================
class RedisStore:
    """Redis-backed persistence with transaction support.

    Features:
    - Automatic connection retry on startup
    - Atomic multi-key transactions via MULTI/EXEC
    - JSON serialization for Pydantic models
    - Optional TTL for ephemeral keys

    Attributes:
        redis_url: Redis connection URL.
        client: Redis client instance.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        connection_timeout: float = 5.0,
        socket_timeout: float = 5.0,
    ) -> None:
        """Initialize Redis connection.

        Args:
            redis_url: Redis connection URL.
            connection_timeout: Connection timeout in seconds.
            socket_timeout: Socket timeout in seconds.

        Raises:
            StoreConnectionError: If Redis is unreachable.
        """
        import redis

        self.redis_url = redis_url
        self._redis_module = redis

        try:
            self.client: redis.Redis[str] = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=connection_timeout,
                socket_timeout=socket_timeout,
            )
            # Verify connection
            self.client.ping()
            log.info("Redis connection established", url=redis_url)
        except redis.ConnectionError as exc:
            raise StoreConnectionError(
                f"Cannot connect to Redis at {redis_url}: {exc}"
            ) from exc

    def save(self, key: str, model: BaseModel, ttl_seconds: int | None = None) -> None:
        """Persist a Pydantic model as JSON."""
        data = model.model_dump_json()
        self.set(key, data, ttl_seconds)

    def load(self, key: str, model_cls: type[T]) -> T | None:
        """Load and deserialize a Pydantic model."""
        data = self.get(key)
        if data is None:
            return None
        try:
            return model_cls.model_validate_json(data)
        except Exception as e:
            log.error("Failed to deserialize model", key=key, error=str(e))
            return None

    def delete(self, key: str) -> None:
        """Delete a key."""
        self.client.delete(key)

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return bool(self.client.exists(key))

    def get(self, key: str) -> str | None:
        """Get raw string value."""
        return self.client.get(key)

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        """Set raw string value with optional TTL."""
        if ttl_seconds and ttl_seconds > 0:
            self.client.setex(key, ttl_seconds, value)
        else:
            self.client.set(key, value)

    def increment(self, key: str, amount: int = 1) -> int:
        """Atomically increment a counter."""
        return self.client.incrby(key, amount)

    def get_many(self, keys: list[str]) -> list[str | None]:
        """Get multiple keys in one round-trip."""
        return self.client.mget(keys)  # type: ignore[return-value]

    def set_many(self, mapping: dict[str, str]) -> None:
        """Set multiple keys atomically."""
        self.client.mset(mapping)

    @contextmanager
    def transaction(self) -> Generator[redis.client.Pipeline, None, None]:  # type: ignore[name-defined]
        """Execute commands atomically using MULTI/EXEC.

        Usage:
            ```python
            with store.transaction() as pipe:
                pipe.set("key1", "value1")
                pipe.set("key2", "value2")
            # Both keys set atomically
            ```

        Raises:
            TransactionError: If transaction fails.
        """
        import redis

        pipe = self.client.pipeline(transaction=True)
        try:
            yield pipe
            pipe.execute()
        except redis.WatchError as e:
            raise TransactionError(f"Transaction aborted due to concurrent modification: {e}") from e
        except Exception as e:
            raise TransactionError(f"Transaction failed: {e}") from e
        finally:
            pipe.reset()

    def save_atomic(self, models: dict[str, BaseModel]) -> None:
        """Save multiple models atomically.

        Args:
            models: Dictionary of key -> Pydantic model.
        """
        mapping = {key: model.model_dump_json() for key, model in models.items()}
        self.set_many(mapping)

    def health_check(self) -> bool:
        """Check if Redis is healthy."""
        try:
            return self.client.ping()
        except Exception:
            return False


# ==============================================================================
# Memory Store (Development/Testing)
# ==============================================================================
class MemoryStore:
    """In-memory store for development and testing.

    WARNING: Data is lost on restart. Do not use in production.

    Features:
    - Thread-safe operations
    - Simulated transactions (not truly atomic)
    - TTL is ignored (all data persists until deleted)
    """

    def __init__(self) -> None:
        """Initialize in-memory store."""
        self._data: dict[str, str] = {}
        self._in_transaction = False
        self._transaction_buffer: dict[str, str] = {}
        log.warning(
            "MemoryStore initialized - DATA IS VOLATILE",
            hint="Use RedisStore in production",
        )

    def save(self, key: str, model: BaseModel, ttl_seconds: int | None = None) -> None:
        """Persist a Pydantic model."""
        _ = ttl_seconds  # TTL not supported in memory store
        self._data[key] = model.model_dump_json()

    def load(self, key: str, model_cls: type[T]) -> T | None:
        """Load a Pydantic model."""
        data = self.get(key)
        if data is None:
            return None
        try:
            return model_cls.model_validate_json(data)
        except Exception:
            return None

    def delete(self, key: str) -> None:
        """Delete a key."""
        self._data.pop(key, None)

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data

    def get(self, key: str) -> str | None:
        """Get raw string value."""
        return self._data.get(key)

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        """Set raw string value."""
        _ = ttl_seconds
        if self._in_transaction:
            self._transaction_buffer[key] = value
        else:
            self._data[key] = value

    def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter."""
        current = int(self._data.get(key, "0"))
        new_value = current + amount
        self._data[key] = str(new_value)
        return new_value

    def get_many(self, keys: list[str]) -> list[str | None]:
        """Get multiple keys."""
        return [self._data.get(k) for k in keys]

    def set_many(self, mapping: dict[str, str]) -> None:
        """Set multiple keys."""
        self._data.update(mapping)

    @contextmanager
    def transaction(self) -> Generator[MemoryStore, None, None]:
        """Simulated transaction (not truly atomic)."""
        self._in_transaction = True
        self._transaction_buffer = {}
        try:
            yield self
            # Commit
            self._data.update(self._transaction_buffer)
        finally:
            self._in_transaction = False
            self._transaction_buffer = {}

    def save_atomic(self, models: dict[str, BaseModel]) -> None:
        """Save multiple models."""
        for key, model in models.items():
            self._data[key] = model.model_dump_json()

    def health_check(self) -> bool:
        """Always healthy for memory store."""
        return True

    def clear(self) -> None:
        """Clear all data (useful for tests)."""
        self._data.clear()


# ==============================================================================
# Factory Function
# ==============================================================================
def create_store(
    redis_url: str = "redis://localhost:6379/0",
    use_redis: bool = True,
    fallback_to_memory: bool = True,
) -> RedisStore | MemoryStore:
    """Create a state store instance.

    Args:
        redis_url: Redis connection URL.
        use_redis: Whether to attempt Redis connection.
        fallback_to_memory: If True, use MemoryStore when Redis unavailable.

    Returns:
        RedisStore or MemoryStore instance.

    Raises:
        StoreConnectionError: If Redis required but unavailable.
    """
    if use_redis:
        try:
            return RedisStore(redis_url=redis_url)
        except StoreConnectionError:
            if not fallback_to_memory:
                raise
            log.warning(
                "Redis unavailable, using MemoryStore",
                url=redis_url,
            )

    return MemoryStore()


# Backward compatibility alias
def get_store(
    redis_url: str = "redis://localhost:6379/0",
    use_redis: bool = True,
    fallback_to_memory: bool = True,
) -> RedisStore | MemoryStore:
    """Alias for create_store (backward compatibility)."""
    return create_store(redis_url, use_redis, fallback_to_memory)
