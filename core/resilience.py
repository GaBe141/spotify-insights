"""
Enhanced resilience system for API interactions and error handling.
Implements retry patterns, circuit breakers, and rate limiting.
"""

import asyncio
import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

import aiohttp

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker tracking."""

    failures: int = 0
    successes: int = 0
    last_failure_time: datetime | None = None
    state: CircuitState = CircuitState.CLOSED
    open_until: datetime | None = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class EnhancedResilience:
    """
    Comprehensive resilience system for music discovery APIs.

    Features:
    - Exponential backoff with jitter
    - Circuit breaker pattern
    - Rate limiting with token bucket
    - Async/await support
    - Detailed metrics and logging
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.circuit_breakers: dict[str, CircuitBreakerMetrics] = {}
        self.rate_limiters: dict[str, dict[str, Any]] = {}
        self.request_history: list[dict[str, Any]] = []

    def with_retry(
        self, config: RetryConfig | None = None, exceptions: tuple = (Exception,)
    ) -> Callable:
        """
        Decorator that retries functions with exponential backoff and jitter.

        Args:
            config: Retry configuration
            exceptions: Tuple of exceptions to catch and retry
        """
        if config is None:
            config = RetryConfig()

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                last_exception = None

                for attempt in range(1, config.max_attempts + 1):
                    try:
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            result = func(*args, **kwargs)

                        # Log successful retry
                        if attempt > 1:
                            self.logger.info(
                                f"Function {func.__name__} succeeded on attempt {attempt}"
                            )
                        return result

                    except exceptions as e:
                        last_exception = e

                        if attempt == config.max_attempts:
                            # Last attempt failed
                            self.logger.error(
                                f"Function {func.__name__} failed after {config.max_attempts} attempts: {str(e)}"
                            )
                            break

                        # Calculate delay with exponential backoff and jitter
                        delay = min(
                            config.base_delay * (config.exponential_base ** (attempt - 1)),
                            config.max_delay,
                        )

                        if config.jitter:
                            # Add random jitter (±25% of delay)
                            import random

                            jitter_amount = delay * 0.25
                            delay += random.uniform(-jitter_amount, jitter_amount)

                        self.logger.warning(
                            f"Attempt {attempt}/{config.max_attempts} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {delay:.2f}s"
                        )

                        await asyncio.sleep(delay)

                # All attempts failed
                raise last_exception

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                # For sync functions, run the async wrapper in an event loop
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                return loop.run_until_complete(async_wrapper(*args, **kwargs))

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
    ) -> Callable:
        """
        Circuit breaker pattern implementation.

        Args:
            name: Unique name for this circuit
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreakerMetrics()

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T | None:
                circuit = self.circuit_breakers[name]
                current_time = datetime.now()

                # Check circuit state
                if circuit.state == CircuitState.OPEN:
                    if circuit.open_until and current_time < circuit.open_until:
                        self.logger.warning(
                            f"Circuit '{name}' is OPEN. Request blocked. "
                            f"Will retry after {circuit.open_until}"
                        )
                        return None
                    else:
                        # Transition to half-open
                        circuit.state = CircuitState.HALF_OPEN
                        self.logger.info(f"Circuit '{name}' transitioning to HALF_OPEN")

                try:
                    # Execute the function
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    # Success - reset circuit breaker
                    if circuit.state == CircuitState.HALF_OPEN:
                        circuit.state = CircuitState.CLOSED
                        self.logger.info(f"Circuit '{name}' CLOSED - recovery successful")

                    circuit.failures = 0
                    circuit.successes += 1
                    return result

                except expected_exception as e:
                    circuit.failures += 1
                    circuit.last_failure_time = current_time

                    self.logger.warning(f"Circuit '{name}' failure #{circuit.failures}: {str(e)}")

                    # Check if we should open the circuit
                    if circuit.failures >= failure_threshold:
                        circuit.state = CircuitState.OPEN
                        circuit.open_until = current_time + timedelta(seconds=recovery_timeout)

                        self.logger.error(
                            f"Circuit '{name}' OPENED due to {circuit.failures} failures. "
                            f"Will attempt recovery at {circuit.open_until}"
                        )

                    raise e

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T | None:
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                return loop.run_until_complete(async_wrapper(*args, **kwargs))

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def rate_limit(
        self, name: str, requests_per_minute: int = 60, burst_size: int | None = None
    ) -> Callable:
        """
        Token bucket rate limiter.

        Args:
            name: Unique name for this rate limiter
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size (defaults to requests_per_minute)
        """
        if burst_size is None:
            burst_size = requests_per_minute

        if name not in self.rate_limiters:
            self.rate_limiters[name] = {
                "tokens": burst_size,
                "last_refill": datetime.now(),
                "requests_per_minute": requests_per_minute,
                "burst_size": burst_size,
            }

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                limiter = self.rate_limiters[name]
                current_time = datetime.now()

                # Refill tokens based on time elapsed
                time_passed = (current_time - limiter["last_refill"]).total_seconds()
                tokens_to_add = (time_passed / 60.0) * limiter["requests_per_minute"]

                limiter["tokens"] = min(limiter["burst_size"], limiter["tokens"] + tokens_to_add)
                limiter["last_refill"] = current_time

                # Check if we have tokens available
                if limiter["tokens"] < 1:
                    wait_time = 60.0 / limiter["requests_per_minute"]
                    self.logger.warning(
                        f"Rate limit exceeded for '{name}'. Waiting {wait_time:.2f}s"
                    )
                    await asyncio.sleep(wait_time)
                    limiter["tokens"] = 1

                # Consume a token
                limiter["tokens"] -= 1

                # Execute function
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                return loop.run_until_complete(async_wrapper(*args, **kwargs))

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def health_check(self, timeout: float = 5.0) -> dict[str, Any]:
        """
        Perform health check on all circuit breakers and rate limiters.

        Args:
            timeout: Timeout for health check operations

        Returns:
            Health status report
        """
        current_time = datetime.now()

        # Check circuit breakers
        circuit_status = {}
        for name, circuit in self.circuit_breakers.items():
            circuit_status[name] = {
                "state": circuit.state.value,
                "failures": circuit.failures,
                "successes": circuit.successes,
                "last_failure": (
                    circuit.last_failure_time.isoformat() if circuit.last_failure_time else None
                ),
                "open_until": circuit.open_until.isoformat() if circuit.open_until else None,
            }

        # Check rate limiters
        rate_limiter_status = {}
        for name, limiter in self.rate_limiters.items():
            rate_limiter_status[name] = {
                "available_tokens": int(limiter["tokens"]),
                "requests_per_minute": limiter["requests_per_minute"],
                "burst_size": limiter["burst_size"],
                "last_refill": limiter["last_refill"].isoformat(),
            }

        return {
            "timestamp": current_time.isoformat(),
            "circuit_breakers": circuit_status,
            "rate_limiters": rate_limiter_status,
            "total_requests": len(self.request_history),
        }

    def log_request(
        self,
        endpoint: str,
        method: str = "GET",
        status_code: int | None = None,
        response_time: float | None = None,
        error: str | None = None,
    ) -> None:
        """
        Log API request for monitoring and analytics.

        Args:
            endpoint: API endpoint called
            method: HTTP method
            status_code: Response status code
            response_time: Response time in seconds
            error: Error message if request failed
        """
        request_log = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time": response_time,
            "error": error,
        }

        self.request_history.append(request_log)

        # Keep only last 1000 requests to prevent memory issues
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]

        # Log based on status
        if error:
            self.logger.error(f"Request failed: {endpoint} - {error}")
        elif status_code and status_code >= 400:
            self.logger.warning(f"Request returned {status_code}: {endpoint}")
        else:
            self.logger.debug(f"Request successful: {endpoint} ({response_time:.2f}s)")

    def get_performance_metrics(self, hours: int = 24) -> dict[str, Any]:
        """
        Get performance metrics for the last N hours.

        Args:
            hours: Number of hours to analyze

        Returns:
            Performance metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter recent requests
        recent_requests = [
            req
            for req in self.request_history
            if datetime.fromisoformat(req["timestamp"]) > cutoff_time
        ]

        if not recent_requests:
            return {"message": "No requests in the specified time period"}

        # Calculate metrics
        total_requests = len(recent_requests)
        successful_requests = len(
            [r for r in recent_requests if not r.get("error") and (r.get("status_code", 200) < 400)]
        )
        failed_requests = total_requests - successful_requests

        # Calculate average response time
        response_times = [r["response_time"] for r in recent_requests if r.get("response_time")]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        # Group by endpoint
        endpoint_stats = {}
        for req in recent_requests:
            endpoint = req["endpoint"]
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {"total": 0, "errors": 0, "avg_time": 0}

            endpoint_stats[endpoint]["total"] += 1
            if req.get("error") or (req.get("status_code", 200) >= 400):
                endpoint_stats[endpoint]["errors"] += 1

        return {
            "time_period_hours": hours,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (
                (successful_requests / total_requests) * 100 if total_requests > 0 else 0
            ),
            "average_response_time": avg_response_time,
            "endpoint_breakdown": endpoint_stats,
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_resilience():
        """Test the resilience system."""
        resilience = EnhancedResilience()

        # Test retry with simulated failures
        @resilience.with_retry(RetryConfig(max_attempts=3, base_delay=0.1))
        async def flaky_api_call():
            import random

            if random.random() < 0.7:  # 70% chance of failure
                raise aiohttp.ClientError("Simulated network error")
            return {"status": "success", "data": "Important music data"}

        # Test circuit breaker
        @resilience.circuit_breaker("test_api", failure_threshold=2, recovery_timeout=5)
        async def unreliable_api():
            import random

            if random.random() < 0.8:  # 80% chance of failure
                raise Exception("API is down")
            return {"status": "success"}

        # Test rate limiter
        @resilience.rate_limit("test_rate_limit", requests_per_minute=10)
        async def rate_limited_api():
            return {"timestamp": datetime.now().isoformat()}

        print("Testing resilience system...")

        # Test retries
        try:
            result = await flaky_api_call()
            print(f"✓ Retry test passed: {result}")
        except Exception as e:
            print(f"✗ Retry test failed: {e}")

        # Test circuit breaker
        for i in range(5):
            try:
                result = await unreliable_api()
                print(f"✓ Circuit breaker test {i+1}: {result}")
            except Exception as e:
                print(f"✗ Circuit breaker test {i+1}: {e}")

        # Test rate limiter
        for i in range(3):
            result = await rate_limited_api()
            print(f"✓ Rate limit test {i+1}: {result}")

        # Show health check
        health = resilience.health_check()
        print("\n📊 Health Check:")
        print(f"Circuit Breakers: {list(health['circuit_breakers'].keys())}")
        print(f"Rate Limiters: {list(health['rate_limiters'].keys())}")

        # Show performance metrics
        metrics = resilience.get_performance_metrics(1)  # Last hour
        print("\n📈 Performance Metrics:")
        print(f"Total Requests: {metrics.get('total_requests', 0)}")
        print(f"Success Rate: {metrics.get('success_rate', 0):.1f}%")

    # Run the test
    asyncio.run(test_resilience())
