"""Dependency Injection Container for Audora.

This module provides a lightweight dependency injection container for managing
service lifecycles and dependencies throughout the application.
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceLifetime:
    """Service lifetime options."""

    SINGLETON = "singleton"  # Single instance per container
    TRANSIENT = "transient"  # New instance every time
    SCOPED = "scoped"  # Single instance per scope


class Container:
    """Lightweight dependency injection container.

    Supports singleton, transient, and scoped service lifetimes.
    Automatically resolves constructor dependencies.

    Example:
        ```python
        container = Container()

        # Register services
        container.register(
            DataStore,
            lambda: DataStore("data/db.sqlite"),
            lifetime=ServiceLifetime.SINGLETON
        )

        container.register(
            Analytics,
            lambda c: Analytics(c.resolve(DataStore)),
            lifetime=ServiceLifetime.TRANSIENT
        )

        # Resolve services
        analytics = container.resolve(Analytics)
        ```
    """

    def __init__(self) -> None:
        """Initialize the container."""
        self._services: dict[type, dict[str, Any]] = {}
        self._singletons: dict[type, Any] = {}
        self._scoped_instances: dict[type, Any] = {}
        self._scope_active = False
        logger.debug("Dependency injection container initialized")

    def register(
        self,
        service_type: type[T],
        factory: Callable[[Any], T] | Callable[[], T],
        lifetime: str = ServiceLifetime.SINGLETON,
    ) -> None:
        """Register a service with its factory function.

        Args:
            service_type: The type/interface of the service
            factory: Factory function to create the service.
                     Can take container as argument for dependency resolution.
            lifetime: Service lifetime (singleton, transient, or scoped)

        Example:
            ```python
            # Simple factory
            container.register(Config, lambda: Config.load())

            # Factory with dependencies
            container.register(
                Analytics,
                lambda c: Analytics(c.resolve(DataStore))
            )
            ```
        """
        if lifetime not in (
            ServiceLifetime.SINGLETON,
            ServiceLifetime.TRANSIENT,
            ServiceLifetime.SCOPED,
        ):
            raise ValueError(
                f"Invalid lifetime: {lifetime}. " f"Must be singleton, transient, or scoped."
            )

        # Check if factory accepts container parameter
        sig = inspect.signature(factory)
        accepts_container = len(sig.parameters) > 0

        self._services[service_type] = {
            "factory": factory,
            "lifetime": lifetime,
            "accepts_container": accepts_container,
        }

        logger.debug(f"Registered {service_type.__name__} with {lifetime} lifetime")

    def register_instance(self, service_type: type[T], instance: T) -> None:
        """Register an existing instance as a singleton.

        Args:
            service_type: The type/interface of the service
            instance: The instance to register

        Example:
            ```python
            config = Config.load()
            container.register_instance(Config, config)
            ```
        """
        self._singletons[service_type] = instance
        logger.debug(f"Registered instance of {service_type.__name__}")

    def resolve(self, service_type: type[T]) -> T:
        """Resolve a service instance.

        Args:
            service_type: The type/interface to resolve

        Returns:
            Instance of the requested service

        Raises:
            KeyError: If service is not registered

        Example:
            ```python
            data_store = container.resolve(DataStore)
            ```
        """
        # Check for existing singleton
        if service_type in self._singletons:
            logger.debug(f"Returning singleton instance of {service_type.__name__}")
            return self._singletons[service_type]

        # Check for scoped instance
        if self._scope_active and service_type in self._scoped_instances:
            logger.debug(f"Returning scoped instance of {service_type.__name__}")
            return self._scoped_instances[service_type]

        # Check if service is registered
        if service_type not in self._services:
            raise KeyError(
                f"Service {service_type.__name__} is not registered. "
                f"Register it with container.register() first."
            )

        service_info = self._services[service_type]
        factory = service_info["factory"]
        lifetime = service_info["lifetime"]
        accepts_container = service_info["accepts_container"]

        # Create instance
        if accepts_container:
            instance = factory(self)
        else:
            instance = factory()

        logger.debug(f"Created new instance of {service_type.__name__}")

        # Store based on lifetime
        if lifetime == ServiceLifetime.SINGLETON:
            self._singletons[service_type] = instance
        elif lifetime == ServiceLifetime.SCOPED and self._scope_active:
            self._scoped_instances[service_type] = instance

        return instance

    def create_scope(self) -> "ServiceScope":
        """Create a new service scope for scoped services.

        Returns:
            ServiceScope context manager

        Example:
            ```python
            with container.create_scope() as scope:
                # Scoped services are shared within this block
                service1 = scope.resolve(MyService)
                service2 = scope.resolve(MyService)
                # service1 is service2 (same instance)
            # Scope ended, scoped instances are discarded
            ```
        """
        return ServiceScope(self)

    def is_registered(self, service_type: type) -> bool:
        """Check if a service type is registered.

        Args:
            service_type: The type to check

        Returns:
            True if registered, False otherwise
        """
        return service_type in self._services or service_type in self._singletons

    def clear_singletons(self) -> None:
        """Clear all singleton instances.

        Useful for testing or reinitialization.
        """
        count = len(self._singletons)
        self._singletons.clear()
        logger.debug(f"Cleared {count} singleton instances")

    def get_registered_services(self) -> list[type]:
        """Get list of all registered service types.

        Returns:
            List of registered service types
        """
        return list(self._services.keys()) + list(self._singletons.keys())


class ServiceScope:
    """Service scope context manager for scoped services.

    Scoped services maintain a single instance within the scope,
    but create new instances for each scope.
    """

    def __init__(self, container: Container) -> None:
        """Initialize the scope.

        Args:
            container: The parent container
        """
        self._container = container
        logger.debug("Service scope created")

    def __enter__(self) -> "ServiceScope":
        """Enter the scope."""
        self._container._scope_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the scope and clear scoped instances."""
        count = len(self._container._scoped_instances)
        self._container._scoped_instances.clear()
        self._container._scope_active = False
        logger.debug(f"Service scope closed, cleared {count} scoped instances")

    def resolve(self, service_type: type[T]) -> T:
        """Resolve a service within this scope.

        Args:
            service_type: The type/interface to resolve

        Returns:
            Instance of the requested service
        """
        return self._container.resolve(service_type)


# Global container instance
_global_container: Container | None = None


def get_container() -> Container:
    """Get the global container instance.

    Returns:
        The global container

    Example:
        ```python
        from core.dependency_injection import get_container

        container = get_container()
        service = container.resolve(MyService)
        ```
    """
    global _global_container
    if _global_container is None:
        _global_container = Container()
        logger.info("Global dependency injection container created")
    return _global_container


def reset_container() -> None:
    """Reset the global container.

    Useful for testing.
    """
    global _global_container
    _global_container = None
    logger.debug("Global container reset")


__all__ = [
    "Container",
    "ServiceLifetime",
    "ServiceScope",
    "get_container",
    "reset_container",
]
