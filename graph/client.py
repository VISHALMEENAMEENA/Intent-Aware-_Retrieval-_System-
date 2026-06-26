"""
graph/client.py
===============
Neo4j driver singleton and session context-manager helper.
"""

from __future__ import annotations

import contextlib
from typing import Any, Generator

try:
    from neo4j import GraphDatabase, Driver, Session
except ImportError:
    GraphDatabase = None
    Driver = Any
    Session = Any

from graph.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, validate_config

_driver: Driver | None = None


def get_driver() -> Driver:
    """Return the singleton Neo4j driver, creating it on first call."""
    global _driver
    if _driver is None:
        if GraphDatabase is None:
            raise RuntimeError("neo4j package is not installed")
        validate_config()
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_pool_size=10,
        )
    return _driver


def close_driver() -> None:
    """Cleanly close the singleton driver (call at application shutdown)."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


@contextlib.contextmanager
def get_session(database: str = "neo4j") -> Generator[Session, None, None]:
    """Context-manager that yields an open Neo4j session."""
    driver = get_driver()
    session = driver.session(database=database)
    try:
        yield session
    finally:
        session.close()


def verify_connection() -> bool:
    """Ping Neo4j and return True on success."""
    try:
        with get_session() as session:
            session.run("RETURN 1").consume()
        return True
    except Exception:
        return False
