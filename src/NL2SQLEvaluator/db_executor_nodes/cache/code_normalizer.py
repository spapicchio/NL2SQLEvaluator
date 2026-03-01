"""
This module contains the normalizer function used in the CacheProtocol for different code strategies.
The normalizer function is responsible for transforming the input code (e.g., SQL queries) into a standardized format before hashing and caching.
"""

from typing import Callable


def base_normalize_whitespace_and_case(query: str, dialect: str | None = None, *args, **kwargs) -> str:
    """Standard normalization: removes extra spaces and lowers case."""
    return " ".join(query.strip().split()).lower()


def normalize_sql(query: str, dialect: str | None = None, *args, **kwargs) -> str:
    """SQL specific normalization using sqlglot."""
    try:
        import sqlglot
        # identity=True ensures it returns the query in a canonical format
        return sqlglot.transpile(query, dialect=dialect, identity=True)[0].lower()
    except Exception:
        return base_normalize_whitespace_and_case(query)


def normalize_cypher(query: str, dialect: str | None = None, *args, **kwargs) -> str:
    """Cypher specific normalization (placeholder)."""
    # TODO Implement Cypher-specific normalization logic here
    return base_normalize_whitespace_and_case(query)


def normalize_sparql(query: str, dialect: str | None = None, *args, **kwargs) -> str:
    """SPARQL specific normalization (placeholder)."""
    # TODO Implement SPARQL-specific normalization logic here
    return base_normalize_whitespace_and_case(query)


# Registry for language-specific normalizers
QUERY_NORMALIZERS: dict[str, Callable[[str, str | None], str]] = {
    "sql": normalize_sql,
    "sqlite": normalize_sql,
    "postgres": normalize_sql,
    "cypher": normalize_cypher,  # Placeholder for Cypher logic
    "sparql": normalize_sparql,  # Placeholder for SPARQL logic
}
