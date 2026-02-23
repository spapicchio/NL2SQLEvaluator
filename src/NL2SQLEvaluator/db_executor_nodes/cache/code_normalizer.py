"""
This module contains the normalizer function used in the CacheProtocol for different code strategies.
The normalizer function is responsible for transforming the input code (e.g., SQL queries) into a standardized format before hashing and caching.
"""

import re
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


_CYPHER_KEYWORDS = {
    "MATCH", "WHERE", "RETURN", "WITH", "ORDER BY", "LIMIT", "SKIP",
    "OPTIONAL MATCH", "CREATE", "SET", "DELETE", "MERGE", "UNWIND",
    "CALL", "YIELD", "AS", "AND", "OR", "NOT", "IN", "IS", "NULL",
    "TRUE", "FALSE", "DISTINCT", "UNION", "ALL",
}

# Sort by length descending so multi-word keywords like "OPTIONAL MATCH" are matched first
_CYPHER_KEYWORDS_PATTERN = re.compile(
    r'\b(' + '|'.join(
        re.escape(kw) for kw in sorted(_CYPHER_KEYWORDS, key=len, reverse=True)
    ) + r')\b',
    re.IGNORECASE
)


def normalize_cypher(query: str, dialect: str | None = None, *args, **kwargs) -> str:
    """Cypher specific normalization: whitespace + case-fold keywords, preserving string literals."""
    query = " ".join(query.strip().split())

    # Split on quoted segments to preserve string literals
    parts = re.split(r"""('[^']*'|"[^"]*")""", query)
    normalized_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Quoted segment — preserve as-is
            normalized_parts.append(part)
        else:
            # Non-quoted segment — fold keywords to uppercase then lowercase
            normalized_parts.append(
                _CYPHER_KEYWORDS_PATTERN.sub(lambda m: m.group(0).lower(), part.lower())
            )

    return " ".join("".join(normalized_parts).split())


_SPARQL_KEYWORDS = {
    "SELECT", "WHERE", "FILTER", "OPTIONAL", "PREFIX", "ASK", "CONSTRUCT",
    "DESCRIBE", "ORDER BY", "LIMIT", "OFFSET", "GROUP BY", "HAVING",
    "BIND", "VALUES", "UNION", "DISTINCT", "REDUCED", "AS",
}

_SPARQL_KEYWORDS_PATTERN = re.compile(
    r'\b(' + '|'.join(
        re.escape(kw) for kw in sorted(_SPARQL_KEYWORDS, key=len, reverse=True)
    ) + r')\b',
    re.IGNORECASE
)


def normalize_sparql(query: str, dialect: str | None = None, *args, **kwargs) -> str:
    """SPARQL specific normalization: whitespace + case-fold keywords, preserving URIs and literals."""
    query = " ".join(query.strip().split())

    # Split on URIs (<...>) and quoted segments to preserve them
    parts = re.split(r"""(<[^>]*>|'[^']*'|"[^"]*")""", query)
    normalized_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # URI or quoted segment — preserve as-is
            normalized_parts.append(part)
        else:
            # Non-quoted/URI segment — fold keywords
            normalized_parts.append(
                _SPARQL_KEYWORDS_PATTERN.sub(lambda m: m.group(0).lower(), part.lower())
            )

    return " ".join("".join(normalized_parts).split())


# Registry for language-specific normalizers
QUERY_NORMALIZERS: dict[str, Callable[[str, str | None], str]] = {
    "sql": normalize_sql,
    "sqlite": normalize_sql,
    "postgres": normalize_sql,
    "cypher": normalize_cypher,
    "neo4j": normalize_cypher,
    "sparql": normalize_sparql,
}
