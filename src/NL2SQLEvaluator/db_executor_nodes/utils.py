import re


def extract_last_match(text: str, pattern: str) -> str:
    """Extracts the last match of a regex pattern or returns the original string."""
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else text


def utils_extract_sql_or_same(generation: str) -> str:
    """Parses SQL from LLM responses, looking for <answer> or code blocks."""
    content = extract_last_match(generation, r"<answer>(.*?)</answer>")
    content = extract_last_match(content, r"```sql\s*(.*?)\s*```")
    content = extract_last_match(content, r"```\s*(.*?)\s*```")
    return content.strip().strip("`").strip()


def utils_extract_cypher_or_same(generation: str) -> str:
    """Parses Cypher from LLM responses, looking for <answer> or code blocks."""
    content = extract_last_match(generation, r"<answer>(.*?)</answer>")
    content = extract_last_match(content, r"```cypher\s*(.*?)\s*```")
    content = extract_last_match(content, r"```\s*(.*?)\s*```")
    return content.strip().strip("`").strip()


def utils_extract_sparql_or_same(generation: str) -> str:
    """Parses SPARQL from LLM responses, looking for <answer> or code blocks."""
    content = extract_last_match(generation, r"<answer>(.*?)</answer>")
    content = extract_last_match(content, r"```sparql\s*(.*?)\s*```")
    content = extract_last_match(content, r"```\s*(.*?)\s*```")
    return content.strip().strip("`").strip()
