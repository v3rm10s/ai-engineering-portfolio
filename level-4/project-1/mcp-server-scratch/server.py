import platform
import math
from typing import Literal
from mcp.server import MCPServer

# Initialize the MCP Server instance
mcp = MCPServer(name="SystemUtilitiesServer")

@mcp.tool()
def get_system_info() -> dict[str, str]:
    """Retrieve host platform metadata and architecture details."""
    return {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }

@mcp.tool()
def compute_statistics(
    numbers: list[float],
    operation: Literal["mean", "median", "std_dev", "sum"]
) -> dict[str, float | str]:
    """Compute mathematical aggregates on a list of floating-point numbers.

    :param numbers: List of numerical values to aggregate.
    :param operation: The aggregation operation to perform (mean, median, std_dev, sum).
    """
    if not numbers:
        return {"error": "Input list cannot be empty"}

    n = len(numbers)
    if operation == "sum":
        result = sum(numbers)
    elif operation == "mean":
        result = sum(numbers) / n
    elif operation == "median":
        sorted_nums = sorted(numbers)
        mid = n // 2
        result = sorted_nums[mid] if n % 2 != 0 else (sorted_nums[mid - 1] + sorted_nums[mid]) / 2.0
    elif operation == "std_dev":
        mean = sum(numbers) / n
        variance = sum((x - mean) ** 2 for x in numbers) / n
        result = math.sqrt(variance)
    else:
        return {"error": f"Unsupported operation: {operation}"}

    return {"operation": operation, "count": float(n), "result": round(result, 4)}

@mcp.tool()
def format_text(
    text: str,
    style: Literal["uppercase", "lowercase", "titlecase", "reverse"]
) -> str:
    """Transform an input string according to the requested casing or ordering style.

    :param text: The source string to format.
    :param style: The transformation target format.
    """
    if style == "uppercase":
        return text.upper()
    elif style == "lowercase":
        return text.lower()
    elif style == "titlecase":
        return text.title()
    elif style == "reverse":
        return text[::-1]
    return text

if __name__ == "__main__":
    mcp.run()