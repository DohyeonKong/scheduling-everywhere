"""
Pure Python tools for MiniZinc data generation.
These tools are called by LLM via function calling to build .dzn files.
"""

import re
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel


# ============================================================
# Structured Output Models
# ============================================================

class QuerySpec(BaseModel):
    """Specification for generating a .dzn parameter value"""
    param_name: str                           # "prod_demand"
    value_type: str                           # "constant" | "query" | "expression"

    # If value_type == "constant":
    constant_value: Optional[Union[int, float, str, List[int], List[float], List[str]]] = None

    # If value_type == "expression":
    expr: Optional[str] = None                # "[num_shifts] * len(prod_demand)"

    # If value_type == "query":
    filename: Optional[str] = None            # "demand.csv"
    operation: Optional[str] = None           # "sum", "unique", "list", "count", "first", "index_map"
    column: Optional[str] = None              # "QTY"
    group_by: Optional[List[str]] = None      # ["SIZE", "COLOR"]
    filter_expr: Optional[str] = None         # "COLOR == 'WHITE'"
    reference: Optional[List[str]] = None     # for index_map: explicit reference list
    reference_file: Optional[str] = None      # for index_map: file to build reference from
    reference_column: Optional[str] = None    # for index_map: column in reference_file


class ModelingResponse(BaseModel):
    """Structured output from LLM - Step 1: Generate .mzn model"""
    explanation: str                          # Chat response
    math_model: Optional[str] = None          # LaTeX formulation
    mzn_code: Optional[str] = None            # .mzn model code


class DataGenResponse(BaseModel):
    """Structured output from LLM - Step 2: Generate .dzn queries for declared params"""
    dzn_queries: List[QuerySpec]              # Recipe for .dzn (only for declared params)


def get_declarations(mzn_code: str) -> Dict[str, Any]:
    """
    Parse .mzn code to extract parameter declarations that need data.

    Returns dict with:
    - 'needs_data': list of param names without values (need .dzn)
    - 'has_value': list of param names with values (constants)
    """
    needs_data = []
    has_value = []

    # Pattern for parameter declarations
    # Matches: int: name; or array[...] of type: name;
    # But NOT: int: name = value; (has assignment)

    lines = mzn_code.split('\n')

    for line in lines:
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith('%') or line.startswith('//'):
            continue

        # Skip constraint, solve, include lines
        if any(line.startswith(kw) for kw in ['constraint', 'solve', 'include', 'var ', 'output']):
            continue

        # Pattern: type declaration with name
        # int: name; OR int: name = value;
        # array[int] of int: name; OR array[int] of int: name = [...];

        # Check if it's a parameter declaration (not var)
        param_pattern = r'^(int|float|bool|string|array\[.*?\]\s*of\s*(?:int|float|bool|string)):\s*(\w+)'
        match = re.match(param_pattern, line)

        if match:
            param_name = match.group(2)

            # Check if it has an assignment (= something)
            if '=' in line:
                has_value.append(param_name)
            else:
                needs_data.append(param_name)

    return {
        'needs_data': needs_data,
        'has_value': has_value
    }


def read_table(filename: str) -> Dict[str, Any]:
    """
    Read CSV file and return schema + sample rows.

    Returns dict with:
    - 'columns': list of column names
    - 'dtypes': dict of column -> dtype
    - 'shape': (rows, cols)
    - 'sample': first 5 rows as list of dicts
    """
    df = pd.read_csv(filename)

    return {
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'shape': list(df.shape),
        'sample': df.head(5).to_dict(orient='records')
    }


def query(filename: str, operation: str, column: Optional[str] = None,
          group_by: Optional[List[str]] = None,
          filter_expr: Optional[str] = None,
          reference: Optional[List[str]] = None,
          reference_file: Optional[str] = None,
          reference_column: Optional[str] = None) -> Any:
    """
    Execute query on CSV file.

    Args:
        filename: path to CSV file
        operation: one of 'unique', 'count', 'sum', 'list', 'unique_count', 'index_map'
        column: column to operate on
        group_by: columns to group by (for aggregations)
        filter_expr: pandas query expression (e.g., "SIZE == '6'")
        reference: for index_map, the reference list to map values to indices

    Returns:
        Query result (list, int, or dict depending on operation)
    """
    df = pd.read_csv(filename)

    # Apply filter if provided
    if filter_expr:
        df = df.query(filter_expr)

    # Apply group_by if provided
    if group_by:
        grouped = df.groupby(group_by, sort=False)

        if operation == 'sum':
            result = grouped[column].sum()
            return result.tolist()
        elif operation == 'count':
            result = grouped.size()
            return result.tolist()
        elif operation == 'list':
            result = grouped[column].apply(list)
            return result.tolist()
        elif operation == 'first':
            result = grouped[column].first()
            return result.tolist()
        elif operation == 'index_map':
            # Map each group's column value to 1-based index in reference list
            first_vals = grouped[column].first()

            # Determine reference list
            if reference:
                ref_list = reference
            elif reference_file and reference_column:
                # Build reference from another file
                ref_df = pd.read_csv(reference_file)
                ref_list = ref_df[reference_column].tolist()
            else:
                # Build reference from unique values in same file
                ref_list = df[column].unique().tolist()

            # Map values to indices
            try:
                indices = [ref_list.index(v) + 1 for v in first_vals]
            except ValueError as e:
                print(f"[WARN] index_map: value not in reference - {e}")
                # Fallback: use unique values from data
                ref_list = df[column].unique().tolist()
                indices = [ref_list.index(v) + 1 for v in first_vals]
            return indices
    else:
        # No grouping
        if operation == 'unique':
            return df[column].unique().tolist()
        elif operation == 'unique_count':
            return df[column].nunique()
        elif operation == 'count':
            return len(df)
        elif operation == 'sum':
            return df[column].sum()
        elif operation == 'list':
            return df[column].tolist()
        elif operation == 'value_counts':
            return df[column].value_counts().to_dict()
        elif operation == 'index_map':
            # Map column values to 1-based indices
            if reference:
                return [reference.index(v) + 1 for v in df[column]]
            else:
                unique_vals = df[column].unique().tolist()
                return [unique_vals.index(v) + 1 for v in df[column]]

    return None


def get_product_order(filename: str, group_by: List[str]) -> List[Dict[str, str]]:
    """
    Get product order matching groupby order used in queries.
    Returns list of dicts with lowercase keys: mold, size, color (for display).
    """
    df = pd.read_csv(filename)
    grouped = df.groupby(group_by, sort=False)

    products = []
    for keys, _ in grouped:
        if isinstance(keys, tuple):
            raw_product = dict(zip(group_by, keys))
        else:
            raw_product = {group_by[0]: keys}

        # Build product dict with lowercase keys for display
        product = {}

        # Add SIZE -> size
        if 'SIZE' in raw_product:
            product['size'] = str(raw_product['SIZE'])

        # Add COLOR -> color (simplified)
        if 'COLOR' in raw_product:
            color = raw_product['COLOR']
            product['color'] = color.split('_')[-1] if '_' in color else color

        # Add MOLD if available and not in group_by
        if 'MOLD' in df.columns and 'MOLD' not in group_by:
            # Get first MOLD value for this group
            mask = True
            for col, val in raw_product.items():
                mask = mask & (df[col] == val)
            product['mold'] = df[mask]['MOLD'].iloc[0]
        elif 'MOLD' in raw_product:
            product['mold'] = raw_product['MOLD']

        products.append(product)

    return products


def validate_dimensions(mzn_code: str, dzn_code: str) -> Dict[str, Any]:
    """
    Validate that array dimensions in .dzn match declarations in .mzn.

    Returns dict with:
    - 'valid': True if all dimensions match
    - 'errors': list of error messages
    - 'array_lengths': dict of array_name -> length
    """
    errors = []
    array_lengths = {}

    # Extract array lengths from .dzn
    # Pattern: name = [val1, val2, ...];
    dzn_array_pattern = r'(\w+)\s*=\s*\[(.*?)\];'

    for match in re.finditer(dzn_array_pattern, dzn_code, re.DOTALL):
        name = match.group(1)
        values = match.group(2)

        # Count elements (split by comma, handling nested brackets)
        # Simple approach: count commas + 1 (works for flat arrays)
        if values.strip():
            # Handle string arrays with commas inside quotes
            if '"' in values:
                # Count quoted strings
                count = len(re.findall(r'"[^"]*"', values))
            else:
                count = len([v.strip() for v in values.split(',') if v.strip()])
        else:
            count = 0

        array_lengths[name] = count

    # Find arrays that should have same length (indexed by PRODUCTS, MOLDSIZES, etc.)
    # Look for patterns like: array[PRODUCTS] of int: prod_demand;
    # Only check NAMED index sets (skip 'int' which is generic)
    mzn_array_pattern = r'array\[(\w+)\]\s*of\s*\w+:\s*(\w+)'

    index_sets = {}  # index_name -> list of array names

    for match in re.finditer(mzn_array_pattern, mzn_code):
        index_name = match.group(1)
        array_name = match.group(2)

        # Skip generic 'int' index - only check named sets like PRODUCTS, MOLDSIZES
        if index_name == 'int':
            continue

        if index_name not in index_sets:
            index_sets[index_name] = []
        index_sets[index_name].append(array_name)

    # Check that arrays with same index have same length
    for index_name, arrays in index_sets.items():
        lengths = {}
        for arr in arrays:
            if arr in array_lengths:
                lengths[arr] = array_lengths[arr]

        if len(lengths) > 1 and len(set(lengths.values())) > 1:
            errors.append(
                f"Index '{index_name}' arrays have mismatched lengths: {lengths}"
            )

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'array_lengths': array_lengths,
        'index_sets': index_sets
    }


# ============================================================
# Model Builder State (for agent loop)
# ============================================================
class ModelBuilder:
    """Stateful builder for .mzn and .dzn during agent loop"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.mzn_code: Optional[str] = None
        self.math_model: Optional[str] = None
        self.dzn_params: Dict[str, Any] = {}
        self.declarations: Optional[Dict] = None
        self.product_group_by: Optional[List[str]] = None

    def _fix_mzn_syntax(self, code: str) -> str:
        """Fix common MiniZinc syntax issues from LLM output"""
        # Fix double-escaped backslashes in logical operators
        # /\\ -> /\  (AND operator)
        # \\/ -> \/  (OR operator)
        code = code.replace('/\\\\', '/\\')
        code = code.replace('\\\\/', '\\/')
        # Also fix triple or more backslashes that might occur
        while '/\\\\' in code:
            code = code.replace('/\\\\', '/\\')
        while '\\\\/' in code:
            code = code.replace('\\\\/', '\\/')
        return code

    def set_mzn(self, code: str, math_model: Optional[str] = None) -> Dict:
        """Set the .mzn code and return declarations"""
        # Fix common escaping issues from LLM output
        code = self._fix_mzn_syntax(code)
        self.mzn_code = code
        if math_model:
            self.math_model = math_model
        self.declarations = get_declarations(code)

        # Debug: print first 20 lines of code to verify parsing
        print(f"[DEBUG] set_mzn received code ({len(code)} chars)")
        for i, line in enumerate(code.split('\n')[:20]):
            print(f"[DEBUG]   {i+1}: {line[:80]}")
        print(f"[DEBUG] Parsed needs_data: {self.declarations['needs_data']}")
        print(f"[DEBUG] Parsed has_value: {self.declarations['has_value']}")

        return {
            "status": "ok",
            "needs_data": self.declarations["needs_data"],
            "has_value": self.declarations["has_value"],
            "message": f"Model saved. {len(self.declarations['needs_data'])} parameters need data values: {self.declarations['needs_data']}"
        }

    def set_param(self, param_name: str, value: Any) -> Dict:
        """Set a .dzn parameter value"""
        if self.declarations and param_name not in self.declarations["needs_data"]:
            return {
                "status": "warning",
                "message": f"Parameter '{param_name}' is not declared in .mzn as needing data. Declared params: {self.declarations['needs_data']}"
            }
        self.dzn_params[param_name] = value
        remaining = self.get_missing_params()
        return {
            "status": "ok",
            "message": f"Set {param_name} (length={len(value) if isinstance(value, list) else 1}). Remaining: {remaining if remaining else 'None - ready to finalize!'}"
        }

    def get_missing_params(self) -> List[str]:
        """Get list of params still needing values"""
        if not self.declarations:
            return []
        return [p for p in self.declarations["needs_data"] if p not in self.dzn_params]

    def build_dzn(self) -> str:
        """Build .dzn content from params"""
        lines = ["% Auto-generated data file"]
        for name, value in self.dzn_params.items():
            lines.append(f'{name} = {self._format_dzn_value(value)};')
        return "\n".join(lines)

    def _format_dzn_value(self, value: Any) -> str:
        """Format a Python value for MiniZinc .dzn syntax"""
        if isinstance(value, bool):
            # MiniZinc uses lowercase true/false
            return "true" if value else "false"
        elif isinstance(value, str) and not value.startswith("["):
            return f'"{value}"'
        elif isinstance(value, list):
            if not value:
                return "[]"
            formatted_items = [self._format_dzn_value(v) for v in value]
            return f'[{", ".join(formatted_items)}]'
        else:
            return str(value)

    def finalize(self, math_model: Optional[str] = None, explanation: Optional[str] = None) -> Dict:
        """Finalize and validate the model"""
        if not self.mzn_code:
            return {
                "status": "error",
                "message": "No .mzn model set. Call set_mzn first."
            }

        missing = self.get_missing_params()
        if missing:
            return {
                "status": "incomplete",
                "missing_params": missing,
                "message": f"Still need values for: {missing}"
            }

        # Store math_model and explanation
        if math_model:
            self.math_model = math_model
        if explanation:
            self.explanation = explanation

        dzn_code = self.build_dzn()
        validation = validate_dimensions(self.mzn_code, dzn_code)

        return {
            "status": "complete" if validation["valid"] else "warning",
            "mzn_code": self.mzn_code,
            "dzn_code": dzn_code,
            "math_model": self.math_model,
            "explanation": getattr(self, 'explanation', None),
            "validation": validation,
            "message": "Model complete and validated!" if validation["valid"] else f"Warnings: {validation['errors']}"
        }


# ============================================================
# OpenAI Function Calling schema definitions
# ============================================================
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_table",
            "description": "Read CSV file and return schema with sample rows. Use this first to understand the data structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "enum": ["demand.csv", "molds.csv"],
                        "description": "Name of the CSV file"
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": "Execute query on CSV data to get values for .dzn parameters. Operations: unique (distinct values), list (all values), sum (aggregate), count, first, index_map (map values to 1-based indices).",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "enum": ["demand.csv", "molds.csv"],
                        "description": "Name of the CSV file"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["unique", "count", "sum", "list", "unique_count", "first", "index_map"],
                        "description": "The operation to perform"
                    },
                    "column": {
                        "type": "string",
                        "description": "Column to operate on"
                    },
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to group by (e.g., ['SIZE', 'COLOR'] to define products)"
                    },
                    "filter_expr": {
                        "type": "string",
                        "description": "Pandas query expression for filtering (e.g., \"COLOR == 'WHITE'\")"
                    },
                    "reference_file": {
                        "type": "string",
                        "description": "For index_map: reference file to get index order from"
                    },
                    "reference_column": {
                        "type": "string",
                        "description": "For index_map: column in reference file"
                    }
                },
                "required": ["filename", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_mzn",
            "description": "Set the MiniZinc model code (.mzn). Returns list of parameters that need data values. You must call this before setting parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The complete MiniZinc model code"
                    },
                    "math_model": {
                        "type": "string",
                        "description": "LaTeX mathematical formulation (optional)"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_param",
            "description": "Set a .dzn parameter value. Call this for each parameter returned by set_mzn as needing data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_name": {
                        "type": "string",
                        "description": "Parameter name (must match a declaration in .mzn)"
                    },
                    "value": {
                        "description": "The value - can be int, float, string, or array"
                    }
                },
                "required": ["param_name", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": "Finalize the model after setting all parameters. Validates dimensions and returns complete model. Call this when all parameters are set. MUST include math_model (LaTeX) and explanation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "math_model": {
                        "type": "string",
                        "description": "LaTeX mathematical formulation with Sets, Parameters, Decision Variables, Constraints, and Objective"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Explanation of the modeling approach: problem summary, key decisions, data mapping"
                    }
                },
                "required": ["math_model", "explanation"]
            }
        }
    }
]


def execute_tool(tool_name: str, arguments: dict, builder: ModelBuilder) -> Any:
    """
    Execute a tool by name with given arguments.
    Used by the function calling loop.
    """
    from pathlib import Path

    if tool_name == "read_table":
        filepath = str(Path(builder.data_dir) / arguments["filename"])
        return read_table(filepath)

    elif tool_name == "query":
        filepath = str(Path(builder.data_dir) / arguments["filename"])
        ref_filepath = None
        if arguments.get("reference_file"):
            ref_filepath = str(Path(builder.data_dir) / arguments["reference_file"])

        result = query(
            filename=filepath,
            operation=arguments["operation"],
            column=arguments.get("column"),
            group_by=arguments.get("group_by"),
            filter_expr=arguments.get("filter_expr"),
            reference_file=ref_filepath,
            reference_column=arguments.get("reference_column")
        )

        # Track group_by for product display order
        if arguments.get("group_by") and arguments["filename"] == "demand.csv":
            builder.product_group_by = arguments["group_by"]

        return result

    elif tool_name == "set_mzn":
        return builder.set_mzn(
            code=arguments["code"],
            math_model=arguments.get("math_model")
        )

    elif tool_name == "set_param":
        return builder.set_param(
            param_name=arguments["param_name"],
            value=arguments["value"]
        )

    elif tool_name == "finalize":
        return builder.finalize(
            math_model=arguments.get("math_model"),
            explanation=arguments.get("explanation")
        )

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def assemble_dzn(queries: List[QuerySpec], data_dir: str) -> str:
    """
    Execute query specs and assemble .dzn file content.

    Args:
        queries: List of QuerySpec from LLM
        data_dir: Path to data directory containing CSV files

    Returns:
        Complete .dzn file content as string
    """
    from pathlib import Path

    lines = ["% Auto-generated data file"]
    results = {}  # Store results for cross-reference (e.g., prod_deadline needs prod_demand length)

    for q in queries:
        if q.value_type == "constant":
            # Direct constant value
            value = q.constant_value
            if value is None:
                # Skip None values
                results[q.param_name] = None
                continue
            if isinstance(value, str):
                lines.append(f'{q.param_name} = "{value}";')
                results[q.param_name] = value
            elif isinstance(value, list):
                # Format list appropriately
                if value and isinstance(value[0], str):
                    formatted = ", ".join(f'"{v}"' for v in value)
                else:
                    formatted = ", ".join(str(v) for v in value)
                lines.append(f'{q.param_name} = [{formatted}];')
                results[q.param_name] = value
            else:
                lines.append(f'{q.param_name} = {value};')
                results[q.param_name] = value

        elif q.value_type == "expression":
            # Evaluate Python expression with access to previously computed values
            try:
                result = eval(q.expr, {"__builtins__": {"len": len, "sum": sum, "min": min, "max": max, "int": int, "float": float}}, results)
                results[q.param_name] = result
                print(f"[DEBUG] Expression '{q.expr}' → {result}")

                # Format result
                if isinstance(result, list):
                    if result and isinstance(result[0], str):
                        formatted = ", ".join(f'"{v}"' for v in result)
                    else:
                        formatted = ", ".join(str(v) for v in result)
                    lines.append(f'{q.param_name} = [{formatted}];')
                else:
                    lines.append(f'{q.param_name} = {result};')
            except Exception as e:
                print(f"[ERROR] Expression eval failed: {q.expr} → {e}")
                raise ValueError(f"Failed to evaluate expression for {q.param_name}: {e}")

        elif q.value_type == "query":
            # Execute query
            filepath = str(Path(data_dir) / q.filename)
            ref_filepath = str(Path(data_dir) / q.reference_file) if q.reference_file else None
            result = query(
                filename=filepath,
                operation=q.operation,
                column=q.column,
                group_by=q.group_by,
                filter_expr=q.filter_expr,
                reference=q.reference,
                reference_file=ref_filepath,
                reference_column=q.reference_column
            )

            # Store result for cross-reference
            results[q.param_name] = result

            # Format result
            if isinstance(result, list):
                if result and isinstance(result[0], str):
                    formatted = ", ".join(f'"{v}"' for v in result)
                else:
                    formatted = ", ".join(str(v) for v in result)
                lines.append(f'{q.param_name} = [{formatted}];')
            else:
                lines.append(f'{q.param_name} = {result};')

    return "\n".join(lines)
