"""
Test script for tools.py functions
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tools import get_declarations, read_table, query, validate_dimensions

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
TEMPLATE_FILE = PROJECT_DIR / "prompt" / "template.mzn"

def test_get_declarations():
    print("=" * 60)
    print("TEST 1: get_declarations()")
    print("=" * 60)

    mzn_code = TEMPLATE_FILE.read_text()
    result = get_declarations(mzn_code)

    print("\nParameters needing data (.dzn):")
    for p in result['needs_data']:
        print(f"  - {p}")

    print("\nParameters with values (constants):")
    for p in result['has_value']:
        print(f"  - {p}")

    return result

def test_read_table():
    print("\n" + "=" * 60)
    print("TEST 2: read_table()")
    print("=" * 60)

    # Test demand.csv
    demand_file = str(DATA_DIR / "demand.csv")
    result = read_table(demand_file)

    print(f"\ndemand.csv:")
    print(f"  Shape: {result['shape']}")
    print(f"  Columns: {result['columns']}")
    print(f"  Sample (first 2 rows):")
    for row in result['sample'][:2]:
        print(f"    {row}")

    # Test molds.csv
    molds_file = str(DATA_DIR / "molds.csv")
    result2 = read_table(molds_file)

    print(f"\nmolds.csv:")
    print(f"  Shape: {result2['shape']}")
    print(f"  Columns: {result2['columns']}")

    return result, result2

def test_query():
    print("\n" + "=" * 60)
    print("TEST 3: query()")
    print("=" * 60)

    demand_file = str(DATA_DIR / "demand.csv")
    molds_file = str(DATA_DIR / "molds.csv")

    # Test 1: Unique colors
    colors = query(demand_file, "unique", column="COLOR")
    print(f"\nUnique colors: {colors}")

    # Test 2: Count unique sizes
    size_count = query(demand_file, "unique_count", column="SIZE")
    print(f"Unique size count: {size_count}")

    # Test 3: Sum demand by SIZE+COLOR
    demands = query(demand_file, "sum", column="QTY", group_by=["SIZE", "COLOR"])
    print(f"\nDemand sums by (SIZE, COLOR): {demands[:5]}... (showing first 5)")

    # Test 4: Mold inventory list
    inventory = query(molds_file, "list", column="INVENTORY")
    print(f"\nMold inventory: {inventory}")

    # Test 5: Mold cycle times
    ct = query(molds_file, "list", column="CT")
    print(f"Cycle times: {ct}")

    return colors, demands, inventory

def test_validate_dimensions():
    print("\n" + "=" * 60)
    print("TEST 4: validate_dimensions()")
    print("=" * 60)

    mzn_code = TEMPLATE_FILE.read_text()

    # Test with CORRECT dzn (matching lengths)
    dzn_correct = """
num_sides = 4;
num_shifts = 12;
colors = ["WHITE", "LTIRONORE"];
moldsize_inventory = [2, 3, 4];
moldsize_ct = [12.0, 6.99, 6.99];
prod_demand = [100, 200, 150];
prod_to_moldsize = [1, 2, 3];
prod_to_color = [1, 1, 2];
prod_deadline = [12, 12, 12];
"""

    result1 = validate_dimensions(mzn_code, dzn_correct)
    print(f"\nCorrect .dzn validation:")
    print(f"  Valid: {result1['valid']}")
    print(f"  Array lengths: {result1['array_lengths']}")

    # Test with WRONG dzn (mismatched lengths)
    dzn_wrong = """
num_sides = 4;
num_shifts = 12;
colors = ["WHITE", "LTIRONORE"];
moldsize_inventory = [2, 3, 4];
moldsize_ct = [12.0, 6.99, 6.99];
prod_demand = [100, 200, 150];
prod_to_moldsize = [1, 2, 3, 4, 5];
prod_to_color = [1, 1];
prod_deadline = [12, 12, 12];
"""

    result2 = validate_dimensions(mzn_code, dzn_wrong)
    print(f"\nWrong .dzn validation:")
    print(f"  Valid: {result2['valid']}")
    print(f"  Errors: {result2['errors']}")
    print(f"  Array lengths: {result2['array_lengths']}")

    return result1, result2

if __name__ == "__main__":
    print("\nTesting tools.py functions with real data\n")

    test_get_declarations()
    test_read_table()
    test_query()
    test_validate_dimensions()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
