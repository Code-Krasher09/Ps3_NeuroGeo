"""Validate notebook code execution.

Loads notebooks/01_master_pipeline.ipynb and notebooks/02_interactive_dashboard.ipynb,
executes all code cells sequentially in an isolated namespace, and reports execution status.
"""

import json
from pathlib import Path
import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def validate_notebook(nb_path: Path):
    print("=" * 70)
    print(f"Validating notebook: {nb_path.name}")
    print("=" * 70)
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    namespace = {
        "__file__": str(nb_path.resolve()),
        "__name__": "__main__",
    }
    
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    print(f"Total cells: {len(nb.get('cells', []))}, Code cells: {len(code_cells)}")
    
    for idx, cell in enumerate(code_cells, start=1):
        source = "".join(cell.get("source", []))
        try:
            exec(source, namespace)
            plt.close("all")
            print(f"  [Cell {idx}/{len(code_cells)}] PASS")
        except Exception as e:
            print(f"  [Cell {idx}/{len(code_cells)}] FAILED: {type(e).__name__}: {e}")
            print("Cell source:")
            print("-" * 50)
            print(source[:500])
            print("-" * 50)
            raise e
            
    print(f"Notebook {nb_path.name} validated successfully with ZERO errors!\n")

def main():
    nb1 = Path("notebooks/01_master_pipeline.ipynb")
    nb2 = Path("notebooks/02_interactive_dashboard.ipynb")
    
    assert nb1.exists(), f"Missing {nb1}"
    assert nb2.exists(), f"Missing {nb2}"
    
    validate_notebook(nb1)
    validate_notebook(nb2)
    print("ALL NOTEBOOKS VALIDATED AND EXECUTABLE!")

if __name__ == "__main__":
    main()
