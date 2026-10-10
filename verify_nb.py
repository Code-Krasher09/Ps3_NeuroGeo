import json

with open("submission.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        try:
            compile(source, f"cell_{i}", "exec")
            print(f"Cell {i} compiled successfully.")
        except Exception as e:
            print(f"Error compiling cell {i}: {e}")
