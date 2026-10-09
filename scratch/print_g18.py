import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from scratch.inspect_edge_cases import g18

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print(f"Total in g18: {len(g18)}")
for i, (_, r) in enumerate(g18.iterrows(), 1):
    print(f"{i:2d}. [{r['address_id']}] ({r['town_id']}): {r['address_text']} --> Landmark: {r['landmarks']}")
