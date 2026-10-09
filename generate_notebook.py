import nbformat as nbf
import subprocess
import os
import sys
from pathlib import Path

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors='replace')
    return result.stdout + result.stderr

def create_master_notebook():
    nb = nbf.v4.new_notebook()
    
    # Add an introductory markdown cell
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# Ps3_NeuroGeo Pipeline\n\n"
        "This notebook executes the full NeuroGeo-Field pipeline (Phases 0-5), "
        "integrating the candidate generation, validation, reranking, and navigation components."
    ))

    phases = [
        ("Phase 0: Baseline Reproduction", "python scripts/run_phase0.py"),
        ("Phase 1: Visit Sanitizer & Miner", "python scripts/run_phase1.py"),
        ("Phase 2: Address Parser & POI", "python scripts/run_phase2.py"),
        ("Phase 3: Spatial Graph & Validation", "python scripts/run_phase3.py"),
        ("Phase 4: Calibration & Reranking", "python scripts/run_phase4.py"),
        ("Phase 5: Navigator & Evidence Cards", "python scripts/run_phase5.py"),
    ]

    for title, cmd in phases:
        nb.cells.append(nbf.v4.new_markdown_cell(f"### {title}"))
        code = f"!{cmd}"
        output_text = run_cmd(cmd)
        
        cell = nbf.v4.new_code_cell(code)
        
        # Add output to the cell
        output = nbf.v4.new_output(
            output_type="stream",
            name="stdout",
            text=output_text
        )
        cell.outputs.append(output)
        cell.execution_count = 1  # Dummy execution count
        
        nb.cells.append(cell)
        
    out_path = "master.ipynb"
    with open(out_path, "w", encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Successfully generated {out_path}")

if __name__ == "__main__":
    create_master_notebook()
