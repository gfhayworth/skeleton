"""Test ensuring notebooks/dialogue_demo.ipynb executes cleanly without error."""

import json
import os
import sys
from pathlib import Path
import pytest


@pytest.mark.asyncio
async def test_notebook_execution():
    project_root = Path(__file__).resolve().parent.parent
    notebook_path = project_root / "notebooks" / "dialogue_demo.ipynb"
    assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Global environment for notebook cell execution
    nb_globals = {
        "__file__": str(notebook_path),
        "sys": sys,
        "os": os,
    }

    for idx, cell in enumerate(nb_data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        # If cell contains top-level await, wrap in async function
        if "await " in source:
            indented = "\n".join("    " + line for line in source.splitlines())
            wrapped_code = f"async def _cell_runner():\n{indented}\n"
            exec(wrapped_code, nb_globals)
            await nb_globals["_cell_runner"]()
        else:
            exec(source, nb_globals)
