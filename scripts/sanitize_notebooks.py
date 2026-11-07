"""
scripts/sanitize_notebooks.py
Validates and sanitizes every Jupyter notebook in the repository:
1. Valid nbformat v4
2. Unique cell IDs on every cell
3. Code cells have "outputs": [] and "execution_count": None
4. Validates via nbformat.validate()
"""
import glob
import os
import uuid
import nbformat

def sanitize_notebook(path: str) -> bool:
    print(f"Processing: {path} ...", end=" ")
    try:
        nb = nbformat.read(path, as_version=4)
    except Exception as e:
        print(f"FAILED to read: {e}")
        return False

    # Ensure format is v4.5 so cell IDs are valid and required
    nb.nbformat = 4
    nb.nbformat_minor = 5

    used_ids = set()
    for idx, cell in enumerate(nb.cells):
        # Ensure cell has a unique ID
        cid = getattr(cell, "id", None)
        if not cid or cid in used_ids:
            cid = f"cell_{idx:03d}_{uuid.uuid4().hex[:6]}"
            cell["id"] = cid
        used_ids.add(cid)

        # Ensure code cells have clean outputs and execution_count
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # Validate
    try:
        nbformat.validate(nb)
    except Exception as e:
        print(f"VALIDATION ERROR: {e}")
        return False

    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print("VALIDATED & SAVED OK")
    return True

def main():
    notebooks = sorted([p for p in glob.glob("**/*.ipynb", recursive=True) if ".venv" not in p])
    print(f"Found {len(notebooks)} notebooks:")
    all_ok = True
    for nb_path in notebooks:
        ok = sanitize_notebook(nb_path)
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll notebooks successfully sanitized and validated!")
    else:
        print("\nSome notebooks failed validation.")
        exit(1)

if __name__ == "__main__":
    main()
