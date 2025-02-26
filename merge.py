import nbformat
import glob
import uuid

def merge_and_fix_notebooks(output_filename="merged_notebook.ipynb"):
    notebooks = sorted(glob.glob("*.ipynb"))  # Get all notebooks

    if not notebooks:
        print("No notebooks found.")
        return

    merged_nb = nbformat.v4.new_notebook()

    for nb_file in notebooks:
        with open(nb_file, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

            for cell in nb.cells:
                if "id" in cell:
                    cell["id"] = str(uuid.uuid4())  # Assign a new unique ID

            merged_nb.cells.extend(nb.cells)  # Append cells while keeping outputs

    with open(output_filename, "w", encoding="utf-8") as f:
        nbformat.write(merged_nb, f)

    print(f"Merged {len(notebooks)} notebooks into {output_filename} with unique cell IDs.")

# Run the function
merge_and_fix_notebooks("merged_output.ipynb")