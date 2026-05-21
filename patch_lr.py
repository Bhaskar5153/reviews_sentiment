import json

with open("notebooks/eda.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "LogisticRegression(max_iter=1000" in src:
        cell["source"] = [src.replace("LogisticRegression(max_iter=1000", "LogisticRegression(max_iter=3000, solver='saga'")]
        cell["outputs"] = []
        cell["execution_count"] = None
        print("Patched LR cell")
        break

with open("notebooks/eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)