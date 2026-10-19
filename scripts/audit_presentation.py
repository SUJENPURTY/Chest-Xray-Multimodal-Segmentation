"""Read-only dependency/integrity audit; never execute notebook calculations."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "presentation_audit.json"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def scan():
    dependencies, notebooks = {}, {}
    paths = [ROOT / "app.py"]
    for folder in ("src", "scripts", "tests", "web_app"):
        paths += sorted((ROOT / folder).rglob("*.py"))
    paths += sorted((ROOT / "notebooks").glob("*.ipynb"))
    for path in paths:
        if path.suffix == ".ipynb":
            book = json.loads(path.read_text(encoding="utf-8"))
            source = "\n".join("".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code")
            notebooks[path.name] = {"code_sha256": hashlib.sha256(source.encode()).hexdigest(),
                                   "cell_count": len(book["cells"]),
                                   "outputs": sum(len(c.get("outputs", [])) for c in book["cells"]),
                                   "errors": [i for i, c in enumerate(book["cells"])
                                              if any(o["output_type"] == "error" for o in c.get("outputs", []))]}
        else:
            source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add("." * node.level + (node.module or ""))
        dependencies[path.relative_to(ROOT).as_posix()] = sorted(imports)
    protected = list((ROOT / "src").rglob("*.py")) + [ROOT / "app.py"]
    for folder in ("tests", "pretrained", "text_annotation"):
        protected += [p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for folder in (ROOT / "outputs/run").iterdir():
        if folder.is_dir() and folder.name not in {"execution", "streamlit_demo"}:
            protected += [p for p in folder.rglob("*") if p.is_file()]
    inventory = {}
    for name in ("notebooks", "src", "scripts", "tests", "outputs", "results", "web_app", "Project", "pretrained", "text_annotation", "dist", ".cache"):
        if name in {".cache", "Project"}:
            inventory[name] = {"children": [p.name for p in (ROOT / name).iterdir()],
                               "policy": "Preserve intact; mixed resources/data/caches; no recursive cleanup"}
            continue
        files = [p for p in (ROOT / name).rglob("*") if p.is_file()]
        inventory[name] = {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}
    print("Dependency/inventory scan complete; hashing protected source and artifacts.", flush=True)
    return {"dependencies": dependencies, "notebooks": notebooks, "inventory": inventory,
            "protected": {p.relative_to(ROOT).as_posix(): digest(p) for p in protected}}


def main():
    current = scan()
    if "--baseline" in sys.argv:
        if REPORT.exists():
            raise FileExistsError("Audit baseline already exists; refusing to overwrite it.")
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print("Saved dependency and SHA256 baseline:", REPORT)
    else:
        baseline = json.loads(REPORT.read_text(encoding="utf-8"))
        assert current["protected"] == baseline["protected"], "Protected file hashes changed"
        expected = {name: entry for name, entry in baseline["notebooks"].items() if name.startswith(tuple("0" + str(i) for i in range(1, 7)))}
        assert set(current["notebooks"]) == set(expected)
        for name, entry in expected.items():
            # Empty cells removed for presentation do not alter any code.
            old = json.loads((ROOT / "archive/notebook_originals" / name).read_text(encoding="utf-8"))
            new = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
            nonempty_code = lambda b: [c["source"] for c in b["cells"] if c["cell_type"] == "code" and "".join(c["source"]).strip()]
            assert nonempty_code(old) == nonempty_code(new), name
            old_code = [c for c in old["cells"] if c["cell_type"] == "code" and "".join(c["source"]).strip()]
            new_code = [c for c in new["cells"] if c["cell_type"] == "code" and "".join(c["source"]).strip()]
            for old_cell, new_cell in zip(old_code, new_code):
                assert [o for o in old_cell.get("outputs", []) if o["output_type"] != "error"] == new_cell.get("outputs", []), name
                assert old_cell.get("execution_count") == new_cell.get("execution_count"), name
            assert not current["notebooks"][name]["errors"], name
        for origin, modules in current["dependencies"].items():
            for module in modules:
                if module == "src" or module.startswith("src."):
                    path = ROOT.joinpath(*module.split("."))
                    assert path.with_suffix(".py").is_file() or (path / "__init__.py").is_file(), (origin, module)
        print("PASS: protected SHA256 hashes, six notebook code bodies, and project imports resolve.")


if __name__ == "__main__":
    main()
