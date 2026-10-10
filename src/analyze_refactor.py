import json
import subprocess
from pathlib import Path
from tabulate import tabulate
from traffic_flow.console import announce_file

announce_file(Path(__file__))

def git(*args): return subprocess.check_output(["git", *args]).decode("utf-8")

def paths(*args): return set(git(*args).split("\0")) - {""}

def count(files, read):
    src = experiments = notebooks = python = 0
    for path in sorted(files):
        if path.startswith("tests/"):
            continue
        if path.endswith(".py"):
            size = len(read(path))
            python += size
            if path.startswith("src/"):
                src += size
            if path.startswith("experiments/"):
                experiments += size
        elif path.endswith(".ipynb"):
            notebook = json.loads(read(path))
            notebooks += sum(
                len("".join(cell.get("source", [])))
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
    return src, experiments, notebooks, python, python + notebooks

ref = git("rev-parse", "origin/main").strip()
main_files = paths("ls-tree", "-r", "--name-only", "-z", ref)

local_files = paths("ls-files", "--cached", "--others", "--exclude-standard", "-z")
ignored = paths("ls-files", "--cached", "--ignored", "--exclude-standard", "-z")
local_files = {p for p in local_files - ignored if Path(p).is_file()}

main = count(main_files, lambda p: git("show", f"{ref}:{p}"))
local = count(local_files, lambda p: Path(p).read_bytes().decode("utf-8"))

labels = [
    "src/ Python",
    "experiments/ Python",
    "Notebook code cells",
    "All .py files",
    "All Python + notebooks",
]

print(f"GitHub main: {ref[:7]}")
print(tabulate(headers=["Scope", "Main", "Local", "Reduction"], tablefmt="outline", tabular_data=[
    (label, f"{old:,}", f"{new:,}", f"{(1 - new / old):.1%}" if old else "N/A")
    for label, old, new in zip(labels, main, local)
]))