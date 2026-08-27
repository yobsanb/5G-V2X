from pathlib import Path

ROOT = Path(__file__).parents[1]
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", ".venv", "data", "outputs"}


def test_project_text_and_paths_are_ascii():
    failures = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            path.relative_to(ROOT)
        except ValueError:
            continue
        if not path.name.isascii():
            failures.append(str(path.relative_to(ROOT)))
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            path.read_text(encoding="ascii")
        except (UnicodeDecodeError, UnicodeEncodeError):
            failures.append(str(path.relative_to(ROOT)))
    assert not failures, "Non-ASCII project content: " + ", ".join(sorted(set(failures)))
