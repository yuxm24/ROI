"""Keep documentation navigation valid as the public interface evolves."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]


def test_relative_markdown_links():
    documents = [
        *ROOT.glob("*.md"),
        *(ROOT / "deployment").glob("*.md"),
        *(ROOT / "deployment/docs").rglob("*.md"),
    ]
    errors = []
    for document in documents:
        content = document.read_text()
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            target = document.parent / unquote(parsed.path) if parsed.path else document
            if not target.exists():
                errors.append(f"{document.relative_to(ROOT)}: missing {link}")
                continue
            if parsed.fragment and target.suffix == ".md":
                headings = re.findall(r"^#+ (.+)$", target.read_text(), re.MULTILINE)
                anchors = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}
                if unquote(parsed.fragment) not in anchors:
                    errors.append(f"{document.relative_to(ROOT)}: missing anchor {link}")
    assert not errors, "\n".join(errors)
