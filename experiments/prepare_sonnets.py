"""Extract verse from the GITenberg ASCII edition of ebook 1041."""
from pathlib import Path
import re

folder = Path("artifacts/window-study")
source = (folder / "sonnets-source.txt").read_text(encoding="ascii")
body = source[source.index("\n  I\n"):source.index("End of Project Gutenberg's Shakespeare's Sonnets")]
lines = [line.strip() for line in body.splitlines() if line.strip() and not re.fullmatch("[IVXLCDM]+", line.strip())]
(folder / "sonnets.txt").write_text("\n".join(lines)+"\n")
