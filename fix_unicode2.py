from pathlib import Path

extra_repls = [
    ("\u2500", "-"),
    ("\u2502", "|"),
    ("\u250c", "+"),
    ("\u2514", "+"),
    ("\u251c", "+"),
    ("\u2510", "+"),
    ("\u2518", "+"),
    ("\u2524", "+"),
    ("\u2212", "-"),
    ("\u00b0", "deg"),
    ("\u2019", "'"),
]

root = Path(r"c:/Users/gupta/Desktop/SAFE/src")
files = list(root.rglob("*.py"))

for fpath in files:
    text = fpath.read_text(encoding="utf-8", errors="replace")
    changed = False
    for uchar, repl in extra_repls:
        if uchar in text:
            text = text.replace(uchar, repl)
            changed = True
    if changed:
        fpath.write_text(text, encoding="utf-8")
        print(f"Fixed: {fpath.name}")
    else:
        print(f"OK:    {fpath.name}")
