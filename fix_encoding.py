import sys
from pathlib import Path

root = Path(r"c:/Users/gupta/Desktop/SAFE/src")
files = list(root.rglob("*.py"))
files.append(Path(r"c:/Users/gupta/Desktop/SAFE/run_pipeline.py"))

repls = [
    ("\u2705", "OK:"),
    ("\u274c", "FAIL:"),
    ("\u26a0", "WARN:"),
    ("\u2192", "->"),
    ("\u2714", "OK"),
    ("\u2716", "X"),
    ("\u25b6", ">"),
    ("\u25cf", "*"),
]

reconfigure_line = 'sys.stdout.reconfigure(encoding="utf-8", errors="replace")\n'

for fpath in files:
    text = fpath.read_text(encoding="utf-8", errors="replace")
    changed = False

    for uchar, repl in repls:
        if uchar in text:
            text = text.replace(uchar, repl)
            changed = True

    if "stdout.reconfigure" not in text:
        # Insert after the first occurrence of 'import sys'
        idx = text.find("import sys\n")
        if idx != -1:
            insert_pos = idx + len("import sys\n")
            text = text[:insert_pos] + reconfigure_line + text[insert_pos:]
            changed = True

    if changed:
        fpath.write_text(text, encoding="utf-8")
        print(f"Fixed: {fpath.name}")
    else:
        print(f"OK   : {fpath.name}")
