import re
from pathlib import Path

p = Path("scripts/make_report.py")
text = p.read_text(encoding="utf-8")

# Replace all non-black RGBColor(...) with RGBColor(0, 0, 0)
text = re.sub(r'RGBColor\([^\)]+\)', 'RGBColor(0, 0, 0)', text)

# Replace table border color
text = text.replace('color="CBD5E0"', 'color="000000"')

# Replace table header background with light gray #F2F2F2
text = text.replace('"1A365D"', '"F2F2F2"')

p.write_text(text, encoding="utf-8")
print("[+] Successfully converted make_report.py to 100% pure black styling!")
