from lxml import etree
import cairosvg
import os

INPUT = "generated_samples.txt"
OUT_DIR = "plots/generated"
os.makedirs(OUT_DIR, exist_ok=True)

with open(INPUT) as f:
    raw = f.read()

samples = []
for block in raw.split("--- SAMPLE"):
    if "<svg" in block:
        svg = block[block.index("<svg"):].strip()
        samples.append(svg)

valid_xml = 0
rendered = 0

for i, svg in enumerate(samples):
    try:
        etree.fromstring(svg.encode("utf-8"))
        valid_xml += 1
    except Exception:
        continue

    try:
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=f"{OUT_DIR}/sample_{i}.png"
        )
        rendered += 1
    except Exception:
        pass

print("Total samples:", len(samples))
print("XML valid:", valid_xml)
print("XML validity rate:", valid_xml / max(1, len(samples)))
print("Rendered:", rendered)
print("Render rate:", rendered / max(1, len(samples)))
