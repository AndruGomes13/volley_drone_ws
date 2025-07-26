#!/usr/bin/env python3
import json, pathlib, itertools, sys
root = pathlib.Path('.')
combined = []
for f in root.glob('build/**/compile_commands.json'):
    combined += json.loads(f.read_text())
(root / 'build/compile_commands.json').write_text(json.dumps(combined, indent=2))
print(f"Merged compile_commands.json files into {root / 'build/compile_commands.json'}")
