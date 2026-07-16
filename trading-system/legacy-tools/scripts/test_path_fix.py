import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

modules = [
    'core.hermes_brain', 'core.deep_think', 'core.cognitive_network',
    'core.attention_memory', 'core.instinct_engine', 'core.meta_cognition',
    'core.predictive_engine', 'core.temporal_integration',
    'core.cognitive_topology', 'core.deep_causal_reasoner',
    'core.neurotransmitter_system', 'core.memory_crystallizer',
    'core.obsidian_bridge', 'core.adaptive_router',
    'core.cognitive_loop', 'core.quality_gate',
    'core.anti_hallucination', 'core.agent_factory',
]

print("=" * 60)
print("Path Fix Verification: All Module Imports")
print("=" * 60)

passed = 0
failed = 0
for m in modules:
    try:
        __import__(m)
        print(f'  OK: {m}')
        passed += 1
    except Exception as e:
        print(f'  FAIL: {m}: {e}')
        failed += 1

print()
print(f"Result: {passed}/{passed+failed} modules imported successfully")

print()
print("--- Hardcoded Path Scan ---")
from pathlib import Path
core_dir = Path(__file__).resolve().parent.parent / 'core'
found = 0
for py_file in core_dir.glob('*.py'):
    content = py_file.read_text(encoding='utf-8', errors='ignore')
    if r'c:\Users' in content or r'C:\Users' in content:
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if r'c:\Users' in line or r'C:\Users' in line:
                print(f'  FOUND: {py_file.name}:{i}: {line.strip()[:80]}')
                found += 1

if found == 0:
    print("  No hardcoded Windows paths in core/ modules")
else:
    print(f"  WARNING: {found} hardcoded paths still found")

print()
print("--- ObsidianBridge Path Resolution Test ---")
from core.obsidian_bridge import ObsidianBridge
ob = ObsidianBridge()
print(f"  Vault path: {ob._vault_path}")
print(f"  Vault exists: {ob._vault_path.exists()}")
