import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 60)
print("P0-1: attention_memory.py Real Status Check")
print("=" * 60)

import os
path = os.path.join(os.path.dirname(__file__), '..', 'core', 'attention_memory.py')
path = os.path.normpath(path)
size = os.path.getsize(path)
print(f"  File exists: True")
print(f"  File size: {size} bytes")

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')
    print(f"  Line count: {len(lines)}")

has_class = 'class AttentionMemory' in content
print(f"  Has AttentionMemory class: {has_class}")

has_filter = 'filter_dimensions' in content
has_allocate = 'allocate_attention' in content
has_classify = 'classify_task_type' in content
print(f"  Has filter_dimensions: {has_filter}")
print(f"  Has allocate_attention: {has_allocate}")
print(f"  Has classify_task_type: {has_classify}")

print()
print("--- Import & Instantiation Test ---")
try:
    from core.attention_memory import AttentionMemory
    am = AttentionMemory()
    print("  Import: OK")
    print("  Instantiation: OK")
    print(f"  filter_dimensions: {hasattr(am, 'filter_dimensions')}")
    print(f"  allocate_attention: {hasattr(am, 'allocate_attention')}")
    print(f"  classify_task_type: {hasattr(am, 'classify_task_type')}")
except Exception as e:
    print(f"  Import/Instantiation FAILED: {e}")

print()
print("--- Functional Test ---")
try:
    from core.attention_memory import AttentionMemory
    am = AttentionMemory()

    task_type = am.classify_task_type("修复 agent_bus.py 中的长函数")
    print(f"  classify_task_type('修复...'): {task_type}")

    dims = am.filter_dimensions(task="修复代码", task_type=task_type, num_to_focus=4)
    print(f"  filter_dimensions result: {dims}")
    print(f"  filter_dimensions count: {len(dims) if dims else 0}")

    alloc = am.allocate_attention(task="修复代码", available_dimensions=["clarification", "ambiguity", "constraint", "solution", "risk", "self_challenge"])
    print(f"  allocate_attention result: {alloc}")

    print()
    print("VERDICT: attention_memory.py is a VALID, FUNCTIONAL module")
except Exception as e:
    print(f"  Functional test FAILED: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("VERDICT: attention_memory.py EXISTS but has FUNCTIONAL ISSUES")
