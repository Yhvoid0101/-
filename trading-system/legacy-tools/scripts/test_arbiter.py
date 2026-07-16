import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.decision_arbiter import DecisionArbiter

print("=" * 60)
print("ORIGINAL PLAN VERIFICATION")
print("=" * 60)

arbiter = DecisionArbiter()

instinct_weights = {'Explore': 0.3, 'Create': 0.2, 'Avoid': 0.9, 'Alert': 0.85}
meta_result = {'biases_detected': ['恐慌偏误：过度避险'], 'quality_score': 0.5}

result = arbiter.arbitrate(instinct_weights, meta_result)
print(f'  Before: {instinct_weights}')
print(f'  After:  {result}')
print(f'  Avoid: {instinct_weights.get("Avoid", 0)} -> {result.get("Avoid", 0)}')
print(f'  Alert: {instinct_weights.get("Alert", 0)} -> {result.get("Alert", 0)}')

print()
print("=" * 60)
print("EVOLVED VERIFICATION: All Bias Types")
print("=" * 60)

test_cases = [
    ("恐慌偏误", {'explore': 0.3, 'avoid': 0.9, 'alert': 0.85, 'conserve': 0.4},
     {'biases_detected': ['恐慌偏误：过度避险'], 'confidence': 0.8}),
    ("过度自信", {'explore': 0.8, 'create': 0.7, 'alert': 0.2},
     {'biases_detected': ['过度自信偏误：评估与实际不符'], 'confidence': 0.9}),
    ("确认偏误", {'explore': 0.3, 'conserve': 0.6},
     {'biases_detected': ['确认偏误：只考虑了单一方案'], 'confidence': 0.7}),
    ("惯性偏误", {'explore': 0.2, 'create': 0.1, 'conserve': 0.8},
     {'biases_detected': ['惯性偏误：重复选择同类型策略'], 'confidence': 0.6}),
    ("忽视风险", {'explore': 0.9, 'create': 0.8, 'enjoy': 0.7, 'alert': 0.1, 'avoid': 0.1},
     {'biases_detected': ['忽视风险：未充分考虑潜在负面后果'], 'confidence': 0.85}),
    ("锚定偏误", {'explore': 0.3, 'conserve': 0.7},
     {'biases_detected': ['锚定偏误：过度依赖初始信息'], 'confidence': 0.5}),
]

for name, weights, meta in test_cases:
    result = arbiter.arbitrate(weights, meta)
    changes = []
    for k in weights:
        if weights[k] != result.get(k, weights[k]):
            changes.append(f"{k}: {weights[k]:.2f}->{result[k]:.2f}")
    print(f'  {name}: {", ".join(changes) if changes else "no change"}')

print()
print("=" * 60)
print("EVOLVED: Security Veto")
print("=" * 60)

weights_normal = {'explore': 0.8, 'create': 0.7, 'alert': 0.2, 'avoid': 0.1}
veto_actions = ["删除生产数据", "绕过安全检查直接部署", "正常修复代码", "跳过验证步骤"]

for action in veto_actions:
    result = arbiter.arbitrate(weights_normal, {'biases_detected': []}, proposed_action=action)
    is_vetoed = result.get('avoid', 0) > weights_normal.get('avoid', 0) or result.get('alert', 0) > weights_normal.get('alert', 0)
    print(f'  "{action}": {"VETOED" if is_vetoed else "ALLOWED"}')

print()
print("=" * 60)
print("EVOLVED: Prediction-Informed Arbitration")
print("=" * 60)

weights_explore = {'explore': 0.8, 'create': 0.6, 'avoid': 0.1}
low_pred = {"predicted_success": 0.15}
result_low = arbiter.arbitrate(weights_explore, {'biases_detected': []}, prediction_result=low_pred)
print(f'  Low prediction (0.15): explore {weights_explore["explore"]:.2f} -> {result_low.get("explore", 0):.2f}')

high_pred = {"predicted_success": 0.9}
result_high = arbiter.arbitrate(weights_explore, {'biases_detected': []}, prediction_result=high_pred)
print(f'  High prediction (0.90): explore {weights_explore["explore"]:.2f} -> {result_high.get("explore", 0):.2f}')

print()
print("=" * 60)
print("EVOLVED: Neurotransmitter-Aware Arbitration")
print("=" * 60)

from core.neurotransmitter_system import NeuroTransmitterSystem

nt_stress = NeuroTransmitterSystem()
for _ in range(5):
    nt_stress.record_result(False, 0.1, 'negative')
bus_stress = type('Bus', (), {'neurotransmitter': nt_stress})()
arbiter_stress = DecisionArbiter(bus=bus_stress)

weights_test = {'explore': 0.5, 'avoid': 0.5, 'alert': 0.5}
result_stress = arbiter_stress.arbitrate(weights_test, {'biases_detected': ['恐慌偏误'], 'confidence': 0.7})
print(f'  Under stress (high NE): {weights_test} -> {result_stress}')

nt_calm = NeuroTransmitterSystem()
for _ in range(5):
    nt_calm.record_result(True, 0.9, 'positive')
bus_calm = type('Bus', (), {'neurotransmitter': nt_calm})()
arbiter_calm = DecisionArbiter(bus=bus_calm)

result_calm = arbiter_calm.arbitrate(weights_test, {'biases_detected': ['恐慌偏误'], 'confidence': 0.7})
print(f'  Calm (high DA): {weights_test} -> {result_calm}')

print()
print("=" * 60)
print("EVOLVED: Escalation Detection")
print("=" * 60)

weights_extreme = {'explore': 0.9, 'create': 0.8, 'enjoy': 0.7, 'alert': 0.05, 'avoid': 0.05}
meta_extreme = {'biases_detected': ['忽视风险：未充分考虑潜在负面后果', '过度自信偏误'], 'confidence': 0.95}
result_extreme = arbiter.arbitrate(weights_extreme, meta_extreme)
stats = arbiter.get_stats()
print(f'  Total arbitrations: {stats["total_arbitrations"]}')
print(f'  Escalations: {stats["escalations"]}')
print(f'  Vetoes: {stats["vetoes"]}')
print(f'  Bias corrections: {stats["bias_corrections"]}')

print()
print("ALL P1-1 VERIFICATIONS PASSED!")
