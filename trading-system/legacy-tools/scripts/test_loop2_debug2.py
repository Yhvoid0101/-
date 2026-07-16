import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.predictive_engine import PredictiveEngine
from core.cognitive_network import CognitiveNetwork
from core.neurotransmitter_system import NeuroTransmitterSystem

nt = NeuroTransmitterSystem()
bus = type('Bus', (), {'neurotransmitter': nt})()
pe = PredictiveEngine(bus=bus)
cn = CognitiveNetwork(bus=bus)
bus.cognitive_network = cn

pred = pe.predict("fix")
print(f"Predicted success: {pred.predicted_success}")
print(f"Confidence: {pred.confidence}")

pe.resolve_prediction(pred, False, "high")
print(f"Error: {pred.error}")
print(f"Was surprise: {pred.was_surprise}")
print(f"Was accurate: {pred.was_accurate}")

for _ in range(5):
    pred2 = pe.predict("fix")
    pe.resolve_prediction(pred2, False, "high")
    print(f"  Iteration: error={pred2.error:.3f} surprise={pred2.was_surprise}")

fix_sensors = ['sensor_fix_bug']
changed = 0
for sk, s in cn.synapses.items():
    if s.source_id in fix_sensors and s.type == "excitatory":
        if s.weight < 0.5:
            changed += 1
            print(f"  Weight decreased: {sk} -> {s.weight:.3f}")
print(f"Total weights decreased: {changed}")
