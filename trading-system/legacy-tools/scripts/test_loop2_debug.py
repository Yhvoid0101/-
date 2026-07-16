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
print(f"Task type: {pred.task_type}")

fix_sensors = [f"sensor_{kw}" for kw in cn.SENSOR_SYNONYMS
               if "fix" in kw or any("fix" in s for s in cn.SENSOR_SYNONYMS.get(kw, []))]
print(f"Matching sensors: {fix_sensors}")

initial = {}
for sk, s in cn.synapses.items():
    if s.source_id in fix_sensors and s.type == "excitatory":
        initial[sk] = s.weight

print(f"Initial weights count: {len(initial)}")

pe.resolve_prediction(pred, False, "high")

changed = 0
for sk, s in cn.synapses.items():
    if s.source_id in fix_sensors and s.type == "excitatory":
        if sk in initial and s.weight != initial[sk]:
            changed += 1
            print(f"  Changed: {sk} {initial[sk]:.3f} -> {s.weight:.3f}")

print(f"Weights changed: {changed}")
