import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.meta_cognition import MetaCognition
from core.neurotransmitter_system import NeuroTransmitterSystem

nt = NeuroTransmitterSystem()
bus = type('Bus', (), {'neurotransmitter': nt})()
mc = MetaCognition(bus=bus)

try:
    mc.reflect({'task': None, 'plan': None, 'result': None, 'evaluation': None})
except Exception as e:
    traceback.print_exc()
