# -*- coding: utf-8 -*-
"""Final verification: all modules initialized"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hermes_brain import HermesBrain
from core.neurotransmitter_system import NeuroTransmitterSystem

nt = NeuroTransmitterSystem()
bus = type('Bus', (), {
    'neurotransmitter': nt, '_physical_layer': None,
    '_neurosymbolic_verifier': None, '_streaming_alignment': None,
    'instinct_engine': None, 'dag_orchestrator': None,
    'permission_guard': None, 'meta_cognition': None,
    'deep_think': None,
})()
brain = HermesBrain(bus=bus)

modules = [
    ('NeuroTransmitter', brain._neurotransmitter),
    ('InstinctEngine', brain._instinct_engine),
    ('MemoryCrystallizer', brain._memory_crystallizer),
    ('CreativityEngine', brain._creativity_engine),
    ('MetaEvolution', brain._meta_evolution),
    ('StreamingAlignment', brain._streaming_alignment),
    ('AgentGovernance', brain._agent_governance),
    ('SelfOrganizingSwarm', brain._self_organizing_swarm),
    ('NeurosymbolicVerifier', brain._neurosymbolic_verifier),
    ('PEASecurity', brain._pea_security),
    ('SocietyOfThought', brain._society_of_thought),
    ('DigitalPersonhood', brain._digital_personhood),
    ('OpenEndedEvolution', brain._open_ended_evolution),
    ('IntrinsicReward', brain._intrinsic_reward),
    ('LifelongMemory', brain._lifelong_memory),
    ('MotivationalDistillation', brain._motivational_distillation),
    ('CrossConsciousness', brain._cross_consciousness),
    ('SelfAwareness', brain._self_awareness),
]

ok = 0
for name, mod in modules:
    status = 'OK' if mod is not None else 'MISSING'
    if mod is not None:
        ok += 1
    print(f'  {name}: {status}')

print(f'\n{ok}/{len(modules)} modules initialized')
if ok == len(modules):
    print('ALL 19 SYSTEMS OPERATIONAL!')
else:
    print(f'WARNING: {len(modules) - ok} modules missing')
