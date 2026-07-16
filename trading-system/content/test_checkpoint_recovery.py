import json
from pathlib import Path
from types import SimpleNamespace

from sandbox_trading.deployment_infra import CheckpointManager
from sandbox_trading.population import PopulationManager


def test_load_population_latest_uses_numeric_generation_and_skips_empty(tmp_path):
    manager = PopulationManager(population_size=1, data_dir=str(tmp_path))
    (tmp_path / 'population_gen0002.jsonl').write_text('', encoding='utf-8')
    (tmp_path / 'population_gen0010.jsonl').write_text(
        json.dumps({'agent_id': 'agent-10', 'generation': 10}) + '\n',
        encoding='utf-8',
    )
    (tmp_path / 'population_gen0009.jsonl').write_text(
        json.dumps({'agent_id': 'agent-9', 'generation': 9}) + '\n',
        encoding='utf-8',
    )

    loaded = manager.load_population(-1)

    assert loaded
    assert manager.generation == 10


def test_find_latest_checkpoint_ignores_empty_and_invalid_files(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    (tmp_path / 'ckpt_gen1_empty.json').write_text('', encoding='utf-8')
    (tmp_path / 'ckpt_gen2_invalid.json').write_text('{', encoding='utf-8')
    valid = tmp_path / 'ckpt_gen3_valid.json'
    valid.write_text(json.dumps({'generation': 3, 'agents': [{'id': 'a'}]}), encoding='utf-8')

    assert manager.find_latest_checkpoint() == str(valid)


def test_evolution_loop_saves_checkpoint_after_evolution(monkeypatch):
    from sandbox_trading import evolution_loop

    calls = []

    class CheckpointSpy:
        def save_checkpoint(self, population, generation, description=''):
            calls.append((population, generation, description))

    monitor = SimpleNamespace(checkpoint_mgr=CheckpointSpy())
    monkeypatch.setattr(evolution_loop, 'EvolutionMonitor', lambda **kwargs: monitor)

    source = Path(evolution_loop.__file__).read_text(encoding='utf-8')
    assert 'self.evolution_monitor.checkpoint_mgr.save_checkpoint' in source
