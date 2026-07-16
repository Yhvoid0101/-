from sandbox_trading.population import PopulationManager


def test_evolution_round_applies_risk_rejections_to_agent_fitness(tmp_path):
    manager = PopulationManager(
        population_size=1,
        data_dir=str(tmp_path),
        use_nsga2=False,
        use_nsga3=False,
        use_feature_map=False,
    )
    agent = manager.initialize_population()[0]

    manager.run_evolution_round({agent.agent_id: []})
    baseline_score = manager._last_scores[agent.agent_id].gt_score

    manager = PopulationManager(
        population_size=1,
        data_dir=str(tmp_path / 'rejected'),
        use_nsga2=False,
        use_nsga3=False,
        use_feature_map=False,
    )
    agent = manager.initialize_population()[0]
    manager.run_evolution_round(
        {agent.agent_id: []},
        risk_rejections_by_agent={agent.agent_id: 2},
    )

    fitness = manager._last_scores[agent.agent_id]
    assert fitness.risk_rejection_count == 2
    assert fitness.gt_score < baseline_score



def test_population_records_hard_risk_lessons_in_child_lineage(tmp_path):
    manager = PopulationManager(
        population_size=10,
        data_dir=str(tmp_path),
        use_nsga2=False,
        use_nsga3=False,
        use_feature_map=False,
    )
    agents = manager.initialize_population()

    manager.run_evolution_round(
        {agent.agent_id: [] for agent in agents},
        err_lessons=[
            {
                "err_id": "risk_hard_rejection",
                "severity": "BLOCK",
                "root_cause": "hard_risk_rejection_count",
                "related_module": agents[0].agent_id,
                "count": 2,
            }
        ],
    )

    assert manager._gene_err_map
    assert all(
        "risk_hard_rejection" in lesson_ids
        for lesson_ids in manager._gene_err_map.values()
    )
