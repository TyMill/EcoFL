from ecofl.benchmark.pipeline import BenchmarkPipeline


def test_pipeline_quick_single_model_runs():
    pipeline = BenchmarkPipeline(
        n_clients=3,
        n_rounds=2,
        n_samples=600,
        hardware_profile="raspberry_pi4",
        random_state=42,
        verbose=False,
    )
    results = pipeline.run_all(model_names=["LogisticRegression"])
    assert len(results) == 3
    assert {r["configuration"] for r in results} == {"centralized", "fedavg", "ecofl"}
