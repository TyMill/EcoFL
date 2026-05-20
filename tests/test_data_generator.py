from ecofl.data.generator import make_dataset, FEATURE_NAMES


def test_make_dataset_shapes_and_partitions():
    data = make_dataset(n_samples=1000, n_clients=5, random_state=42)
    assert data["X_train_all"].shape[1] == len(FEATURE_NAMES)
    assert data["X_test"].shape[1] == len(FEATURE_NAMES)
    assert len(data["client_partitions"]) == 5
    assert sum(len(y) for _, y in data["client_partitions"]) == len(data["y_train_all"])
    assert data["meta"]["n_features"] == 12
