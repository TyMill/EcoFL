from ecofl.models.lightweight import MODEL_CONFIGS, create_model


def test_all_registered_models_can_be_created():
    for model_name in MODEL_CONFIGS:
        model = create_model(model_name)
        assert model is not None
