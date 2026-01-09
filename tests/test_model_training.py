def test_train_model_runs(split_data, scaled_data):
    _, y = split_data
    from src.model_training import train_model
    model = train_model(scaled_data, y)
    assert model is not None

def test_model_predicts(split_data, scaled_data):
    _, y = split_data
    from src.model_training import train_model
    model = train_model(scaled_data, y)
    preds = model.predict(scaled_data)
    assert len(preds) == len(y)
