from src.train import train_and_save_model

def test_accuracy_threshold():
    acc = train_and_save_model()
    assert acc > 0.8, "Accuracy below expected threshold!"
