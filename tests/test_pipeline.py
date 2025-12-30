import pytest
import pandas as pd
import numpy as np
from src.data.load_data import load_data
from src.features.preprocess import preprocess
from src.models.factory import create_model
from src.utils.config_loader import load_cfg

# Test Config Loader
def test_load_cfg(tmp_path):
    # Create a temporary yaml file
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test_config.yaml"
    p.write_text("key: value\nlist: [1, 2]")
    
    cfg = load_cfg(str(p))
    assert cfg["key"] == "value"
    assert cfg["list"] == [1, 2]

# Test Model Factory
def test_create_model():
    model_cfg = {
        "class": "sklearn.linear_model.LogisticRegression",
        "fixed_params": {"max_iter": 100}
    }
    model = create_model(model_cfg, params={"C": 0.5})
    assert model.max_iter == 100
    assert model.get_params()["C"] == 0.5

# Test Preprocessing logic
def test_preprocess_scaling():
    # Create dummy data where mean is not 0 and std is not 1
    X_train = np.array([[10.0, 2.0], [20.0, 4.0], [30.0, 6.0]])
    X_test = np.array([[15.0, 3.0]])
    
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)
    
    # After StandardScaler, mean should be near 0 and std near 1 for train
    assert np.allclose(X_train_scaled.mean(axis=0), 0)
    assert np.allclose(X_train_scaled.std(axis=0), 1)
    # Ensure test was also transformed
    assert X_test_scaled.shape == (1, 2)

# Test Data Loading (Mocking the file read)
def test_load_data_logic(monkeypatch):
    # Mocking pd.read_csv to return a dataframe where we have enough rows
    # 25 rows total * 0.2 test_size = 5 rows in test set (exactly 1 per class)
    def mock_read_csv(*args, **kwargs):
        data = {
            'age': np.random.randint(20, 80, 25),
            'nums': [0, 0, 0, 0, 0, 
                     1, 1, 1, 1, 1, 
                     2, 2, 2, 2, 2, 
                     3, 3, 3, 3, 3, 
                     4, 4, 4, 4, 4]
        }
        return pd.DataFrame(data)
    
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    
    cfg = {
        "path": "dummy_path",
        "cols": ['age', 'nums'],
        "target_col": "nums",
        "test_size": 0.2, # 25 * 0.2 = 5
        "random_state": 42
    }
    
    X_train, X_test, y_train, y_test = load_data(cfg)
    
    # Check that test set size is correct and handles all classes
    assert len(X_test) == 5
    assert len(X_train) == 20
    assert len(np.unique(y_test)) == 5 # Every class represented
