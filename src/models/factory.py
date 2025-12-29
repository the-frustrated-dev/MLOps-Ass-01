from importlib import import_module

def load_class(path):
    module_name, class_name = path.rsplit(".", 1)
    return getattr(import_module(module_name), class_name)

def create_model(model_cfg, params={}):
    class_ = load_class(model_cfg["class"])
    fixed = model_cfg.get("fixed_params", {})
    return class_(**fixed, **params)