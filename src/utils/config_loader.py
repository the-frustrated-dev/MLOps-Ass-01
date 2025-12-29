from yaml import safe_load

def load_cfg(cfg_path):
    with open(cfg_path, "r") as f:
        data = safe_load(f)
    return data