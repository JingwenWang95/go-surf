import os
import yaml


def load_config(exp_name=None, scene="", use_config_snapshot=False):
    log_dir = os.path.join(os.path.dirname(__file__), "logs", scene)
    if exp_name:
        log_dir = os.path.join(log_dir, exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    config_file = scene.split("/")[-1] + ".yaml"
    
    if use_config_snapshot:  # log complete config file, i.e. from log_dir
        config_path = os.path.join(log_dir, config_file)
        
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:  # load from configs/xxxx.yaml, i.e. start a new training
        base_config_path = os.path.join(os.path.dirname(__file__),  "configs/base.yaml")
        config_path = os.path.join(os.path.dirname(__file__),  "configs", config_file)
        
        with open(base_config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = {**config, **yaml.load(f, Loader=yaml.FullLoader)}

        with open(os.path.join(log_dir, config_file), "w") as f:
            yaml.dump(config, f, indent=4, default_flow_style=None, sort_keys=False)
        
    config["log_dir"] = log_dir
    config["checkpoints_dir"] = os.path.join(log_dir, "checkpoints/")

    if not os.path.exists(config["checkpoints_dir"]):
        os.makedirs(config["checkpoints_dir"])

    return config