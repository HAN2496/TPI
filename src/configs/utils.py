import yaml
import pickle
import torch

def load_config(driver_name, model_name, config_path='src/configs/config.yaml', verbose=0):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if driver_name in config and model_name in config[driver_name]:
        result = config[driver_name][model_name]
    else:
        result = config

    if verbose:
        print("=== Loaded Configuration ===")
        print(yaml.dump(result, default_flow_style=False))
        print("============================")
    return result

def load_features(feature_version):
    with open('src/configs/features.yaml', 'r') as f:
        features = yaml.safe_load(f)
    return features[feature_version]

def create_model(driver_name, model_name, feature_version, is_train=True, device="cpu", verbose=0):
    from .registries import MODELS
    config = load_config(driver_name, model_name, verbose=verbose)

    args = config['args'].copy()
    if 'input_dim' not in args:
        features = load_features(feature_version)
        args['input_dim'] = len(features)

    model = MODELS[config['model_type']](**args)

    if not is_train:
        if config['model_type'] == 'online_combination':
            with open(f"artifacts/models/{model_name}/model.pt", 'rb') as f:
                state_dict = pickle.load(f)
            model.load_state_dict(state_dict)
        else:
            model = model.to(device)
            state_dict = torch.load(f"artifacts/models/{model_name}/model.pth", map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()
    else:
        if config['model_type'] != 'online_combination':
            model = model.to(device)

    return model