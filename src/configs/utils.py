import yaml
import torch

def load_config(driver_name, model_name, config_path='src/configs/config.yaml', verbose=0):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if driver_name in config and model_name in config[driver_name]:
        result = config[driver_name][model_name]
    else:
        result = config

    if verbose:
        print(yaml.dump(result, default_flow_style=False))
    return result

def load_features(feature_version):
    with open('src/configs/features.yaml', 'r') as f:
        features = yaml.safe_load(f)
    return features[feature_version]

def create_model(driver_name, model_name, is_train=True, device="cpu", verbose=0):
    from src.configs.registries import MODELS
    config = load_config(driver_name, model_name, verbose=verbose)

    args = config['args'].copy()
    if 'input_dim' not in args:
        features = load_features(config['feature_version'])
        args['input_dim'] = len(features)

    model = MODELS[config['model_type']](**args)

    if config['model_type'] != 'online_combination':
        model = model.to(device)
        if not is_train:
            state_dict = torch.load(f"artifacts/models/{model_name}/model.pth", map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()

    return model