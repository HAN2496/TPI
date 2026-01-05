import yaml
import pickle
import torch


def load_config(driver_name, model_type, model_name='base', config_path=None, verbose=0):
    if config_path is None:
        config_path = f'src/configs/drivers/{driver_name}.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    result = config[model_type][model_name]
    result['model_type'] = model_type

    if verbose:
        print("=== Loaded Configuration ===")
        print(yaml.dump(result, default_flow_style=False))
        print("============================")
    return result

def create_model(driver_name, model_type, model_name, is_train=True, device="cpu", verbose=0):
    from .registries import MODELS
    from ..model.base import RegressionModel
    config = load_config(driver_name, model_type, model_name, verbose=verbose)

    args = config['args'].copy()
    model = MODELS[config['model_type']](**args)

    if not is_train:
        model_full_name = f"{model_type}_{model_name}" if model_name != 'base' else model_type
        if issubclass(MODELS[config['model_type']], RegressionModel):
            with open(f"artifacts/models/{model_full_name}/model.pt", 'rb') as f:
                state_dict = pickle.load(f)
            model.load_state_dict(state_dict)
        else:
            model = model.to(device)
            state_dict = torch.load(f"artifacts/models/{model_full_name}/model.pth", map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()
    else:
        if issubclass(MODELS[config['model_type']], RegressionModel):
            model = model.to(device)

    return model