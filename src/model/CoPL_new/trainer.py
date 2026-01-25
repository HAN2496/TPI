import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(current_step: int):
    """Warm-up and cosine decay learning rate schedule"""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = torch.tensor((current_step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
    return 0.5 * (1.0 + torch.cos(progress * torch.pi))

class GCFTrainer:
    def __init__(self, model, logger, config):
        self.model = model
        self.logger = logger
        self.config = config

        self.optimizer = AdamW(model.parameters(), lr=config['lr'],)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.best_val_loss = float('inf')

        self.best_model_path = self.logger.log_dir / "best_model.pt"
        self.n_epochs = config['n_epochs']
        self.device = config["device"]
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_samples = 0
        batch_metrics = defaultdict(list)

        for batch in train_loader:
            pass

    def evaluate(self, val_loader):
        pass

    def train(self, train_loader, val_loader):
        pass
