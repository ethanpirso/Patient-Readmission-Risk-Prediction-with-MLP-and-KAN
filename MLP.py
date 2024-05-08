import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('train_loss', loss)
        preds = torch.sigmoid(y_hat).round()
        accuracy = (preds == y).float().mean()
        self.log('train_accuracy', accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('test_loss', loss)
        preds = torch.sigmoid(y_hat).round()
        accuracy = (preds == y).float().mean()
        self.log('test_accuracy', accuracy)
        return loss

# Example usage
# model = MLP()
# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(model, train_dataloader)
# trainer.test(model, test_dataloader)
