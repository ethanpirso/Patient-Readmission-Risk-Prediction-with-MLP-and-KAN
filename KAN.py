import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from kan import KANLayer

class KAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.kan1 = KANLayer(in_dim=13, out_dim=1, k=3)
        self.bias1 = nn.Parameter(torch.zeros(1)).requires_grad_(True)
        self.scale1 = nn.Parameter(torch.ones(1)).requires_grad_(True)
        
    def forward(self, x):
        x, _, _, _ = self.kan1(x)
        x = (x + self.bias1) * self.scale1
        return x

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
# model = KAN()
# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(model, train_dataloader)
# trainer.test(model, test_dataloader)
