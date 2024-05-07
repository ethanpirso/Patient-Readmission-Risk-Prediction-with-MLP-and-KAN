import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

class MLP(pl.LightningModule):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001)
        return [optimizer]#, [scheduler]

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
