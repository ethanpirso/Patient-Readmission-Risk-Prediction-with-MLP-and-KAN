import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from kan import KANLayer

class KAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.kan1 = KANLayer(in_dim=13, out_dim=16, k=3)
        self.kan2 = KANLayer(in_dim=16, out_dim=32, k=3)
        self.kan3 = KANLayer(in_dim=32, out_dim=16, k=3)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x, preacts1, postacts1, postspline1 = self.kan1(x)
        x, preacts2, postacts2, postspline2 = self.kan2(x)
        x, preacts3, postacts3, postspline3 = self.kan3(x)
        x = x.view(x.size(0), -1)  # Flatten the output to match the input dimension of fc1
        x = F.relu(self.fc1(x))  # Added a ReLU activation function for non-linearity before the final output
        x = self.fc2(x)
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
# model = KAN()
# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(model, train_dataloader)
# trainer.test(model, test_dataloader)
