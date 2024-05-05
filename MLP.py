import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

class MLP(pl.LightningModule):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(13, 64)  # Input layer to hidden layer
        self.fc2 = nn.Linear(64, 32)  # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(32, 1)   # Hidden layer to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for hidden layer
        x = F.relu(self.fc2(x))  # Activation function for hidden layer
        x = F.sigmoid(self.fc3(x))  # Activation function for output layer
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)  # T_max is the maximum number of iterations
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss_fn_class = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn_class(y_hat, y)
        self.log('train_loss', loss)
        acc = ((y_hat.round() == y).float()).mean()
        self.log('train_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss_fn_class = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn_class(y_hat, y)
        self.log('test_loss', loss)
        acc = ((y_hat.round() == y).float()).mean()
        self.log('test_acc', acc)
        return loss

# Example usage
# model = MLP()
# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(model, train_dataloader)
# trainer.test(model, test_dataloader)
