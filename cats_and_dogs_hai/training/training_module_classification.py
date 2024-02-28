import lightning as L
import torch
import torchvision


class ResnetModule(L.LightningModule):

    def __init__(self, number_classes: int, learning_rate: float = 1e-3):
        self.learning_rate = learning_rate
        super().__init__()
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        num_fc_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_fc_features, number_classes)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimiser

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        # TODO Gather metrics
        # all_preds = torch.stack(self.validation_step_outputs)
        pass

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        # TODO metrics
        # _preds = torch.stack(self.training_step_outputs)
        pass
