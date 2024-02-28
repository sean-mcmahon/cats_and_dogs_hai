import lightning as L
import torch
import torchvision


class ResnetModule(L.LightningModule):

    def __init__(self, number_classes:int):
        super().__init__()

        self.model = torchvision.models.resnet18(num_classes=number_classes)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    
    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimiser


    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)

    def on_validation_epoch_end(self):
        # TODO Gather metrics
        # all_preds = torch.stack(self.validation_step_outputs)
        pass

    def training_step(self, batch, batch_idx):

        x,y = batch
        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def on_train_epoch_end(self):
        # TODO metrics
        # _preds = torch.stack(self.training_step_outputs)
        pass
