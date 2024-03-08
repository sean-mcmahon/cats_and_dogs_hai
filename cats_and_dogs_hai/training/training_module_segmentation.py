import lightning as L
import torch
import torchvision
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryJaccardIndex

from cats_and_dogs_hai.models.segmentation_model import create_segmentation_model


class ResnetModule(L.LightningModule):
    def __init__(self, number_classes: int, learning_rate: float = 1e-4):
        self.learning_rate = learning_rate
        super().__init__()

        self.model = create_segmentation_model(weights_path=None)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.f1_score_val = BinaryF1Score(threshold=0.5)
        self.iou_val = BinaryJaccardIndex(threshold=0.5)

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimiser

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.model(x)
        loss = self.criterion(predictions, y)
        self.log("val_loss", loss)
        self.f1_score_val.update(predictions, y)
        self.iou_val.update(predictions, y)
        return loss

    def on_validation_epoch_end(self):
        f1_score = self.f1_score_val.compute()
        iou = self.iou_val.compute()
        self.log('val_f1_score', f1_score)
        self.log('val_iou', iou)
        self.iou_val.reset()

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.model(x)
        loss = self.criterion(predictions, y)

        self.log("train_loss", loss)
        return loss
