import lightning as L
import torch
import torchvision
from torchmetrics.classification import MultilabelF1Score
from cats_and_dogs_hai.models.classification_model import create_classification_model


class ClassificationTrainModule(L.LightningModule):

    def __init__(self, number_classes: int, learning_rate: float = 1e-3):
        self.learning_rate = learning_rate
        super().__init__()

        self.model = create_classification_model(number_classes)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.f1_score_val = MultilabelF1Score(num_labels=number_classes, threshold=0.7)

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimiser

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.model(x)
        loss = self.criterion(predictions, y)
        self.log("val_loss", loss)
        self.f1_score_val.update(predictions, y)
        return loss

    def on_validation_epoch_end(self):
        f1_score = self.f1_score_val.compute()
        self.log('val_f1_score', f1_score)
        self.f1_score_val.reset()

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.model(x)
        loss = self.criterion(predictions, y)

        self.log("train_loss", loss)
        return loss
