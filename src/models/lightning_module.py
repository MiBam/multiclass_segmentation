import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .unet import UNet  # If you move UNet to a separate module too
import cv2

class MulticlassSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = UNet(in_channels=3, num_classes=num_classes)
        class_weights = torch.tensor([1.0, 10.0, 10.0], dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        self.log("train_loss", loss)
        if batch_idx % 50 == 0:
            self.log_images(images, logits, masks, "Train")
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        self.log("val_loss", loss)
        preds = torch.argmax(logits, dim=1)
        if batch_idx % 10 == 0:
            self.save_predictions(images, preds, masks, batch_idx)
        return loss

    def class_indices_to_grayscale(self, class_mask):
        class_to_gray = {0: 0, 1: 175, 2: 255}
        grayscale_mask = np.vectorize(class_to_gray.get)(class_mask).astype(np.uint8)
        return grayscale_mask

    def save_predictions(self, images, preds, masks, batch_idx):
        save_dir = Path("val_predictions")
        save_dir.mkdir(exist_ok=True, parents=True)
        for i in range(images.shape[0]):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            pred_mask = preds[i].cpu().numpy()
            pred_mask_grayscale = self.class_indices_to_grayscale(pred_mask)
            true_mask = masks[i].cpu().numpy()
            true_mask_grayscale = self.class_indices_to_grayscale(true_mask)
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(img)
            ax[0].set_title("Image")
            ax[1].imshow(true_mask_grayscale, cmap='gray')
            ax[1].set_title("Ground Truth")
            ax[2].imshow(pred_mask_grayscale, cmap='gray')
            ax[2].set_title("Prediction")
            plt.tight_layout()
            plt.savefig(save_dir / f"batch{batch_idx}_img{i}.png")
            plt.close(fig)
            cv2.imwrite(str(save_dir / f"batch{batch_idx}_img{i}_pred.png"), pred_mask_grayscale)
            cv2.imwrite(str(save_dir / f"batch{batch_idx}_img{i}_true.png"), true_mask_grayscale)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-5)

    def log_images(self, images, logits, masks, stage):
        pred = torch.argmax(logits, dim=1)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(images[0].permute(1, 2, 0).cpu())
        ax[0].set_title('Input')
        ax[1].imshow(masks[0].cpu(), cmap='gray')
        ax[1].set_title('Ground Truth')
        ax[2].imshow(pred[0].cpu(), cmap='gray')
        ax[2].set_title('Prediction')
        plt.tight_layout()
        self.logger.experiment.add_figure(f"{stage}_images", fig, self.global_step)
        plt.close(fig)
