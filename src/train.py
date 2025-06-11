import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.dataset import SegmentationDataset
from src.models.lightning_module import MulticlassSegmentationModel
from pytorch_lightning.callbacks import EarlyStopping

if __name__ == "__main__":
    image_dir = 'data/train/images'
    mask_dir = 'data/train/masks'

    train_dataset = SegmentationDataset(image_dir, mask_dir, transform=None)
    val_dataset = SegmentationDataset(image_dir, mask_dir, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = MulticlassSegmentationModel(num_classes=3)

    trainer = pl.Trainer(
        max_epochs=500,
        accelerator='cpu',
        log_every_n_steps=10,
        enable_checkpointing=True,
        logger=pl.loggers.TensorBoardLogger('logs/'),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
    )

    trainer.fit(model, train_loader, val_loader)
