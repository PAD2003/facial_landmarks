import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.regression.mae import MeanAbsoluteError

import numpy as np
from PIL import Image, ImageDraw
import torchvision
import pyrootutils
import wandb

# set up python path
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# find root of this file
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

# set up output path for drawed batch
outputs_path = path / "test_outputs"

def draw_batch(images, targets, preds) -> torch.Tensor:
    # helper function
    def denormalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
        """Reverse COLOR transform"""
        # clone: make a copy
        # permute: [batch, 3, H, W] -> [3, H, W, batch]
        tmp = images.clone().permute(1, 2, 3, 0)

        # denormalize
        for t, m, s in zip(tmp, mean, std):
            t.mul_(s).add_(m)

        # clamp: limit value to [0, 1]
        # permute: [3, H, W, batch] -> [batch, 3, H, W]
        return torch.clamp(tmp, 0, 1).permute(3, 0, 1, 2)
    
    def annotate_image(image, targets, preds):
        """Draw target & pred landmarks on image"""
        # create an ImageDraw object
        draw = ImageDraw.Draw(image)

        # draw target_landmarks on image (green)
        for x, y in targets:
            draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(0, 255, 0))

        # draw pred_landmarks on image (red)
        for x, y in preds:
            draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(255, 0, 0))

        return image
    
    # denormalize
    images = denormalize(images)

    # set an empty list
    images_to_save = []

    # loop through each sample in batch
    for i, t, p in zip(images, targets, preds):
        # get size of x
        i = i.permute(1, 2, 0).numpy() * 255
        height, width, color_channels = i.shape

        # denormalize landmarks -> pixel coordinates
        t = (t + 0.5) * np.array( [width, height] )
        p = (p + 0.5) * np.array( [width, height] )

        # draw landmarks on cropped image
        annotated_image = annotate_image(Image.fromarray(i.astype(np.uint8)), t, p)

        # save drawed cropped image
        images_to_save.append( torchvision.transforms.ToTensor()(annotated_image) )

    return torch.stack(images_to_save)

class DlibModule(pl.LightningModule):
    def __init__(self, 
                 net: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler):
        super().__init__()

        # to display the intermediate input- and output sizes
        self.example_input_array = torch.Tensor(16, 3, 224, 224) 

        # save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['net'])

        # set neural network
        self.net = net

        # set loss func
        self.criterion = nn.MSELoss()

        # -> averaging mae across batchs (on each epoch)
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        # -> averaging loss across batchs (on each epoch)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # -> tracking best so far validation accuracy across epochs (on each training time)
        self.val_mae_best = MinMetric()

        # to make use of all the outputs from each validation_step()
        self.validation_step_outputs = []

    def forward(self, x):
        return self.net(x)

    def common_step(self, batch):
        # get data from batch
        x, y = batch

        # forward pass
        y_hat = self.forward(x)

        # calculate the loss
        loss = self.criterion(y_hat, y)

        return x, y, y_hat, loss
    
    def on_train_start(self):
        # reset before training (before all epochs)
        self.val_mae_best.reset()

    def training_step(self, batch, batch_idx):
        # common step
        x, y, y_hat, loss = self.common_step(batch)

        # update metrics
        self.train_loss(loss)
        self.train_mae(y_hat, y)

        # log each step
        self.log("train/loss", self.train_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/mae", self.train_mae, prog_bar=True, on_step=False, on_epoch=True)

        # return input for on_training_epoch_end()
        return loss
    
    def validation_step(self, batch, batch_idx):
        # common step
        x, y, y_hat, loss = self.common_step(batch)

        # update metrics
        self.val_loss(loss)
        self.val_mae(y_hat, y)

        # log each step
        self.log("val/loss", self.val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/mae", self.val_mae, prog_bar=True, on_step=False, on_epoch=True)

        # save image, targets and preds to draw batch in on_validation_epoch_end
        self.validation_step_outputs.append({"image": x, "targets": y, "preds": y_hat})

        # return input for on_validation_epoch_end()
        return loss
    
    def on_validation_epoch_end(self):
        # update the best mae
        acc = self.val_mae.compute()
        self.val_mae_best(acc)
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True, sync_dist=True)

        # get the result of the first batch of validation epoch
        first_val_batch_result = self.validation_step_outputs[0]
        x = first_val_batch_result["image"]
        y = first_val_batch_result["targets"]
        y_hat = first_val_batch_result["preds"]

        # draw the first batch & save it
        annotated_batch = draw_batch(x, y, y_hat)
        torchvision.utils.save_image(annotated_batch, outputs_path / "test_on_validation_epoch_end.png")
        
        # log the first batch
        wandb.log({"annotated_image": wandb.Image(annotated_batch)})

        # free memory & prepare for the next validation epoch
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        # common step
        x, y, y_hat, loss = self.common_step(batch)

        # update metrics
        self.test_loss(loss)
        self.test_mae(y_hat, y)

        # log each step
        self.log("test/loss", self.test_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/mae", self.test_mae, prog_bar=True, on_step=False, on_epoch=True)

        # return input for on_test_epoch_end()
        return loss

    def configure_optimizers(self):
        # set optimizer
        optimizer = self.hparams.optimizer(params=self.parameters())

        # set schedular
        scheduler = self.hparams.scheduler(optimizer=optimizer)

        return optimizer

if __name__ == "__main__":
    from torchinfo import summary
    import hydra
    import pyrootutils
    from omegaconf import DictConfig

    # ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    # set up config path
    config_path = str(path / "configs" / "model")

    # func to test DlibModule
    def test_module(cfg: DictConfig) -> None:
        # create a model
        dlib_model = hydra.utils.instantiate(cfg)

        # show model
        summary(model=dlib_model,
                input_size=(16, 3, 224, 224),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"])
        
        # test input & output shape
        random_input = torch.randn([16, 3, 224, 224])
        output = dlib_model(random_input)
        print(f"\n\nINPUT SHAPE: {random_input.shape}")
        print(f"OUTPUT SHAPE: {output.shape}\n")

    # def main
    @hydra.main(version_base="1.3", config_path=config_path, config_name="dlib.yaml")
    def main(cfg: DictConfig):
        test_module(cfg)

    # call main
    main()

