from typing import Any, Dict, Optional
import torch
import torchvision
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
import albumentations as A
from components.dlib_dataset import DlibDataset
from components.dlib_transform_dataset import DlibTransformDataset

class DlibDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_train: DlibDataset,
                 data_test: DlibDataset,
                 data_dir: str,
                 train_val_test_split = [5666, 1000],
                 transform_train: Optional[A.Compose] = None,
                 transform_val: Optional[A.Compose] = None,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False) -> None:
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(logger=False)

    def prepare_data(self) -> None:
        """Download data if needed"""
        pass

    def setup(self, stage: Optional[str]=None) -> None:
        """split, transforms, etc..."""
        # set up train & val dataset
        data_train_val = self.hparams.data_train(data_dir=self.hparams.data_dir)
        data_train, data_val = random_split(data_train_val, self.hparams.train_val_test_split)

        # set up test dataset
        data_test = self.hparams.data_test(data_dir=self.hparams.data_dir)

        # set up DlibTransformDataset
        self.data_train = DlibTransformDataset(data_train, self.hparams.transform_train)
        self.data_val = DlibTransformDataset(data_val, self.hparams.transform_val)
        self.data_test = DlibTransformDataset(data_test, self.hparams.transform_val)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data_train,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data_val,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data_test,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig, OmegaConf
    import hydra
    from tqdm import tqdm

    # ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    # set up python path
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    # find root of this file
    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

    # set up config path
    config_path = str(path / "configs" / "data")

    # set up output path
    outputs_path = path / "test_outputs"

    # test dataset
    def test_dataset(cfg: DictConfig):
        print("TEST DATASET\n")

        # create a DlibDataset
        dataset: DlibDataset = hydra.utils.instantiate(cfg.data_train)
        
        # set up data_dir for dataset
        dataset = dataset(data_dir=cfg.data_dir)

        # test __len__()
        print(f"Length of train dataset: {len(dataset)}\n")

        # test __getitem__()
        cropped_image, landmarks = dataset[0]

        # test a sample
        print(f"Size of image: {cropped_image.size}")
        print(f"Shape of landmarks: {landmarks.shape}")

        # test annotate_image()
        annotated_image = dataset.annotate_image(cropped_image, landmarks)
        annotated_image.save(fp=outputs_path / "test_dataset.png", format="PNG")

        print("\nDATASET PASSED\n")

    # test datamodule
    def test_datamodule(cfg: DictConfig):
        print("TEST DATAMODULE\n")

        # create a DlibDataModule
        datamodule: DlibDataModule = hydra.utils.instantiate(cfg)
        
        # test prepare_data()
        datamodule.prepare_data()

        # test setup()
        datamodule.setup()

        # test batch
        train_loader = datamodule.train_dataloader()
        sample_batch = next(iter(train_loader))
        input = sample_batch[0]
        output = sample_batch[1]
        print(f"Shape of input: {input.shape}")
        print(f"Shape of output: {output.shape}\n")

        # test annotate_tensor()
        annotated_batch: torch.Tensor = DlibTransformDataset.annotate_tensor(input, output)
        torchvision.utils.save_image(annotated_batch, outputs_path / "test_datamodule.png")

        # # test train_dataloader()
        # for batch in tqdm(train_loader):
        #     pass
        # print("train dataloader passed")

        # # test val_dataloader()
        # val_loader = datamodule.val_dataloader()
        # for batch in tqdm(val_loader):
        #     pass
        # print("validation dataloader passed")

        # # test test_dataloader()
        # test_loader = datamodule.test_dataloader()
        # for batch in tqdm(test_loader):
        #     pass
        # print("test dataloader passed")

        print("\nDATAMODULE PASSED\n")

    # def main with hydra
    @hydra.main(version_base="1.3", config_path=config_path, config_name="dlib")
    def main(cfg: DictConfig):
        test_dataset(cfg)
        test_datamodule(cfg)
        print(OmegaConf.to_yaml(cfg))

    # call main
    main()