from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.transforms as transforms
import torchvision.transforms.functional as ftrasnforms
from signac.contrib.job import Job
from torchvision.datasets.folder import default_loader


class CSVImageDataset(tdata.Dataset):
    def __init__(self, csv_path: Path, partition: str, path_column_name: str, target_column_name: str, partition_column_name: str, subsample: Optional[int]=None) -> None:
        super().__init__()
        self.path_column_name = path_column_name
        self.target_column_name = target_column_name

        df = pd.read_csv(csv_path)
        self.partition_df = df.loc[df[partition_column_name] == partition]

        if subsample is not None:
            classes = self.partition_df[target_column_name].unique()
            indexes = list()
            for c in classes:
                indexes.extend(self.partition_df.loc[self.partition_df[target_column_name] == c].index.tolist()[:subsample])
            self.partition_df = self.partition_df.loc[indexes]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.partition_df.iloc[index][self.path_column_name]
        target = self.partition_df.iloc[index][self.target_column_name]
        sample = default_loader(path)
        return sample, target

    def __len__(self) -> int:
        return len(self.partition_df)

    @property
    def targets(self) -> List[Any]:
        return self.partition_df[self.target_column_name].to_list()


class TransformedDataset(tdata.Dataset):
    def __init__(self, dataset: tdata.Dataset, transform: Optional[Callable]=None,
                 target_transform: Optional[Callable]=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def untransformed_item(self, index):
        x, y = self.dataset[index]
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    @property
    def targets(self) -> List[Any]:
        if self.transform is not None:
            return list(map(self.target_transform, self.dataset.targets))
        else:
            return self.dataset.targets

    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError()
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore


class MaskedDataset(tdata.Dataset):
    def __init__(self, original_dataset: tdata.Dataset, masks: torch.Tensor) -> None:
        super().__init__()
        self.original_dataset = original_dataset
        self.masks = masks

    def __getitem__(self, index):
        x, y = self.original_dataset[index]
        return x * self.masks[index], y

    def __getattr__(self, name):
        return getattr(self.original_dataset, name)

    def __len__(self) -> int:
        return len(self.original_dataset)  # type: ignore


class ExperimentDataModule(pl.LightningDataModule):
    n_classes: int
    image_size: Tuple[int, int]
    
    def __init__(self, job: Job) -> None:
        super().__init__()
        self.job = job
        self.dataset_csv_path = Path(job.sp.dataset_csv_path)
        self.partition = job.sp.partition
        self.batch_size = job.sp.batch_size
        self.train_input_transform = lambda x: x
        self.eval_input_transform = lambda x: x
        self.non_modified_input_transform = lambda x: x
        self.class_mapping = lambda x: x
        self.n_dataloader_workers = job.sp.n_workers

    def setup(self, stage: str, subsample: Optional[int]=None):
        if stage == "fit":
            original_train_dataset = CSVImageDataset(self.dataset_csv_path, 'train', 'path', 'target', f'partition_{self.partition}')
            self.train_dataset = TransformedDataset(original_train_dataset, transform=self.train_input_transform, target_transform=self.class_mapping)

            original_val_dataset = CSVImageDataset(self.dataset_csv_path, 'validation', 'path', 'target', f'partition_{self.partition}')
            self.val_dataset = TransformedDataset(original_val_dataset, transform=self.eval_input_transform, target_transform=self.class_mapping)

        elif (stage == 'test') or (stage == 'predict'):
            original_test_dataset = CSVImageDataset(self.dataset_csv_path, 'test', 'path', 'target', f'partition_{self.partition}', subsample)
            self.test_dataset = TransformedDataset(original_test_dataset, transform=self.eval_input_transform, target_transform=self.class_mapping)
            self.predict_dataset = self.test_dataset

        '''
        elif stage == 'predict':
            original_predict_dataset = CSVImageDataset(self.dataset_csv_path, 'test', 'path', 'target', f'partition_{self.partition}')
            self.predict_dataset = TransformedDataset(original_predict_dataset, transform=self.eval_input_transform, target_transform=self.class_mapping)
            # TODO: Deberían aparecer el mismo número de muestras que para test pero no
            original_predict_dataset = CSVImageDataset(self.dataset_csv_path, 'test', 'path', 'target', f'partition_{self.partition}')
            self.predict_dataset = TransformedDataset(original_predict_dataset, transform=self.eval_input_transform, target_transform=self.class_mapping)
            original_targets = self.predict_dataset.targets
            possible_targets = sorted(list(set(original_targets)))
            index_lists = [[i for i, t in enumerate(original_targets) if t == pt] for pt in possible_targets]
            interweaved_index_lists = [idx for tup in zip(*index_lists) for idx in tup]
            self.predict_dataset = tdata.Subset(self.predict_dataset, interweaved_index_lists)
            self.predict_targets = [self.class_mapping(original_targets[i]) for i in interweaved_index_lists]
            '''

    def train_dataloader(self):
        return tdata.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_dataloader_workers)

    def val_dataloader(self):
        return tdata.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.n_dataloader_workers)

    def test_dataloader(self):
        return tdata.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.n_dataloader_workers)

    def predict_dataloader(self):
        return tdata.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.n_dataloader_workers)

    def masked_datamodule(self, masks: torch.Tensor):
        assert masks.size(0) == len(self.predict_dataset)
        return pl.LightningDataModule.from_datasets(predict_dataset=MaskedDataset(self.predict_dataset, masks),
                                                    batch_size=self.batch_size, num_workers=self.n_dataloader_workers)
    
    def non_modified_dataset(self):
        dataset = CSVImageDataset(self.dataset_csv_path, 'test', 'path', 'target', f'partition_{self.partition}')
        return TransformedDataset(dataset, transform=self.non_modified_input_transform, target_transform=self.class_mapping)


def who_class_mapping(original_label: str) -> int:
    return int(original_label) - 1

def tbs_class_mapping(original_label: str) -> int:
    original_label_int = int(original_label)
    m = {1:0, 2:0, 3:0, 4:1, 5:2, 6:2, 7:3}
    return m[original_label_int]

class SmearDataModule(ExperimentDataModule):
    CLASS_MAPPINGS = {
        'who': (who_class_mapping, 7),
        'tbs': (tbs_class_mapping, 4),
    }

    def __init__(self, job: Job) -> None:
        super().__init__(job)
        self.train_input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 360)),
        ])
        self.eval_input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
        ])
        self.non_modified_input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
        try:
            self.class_mapping, self.n_classes = SmearDataModule.CLASS_MAPPINGS[job.sp.smear_class_mapping]
        except KeyError:
            raise NotImplementedError()
        self.image_size = (224, 224)


class RetinopathyDataModule(ExperimentDataModule):
    def __init__(self, job: Job) -> None:
        super().__init__(job)
        self.train_input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # No need, already resized
            # transforms.Resize((224, 224)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 360)),
        ])
        self.eval_input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
        ])
        self.non_modified_input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
        self.n_classes = 5
        self.image_size = (224, 224)


class AdienceDataModule(ExperimentDataModule):
    def __init__(self, job: Job) -> None:
        super().__init__(job)
        self.train_input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(10)),
        ])
        self.eval_input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
        ])
        self.non_modified_input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
        self.n_classes = 8
        self.image_size = (224, 224)


def cbis_ddsm_class_mapping(original_label: str) -> int:
    original_label_int = int(original_label)
    m = {0:0, 1:0, 2:1, 3:2, 4:3, 5:4}
    return m[original_label_int]

class CBISDDSMDataModule(ExperimentDataModule):
    def __init__(self, job: Job) -> None:
        super().__init__(job)
        self.train_input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.1, contrast=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(180)),
        ])
        self.eval_input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.non_modified_input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
        self.class_mapping = cbis_ddsm_class_mapping
        self.n_classes = 5
        self.image_size = (224, 224)


class CSAWMDataModule(ExperimentDataModule):
    def __init__(self, job: Job) -> None:
        super().__init__(job)
        self.train_input_transform = transforms.Compose([
            transforms.ToTensor(),
            # partial(ftrasnforms.crop, top=200, left=114, height=300, width=300),
            # transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(10)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.eval_input_transform = transforms.Compose([
            transforms.ToTensor(),
            # partial(ftrasnforms.crop, top=200, left=114, height=300, width=300),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Resize((224, 224)),
        ])
        self.non_modified_input_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.n_classes = 8
        self.image_size = (632, 512)


class InvertNormalization(nn.Module):
    def __init__(self, mean: List[float], std: List[float]) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        for c, (m, s) in enumerate(zip(self.mean, self.std)):
            x[:, c] = (x[:, c] * s) + m
        return x