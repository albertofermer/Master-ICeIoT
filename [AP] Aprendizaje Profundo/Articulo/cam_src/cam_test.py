import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchmetrics
import torchvision.transforms as transforms
from captum.attr import LayerGradCam, LayerAttribution
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from spacecutter.models import OrdinalLogisticModel
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
import matplotlib.pyplot as plt
from torchinfo import summary
from PIL import Image


def _replace_relu(module: nn.Module) -> None:
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) is nn.ReLU or type(mod) is nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value


def targets_and_scores(model: nn.Module, dataloader: tdata.DataLoader):
    pass


_QWK_LOSS_EPSILON = 1e-9

class QWKLoss(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # Create cost matrix and register as buffer
        cost_matrix = torch.tensor(np.reshape(np.tile(range(num_classes), num_classes), (num_classes, num_classes))).float()
        cost_matrix = (cost_matrix - torch.transpose(cost_matrix, 0, 1)) ** 2
        
        self.register_buffer("cost_matrix", cost_matrix)

        self.num_classes = num_classes

    def forward(self, output, target):
        output = nn.functional.softmax(output, dim=1)
        
        costs = self.cost_matrix[target]  # type: ignore

        numerator = costs * output
        numerator = torch.sum(numerator)

        sum_prob = torch.sum(output, dim=0)
        target_prob = nn.functional.one_hot(target, self.num_classes)
        n = torch.sum(target_prob, dim=0)

        denominator = ((self.cost_matrix * sum_prob[None, :]).sum(dim=1) * (n/n.sum())).sum()
        denominator = denominator + _QWK_LOSS_EPSILON

        return torch.log(numerator / denominator)


class OBDOutput(nn.Module):
    classifiers: nn.ModuleList
    
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.classifiers = nn.ModuleList(
            [nn.Linear(input_size, 1) for _ in range(num_classes-1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [classifier(x) for classifier in self.classifiers]
        x = torch.cat(xs, dim=1)
        x = torch.sigmoid(x)
        return x


class ECOCLoss(nn.Module):
    target_class: torch.Tensor

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        target_class = np.ones((num_classes, num_classes-1), dtype=np.float32)
        target_class[np.triu_indices(num_classes, 0, num_classes-1)] = 0.0
        target_class = torch.tensor(target_class, dtype=torch.float32)
        self.register_buffer("target_class", target_class)
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, output, target):
        target_vector = self.target_class[target].to(output.device)
        return self.mse(output, target_vector)


class ECOCTransformer(nn.Module):
    target_class: torch.Tensor
    
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        target_class = np.ones((num_classes, num_classes-1), dtype=np.float32)
        target_class[np.triu_indices(num_classes, 0, num_classes-1)] = 0.0
        target_class = torch.tensor(target_class, dtype=torch.float32)
        self.register_buffer("target_class", target_class)

    def scores(self, probas):
        return -torch.cdist(probas, self.target_class.to(probas.device))

    def labels(self, probas):
        scores = self.scores(probas)
        return scores.argmax(dim=1)
    
    def accuracy(self, probas, target):
        labels = self.labels(probas)
        return torchmetrics.functional.accuracy(labels, target)


class ResNet(pl.LightningModule):
    def __init__(self, n_classes: int, classifier_type: str, lr: float=1e-3):
        super().__init__()
        self.lr = lr
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.classifier_type = classifier_type

        if classifier_type == 'nominal':
            self.model = nn.Sequential(base_model, nn.Linear(1000, n_classes))
            self.loss_function = nn.functional.cross_entropy
            self.accuracy = torchmetrics.functional.accuracy
        elif classifier_type == 'ordinal_qwk':
            self.model = nn.Sequential(base_model, nn.Linear(1000, n_classes))
            self.loss_function = QWKLoss(n_classes)
            self.accuracy = torchmetrics.functional.accuracy
        elif classifier_type == 'ordinal_clm':
            self.model = OrdinalLogisticModel(nn.Sequential(base_model, nn.Linear(1000, 1)), n_classes)
            self.loss_function = nn.functional.cross_entropy
            self.accuracy = torchmetrics.functional.accuracy
        elif classifier_type == 'ordinal_obd':
            self.model = nn.Sequential(base_model, OBDOutput(1000, n_classes))
            self.loss_function = ECOCLoss(n_classes)
            self.ecoc_transformer = ECOCTransformer(n_classes)
            self.accuracy = self.ecoc_transformer.accuracy
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        metrics = self._common_step(batch, batch_idx, 'train')
        return metrics['train_loss']

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.classifier_type == 'ordinal_clm':
            cutpoints = self.model.link.cutpoints.data
            for i in range(cutpoints.shape[0] - 1):
                cutpoints[i].clamp_(-1e6, cutpoints[i + 1])

    def validation_step(self, batch, batch_idx):
        metrics = self._common_step(batch, batch_idx, 'val')
        return metrics['val_loss']

    def test_step(self, batch, batch_idx):
        metrics = self._common_step(batch, batch_idx, 'test')
        return metrics

    def _common_step(self, batch, batch_idx, phase: str):
        x, y = batch
        yhat = self.forward(x)
        loss = self.loss_function(yhat, y)
        accuracy = self.accuracy(yhat, y)
        metrics = {f'{phase}_loss': loss, f'{phase}_acc': accuracy}
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self.predict_from_inputs(x)

    def predict_from_inputs(self, x):
        if self.classifier_type == 'ordinal_obd':
            threshold_probas = self(x)
            return self.ecoc_transformer.scores(threshold_probas)
        else:
            return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class TransformedSubset(tdata.Subset):
    def __init__(self, *args, **kwargs):
        self.transform = kwargs.pop('transform', None)
        self.target_transform = kwargs.pop('target_transform', None)
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


class MaskedDataset(tdata.Dataset):
    def __init__(self, original_dataset: tdata.Dataset, masks: torch.Tensor) -> None:
        super().__init__()
        self.original_dataset = original_dataset
        self.masks = masks

    def __getitem__(self, index):
        x, y = self.original_dataset[index]
        return x * self.masks[index], y

    def __len__(self):
        return len(self.original_dataset)


class SmearDataModule(pl.LightningDataModule):
    @staticmethod
    def who_class_mapping(original_label: str) -> int:
        return int(original_label)

    @staticmethod
    def tbs_class_mapping(original_label: str) -> int:
        original_label_int = int(original_label)
        m = {0:0, 1:0, 2:0, 3:1, 4:2, 5:2, 6:3}
        return m[original_label_int]

    CLASS_MAPPINGS = {
        'who': (who_class_mapping, 7),
        'tbs': (tbs_class_mapping, 4),
    }

    def __init__(self, root_dir: Path, holdout_id: int, class_mapping: Literal['who', 'tbs'], batch_size: int, validation_size: float, n_dataloader_workers: int=4) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.holdout_id = holdout_id
        self.dataset_root = self.root_dir / str(holdout_id)
        self.batch_size = batch_size
        self.validation_size = validation_size
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
        self.n_dataloader_workers = n_dataloader_workers
        try:
            self.class_mapping, self.n_classes = SmearDataModule.CLASS_MAPPINGS[class_mapping]
        except KeyError:
            raise NotImplementedError()

    def setup(self, stage: str):
        if stage == "fit":
            self.trainval_dataset = ImageFolder(str(self.dataset_root / 'train'))
            train_idxs, val_idxs = train_test_split(np.arange(len(self.trainval_dataset)), test_size=self.validation_size, shuffle=True, stratify=self.trainval_dataset.targets)
            self.train_dataset = TransformedSubset(self.trainval_dataset, train_idxs, transform=self.train_input_transform, target_transform=self.class_mapping)
            self.val_dataset = TransformedSubset(self.trainval_dataset, val_idxs, transform=self.eval_input_transform, target_transform=self.class_mapping)

        elif stage == 'test':
            self.test_dataset = ImageFolder(str(self.dataset_root / 'test'), transform=self.eval_input_transform, target_transform=self.class_mapping)

        elif stage == 'predict':
            self.predict_dataset = ImageFolder(str(self.dataset_root / 'test'), transform=self.eval_input_transform, target_transform=self.class_mapping)
            original_targets = self.predict_dataset.targets
            possible_targets = sorted(list(set(original_targets)))
            index_lists = [[i for i, t in enumerate(original_targets) if t == pt] for pt in possible_targets]
            interweaved_index_lists = [idx for tup in zip(*index_lists) for idx in tup]
            self.predict_dataset = tdata.Subset(self.predict_dataset, interweaved_index_lists)
            self.predict_targets = [self.class_mapping(original_targets[i]) for i in interweaved_index_lists]

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--classifier-type', default='nominal', choices=['nominal', 'ordinal_qwk', 'ordinal_clm', 'ordinal_obd'], dest='classifier_type')
    parser.add_argument('--class-mapping', default='tbs', choices=['tbs', 'who'], dest='class_mapping')
    parser.add_argument('--validation-size', default=0.23, type=float, dest='validation_size')
    parser.add_argument('--checkpoint-path', type=Path, dest='checkpoint_path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    
    trainer_callbacks = []
    datamodule = SmearDataModule(Path('data/smear/30holdout'), class_mapping=args.class_mapping, holdout_id=0, batch_size=742, validation_size=args.validation_size)
    checkpoint_dirpath = './checkpoints'
    checkpoint_filename = ' '.join(f'{k}={v}' for k, v in vars(args).items() if k not in ['checkpoint_path'])
    checkpoint_path = Path(checkpoint_dirpath) / f'{checkpoint_filename}.ckpt'

    if checkpoint_path.is_file():
        model = ResNet.load_from_checkpoint(str(checkpoint_path), n_classes=datamodule.n_classes, classifier_type=args.classifier_type)
    else:
        model = ResNet(n_classes=datamodule.n_classes, classifier_type=args.classifier_type, lr=args.lr)
        trainer_callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', patience=args.patience),
            ModelCheckpoint(checkpoint_dirpath, checkpoint_filename, monitor='val_loss', mode='min'),
        ]
        # trainer.tune(model, datamodule)
        # print(f'Tuned batch size: {datamodule.batch_size}')

    _replace_relu(model)
    # print(summary(model, (742, 3, 224, 224)))

    trainer = pl.Trainer(deterministic=True, max_epochs=200, accelerator='cpu', devices=1, auto_scale_batch_size=False,
                         log_every_n_steps=1, callbacks=trainer_callbacks)

    if not checkpoint_path.is_file():
        trainer.fit(model=model, datamodule=datamodule)
    
    trainer.test(model=model, datamodule=datamodule)
    predictions = torch.cat(trainer.predict(model=model, datamodule=datamodule))
    target_scores_before = torch.squeeze(torch.gather(predictions, 1, torch.tensor(datamodule.predict_targets).long().view(-1, 1)), 1).detach()
    del predictions

    predict_dl = datamodule.predict_dataloader()
    activation_map_batches = list()
    gradcam = LayerGradCam(model.predict_from_inputs, model.model[0].layer4[1].conv1)
    for batch in predict_dl:
        x, y = batch
        attributions = gradcam.attribute(x, target=y, relu_attributions=True)
        activation_map_batches.append(LayerAttribution.interpolate(attributions, (224, 224), 'bilinear'))
    activation_maps = torch.cat(activation_map_batches).detach()
    activation_maps = activation_maps / activation_maps.max()

    occluded_dm = datamodule.masked_datamodule(1.0 - activation_maps)
    
    predictions = torch.cat(trainer.predict(model=model, datamodule=occluded_dm))
    target_scores_after_occlusion = torch.squeeze(torch.gather(predictions, 1, torch.tensor(datamodule.predict_targets).long().view(-1, 1)), 1).detach()

    average_drop = (torch.clamp(target_scores_before - target_scores_after_occlusion, min=0) / target_scores_before).sum() / target_scores_before.size(0)
    print(average_drop)

    sharpened_dm = datamodule.masked_datamodule(activation_maps)

    predictions = torch.cat(trainer.predict(model=model, datamodule=sharpened_dm))
    target_scores_after_sharpening = torch.squeeze(torch.gather(predictions, 1, torch.tensor(datamodule.predict_targets).long().view(-1, 1)), 1).detach()

    confidence_increase = (target_scores_after_sharpening > target_scores_before).sum() / target_scores_before.size(0)
    print(confidence_increase)



if __name__ == '__main__':
    main()
