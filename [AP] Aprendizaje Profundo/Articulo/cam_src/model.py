import inspect
from collections import namedtuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from captum.attr import LayerAttribution, LayerGradCam, LayerGradCamPlusPlus, LayerScoreCam, LayerDeepLift
from ordinal_gradcam import OrdinalBinomialGradCam, OrdinalOBDPosNegGradCam, OrdinalStepGradCam
from pytorch_grad_cam import ScoreCAM, AblationCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from signac.contrib.job import Job
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.functional.classification.accuracy import multiclass_accuracy

from data import ExperimentDataModule
from IBA.pytorch import IBA

import fix_resnet_relu


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
        self.num_classes = num_classes
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
        return multiclass_accuracy(labels, target, self.num_classes)


def fact(x):
    return torch.exp(torch.lgamma(x+1))


def log_fact(x):
    return torch.lgamma(x+1)
    

class BinomialUnimodal_CE(nn.Module):
    Kt: torch.Tensor
    kk: torch.Tensor

    def __init__(self, K: int) -> None:
        super().__init__()
        self.K = K
        Kt = torch.tensor(self.K, dtype=torch.float)
        self.register_buffer("Kt", Kt)
        kk = torch.arange(self.K, dtype=torch.float)[None]
        self.register_buffer("kk", kk)

    def forward(self, ypred, ytrue):
        return F.nll_loss(ypred, ytrue)


class BinomialUnimodalActivation(nn.Module):
    Kt: torch.Tensor
    kk: torch.Tensor
    
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.K = num_classes
        Kt = torch.tensor(self.K, dtype=torch.float)
        self.register_buffer("Kt", Kt)
        kk = torch.arange(self.K, dtype=torch.float)[None]
        self.register_buffer("kk", kk)

    def forward(self, ypred):
        log_probs = F.logsigmoid(ypred)
        log_inv_probs = F.logsigmoid(-ypred)
        num = log_fact(self.Kt - 1) + self.kk*log_probs + (self.Kt - self.kk - 1)*log_inv_probs
        den = log_fact(self.kk) + log_fact(self.Kt - self.kk - 1)
        return num - den


PredictOutput = namedtuple('PredictOutput', ['raw_output', 'score', 'proba'])

class ExperimentModel(pl.LightningModule):
    def __init__(self, n_classes: int, job: Job):
        super().__init__()
        self.lr = job.sp.learning_rate
        self.classifier_type = job.sp.classifier_type

        base_model_function = eval(job.sp.base_model)
        base_model_weights = eval(job.sp.base_model_weights)
        base_model = base_model_function(weights=base_model_weights)
        base_model_n_outputs: int = inspect.signature(type(base_model)).parameters['num_classes'].default

        if self.classifier_type == 'nominal':
            self.model = nn.Sequential(base_model, nn.Linear(base_model_n_outputs, n_classes))
            self.loss_function = nn.functional.cross_entropy
            self.accuracy = MulticlassAccuracy(n_classes)
        elif self.classifier_type == 'ordinal_ecoc':
            self.model = nn.Sequential(base_model, OBDOutput(base_model_n_outputs, n_classes))
            self.loss_function = ECOCLoss(n_classes)
            self.ecoc_transformer = ECOCTransformer(n_classes)
            self.accuracy = self.ecoc_transformer.accuracy
        elif self.classifier_type == 'ordinal_unimodal_binomial_ce':
            self.model = nn.Sequential(base_model, nn.Linear(base_model_n_outputs, 1), BinomialUnimodalActivation(n_classes))
            self.loss_function = BinomialUnimodal_CE(n_classes)
            self.accuracy = MulticlassAccuracy(n_classes)
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.model(x)

    def score(self, x):
        x = self.forward(x)

        if self.classifier_type == 'ordinal_ecoc':
            x = self.ecoc_transformer.scores(x)
        
        return x

    def training_step(self, batch, batch_idx):
        metrics = self._common_step(batch, batch_idx, 'train')
        return metrics['train_loss']

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
        self.log_dict(metrics, on_epoch=True, on_step=False)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self.predict_from_inputs(x)

    def predict_from_inputs(self, x):
        raw_output = self(x)

        if self.classifier_type == 'nominal':
            score = raw_output
            probas = torch.softmax(score, dim=1)
        elif self.classifier_type == 'ordinal_ecoc':
            score = self.ecoc_transformer.scores(raw_output)
            probas = torch.softmax(score, dim=1)
        elif self.classifier_type == 'ordinal_unimodal_binomial_ce':
            score = raw_output
            probas = torch.exp(score)
        else:
            raise NotImplementedError()
            
        return PredictOutput(raw_output, score, probas)

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.non_regularized_parameters()},
            {'params': self.regularized_parameters(), 'weight_decay': job.sp.l2_penalty},
        ], lr=job.sp.learning_rate)


class OBDWrapper(nn.Module):
    def __init__(self, obd_model: ExperimentModel) -> None:
        super().__init__()
        self.obd_model = obd_model

    def forward(self, x):
        return self.obd_model.predict_from_inputs(x).score

class DeepLiftWrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.predict_from_inputs(x).score

CAPTUM_CAM_METHODS = [
    'gradcam',
    'gradcam++',
    'scorecam',
    'ordinal_gradcam_binomial',
    'ordinal_gradcam_obd_posneg',
    'gradcam_score',
    'gradcam++_score',
    'ordinal_gradcam_binomial_score',
    'ordinal_gradcam_step',
]

class AttributionModel(pl.LightningModule):
    def __init__(self, job: Job, model: ExperimentModel, datamodule: ExperimentDataModule) -> None:
        super().__init__()

        target_layer = eval(job.sp.cam_layer)

        if job.sp.explanation_method == 'gradcam':
            self.gradient_method = LayerGradCam(lambda x: model.predict_from_inputs(x).proba, target_layer)
        elif job.sp.explanation_method == 'gradcam++':
            self.gradient_method = LayerGradCamPlusPlus(lambda x: model.predict_from_inputs(x).proba, target_layer)
        elif job.sp.explanation_method == 'gradcam_score':
            self.gradient_method = LayerGradCam(lambda x: model.predict_from_inputs(x).score, target_layer)
        elif job.sp.explanation_method == 'gradcam++_score':
            self.gradient_method = LayerGradCamPlusPlus(lambda x: model.predict_from_inputs(x).score, target_layer)
        elif job.sp.explanation_method == 'scorecam':
            self.gradient_method = LayerScoreCam(lambda x: model.predict_from_inputs(x).score, target_layer)
        elif job.sp.explanation_method == 'deeplift':
            self.gradient_method = LayerDeepLift(DeepLiftWrapper(model), target_layer)
        elif job.sp.explanation_method == 'ordinal_gradcam_binomial':
            self.gradient_method = OrdinalBinomialGradCam(lambda x: model.predict_from_inputs(x).proba, target_layer, datamodule.n_classes)
        elif job.sp.explanation_method == 'ordinal_gradcam_binomial_score':
            self.gradient_method = OrdinalBinomialGradCam(lambda x: model.predict_from_inputs(x).score, target_layer, datamodule.n_classes)
        elif job.sp.explanation_method == 'ordinal_gradcam_obd_posneg':
            self.gradient_method = OrdinalOBDPosNegGradCam(lambda x: model.predict_from_inputs(x).raw_output, target_layer, datamodule.n_classes)
        elif job.sp.explanation_method == 'ordinal_gradcam_step':
            self.gradient_method = OrdinalStepGradCam(lambda x: model.predict_from_inputs(x).score, target_layer, datamodule.n_classes, job.sp.step_exponent)
        elif job.sp.explanation_method in ('iba', 'iba_ce'):
            self.iba = IBA(target_layer)
            datamodule.setup('fit')
            train_dataloader = datamodule.train_dataloader()
            self.iba.estimate(model, train_dataloader, n_samples=5000, progbar=False)
        else:
            raise NotImplementedError
        self.model = model
        self.job = job
        self.datamodule = datamodule
        self.n_classes = datamodule.n_classes

    def predict_step(self, batch, batch_idx):
        x, y = batch
        if self.job.sp.explanation_method in CAPTUM_CAM_METHODS:
            attribution = self.gradient_method.attribute(x, target=y, relu_attributions=True).detach()
        elif self.job.sp.explanation_method == 'deeplift':
            attribution = self.gradient_method.attribute(x, target=y).detach()
            attribution = attribution.mean(dim=1, keepdim=True)
            attribution = LayerAttribution.interpolate(attribution, self.datamodule.image_size, 'bilinear')
            batch_size, n_features = attribution.size()[:2]
            maxs = attribution.view(batch_size, n_features, -1).max(dim=-1)[0]
            mins = attribution.view(batch_size, n_features, -1).min(dim=-1)[0]
            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            attribution = (attribution - mins) / torch.clamp(maxs - mins, min=1e-7)
            return attribution
        elif self.job.sp.explanation_method == 'iba':
            attributions_list = list()
            with torch.enable_grad():
                for i in range(x.size(0)):
                    loss_closure = lambda z: self.model.loss_function(self.model(z), y[i].repeat(z.size(0))).mean() 
                    # loss_closure = lambda z: print(z.size())
                    attributions_list.append(torch.tensor(self.iba.analyze(x[i:i+1,:], loss_closure, beta=10)[None, None, :, :]))
            attribution = torch.cat(attributions_list)
        elif self.job.sp.explanation_method == 'iba_ce':
            attributions_list = list()
            with torch.enable_grad():
                for i in range(x.size(0)):
                    if self.model.classifier_type == 'ordinal_unimodal_binomial_ce':
                        loss_closure = lambda z: F.nll_loss(self.model.score(z), y[i].repeat(z.size(0))).mean()
                    else:
                        loss_closure = lambda z: F.cross_entropy(self.model.score(z), y[i].repeat(z.size(0))).mean()
                    # loss_closure = lambda z: print(z.size())
                    attributions_list.append(torch.tensor(self.iba.analyze(x[i:i+1,:], loss_closure, beta=10)[None, None, :, :]))
            attribution = torch.cat(attributions_list)
        else:
            raise NotImplementedError

        attribution = LayerAttribution.interpolate(attribution, self.datamodule.image_size, 'bilinear')
        maxs = attribution.flatten(start_dim=1).max(dim=1).values
        maxs[maxs == 0] = 1.0
        attribution /= maxs[:, None, None, None]
        return attribution

    '''
    def on_predict_epoch_start(self) -> None:
        self.classes_to_sample = set(range(self.n_classes))

    def on_predict_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        if not self.classes_to_sample:
            return
        
        if self.logger is None:
            return

        x, y = batch
        batch_classes = {e.item() for e in y}
        while batch_classes & self.classes_to_sample:
            for i, ctensor in enumerate(y):
                c = ctensor.item()
                if c in self.classes_to_sample:
                    image = x[i:i+1]
                    if self.log_image_transform is not None:
                        image = self.log_image_transform(image)
                    image = image[0]
                    activation = LayerAttribution.interpolate(outputs[i:i+1], (224, 224), 'bilinear')[0, 0]
                    activation = activation / activation.max()
                    self.logger.experiment.add_images(f'2sample of class {c}', image, dataformats='CHW')
                    self.logger.experiment.add_images(f'2activation of sample of class {c}', activation, dataformats='HW')
                    self.classes_to_sample.remove(c)
                    break
    '''
