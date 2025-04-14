from captum.attr import LayerWeightedGradCam as _LayerWeightedGradCam
from captum._utils.typing import TargetType
import torch
from torch import Tensor
import torch.nn as nn
from typing import Union, List, Callable, Tuple, Any
from beta_quartile import bisect3 as estimate_beta_params
import numpy as np
import scipy.stats as st


class LayerWeightedGradCam(_LayerWeightedGradCam):
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        weights = self.weights.to(inputs.device)[target]
        return super().attribute(inputs, weights, additional_forward_args, attribute_to_layer_input, relu_attributions)


class OrdinalBetaGradCam(LayerWeightedGradCam):
    def __init__(
        self,
        forward_func: Callable,
        layer: nn.Module,
        n_outputs: int,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        super().__init__(forward_func, layer, device_ids)

        if n_outputs == 3:
            weights = np.array([
                [0.6, 0.3, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6],
            ], dtype=np.float32)
        else:
            weights = np.zeros((n_outputs, n_outputs), dtype=np.float32)
            for c in range(n_outputs):
                if c == 0:
                    xql = 1 / n_outputs
                    xqu = 2 / n_outputs
                    ql = 0.6
                    qu = 0.9
                elif c == (n_outputs - 1):
                    xql = (n_outputs - 2) / n_outputs
                    xqu = (n_outputs - 1) / n_outputs
                    ql = 0.1
                    qu = 0.4
                else:
                    xql = c / n_outputs
                    xqu = (c+1) / n_outputs
                    ql = 0.25
                    qu = 0.85
                a, b = estimate_beta_params(xql, xqu, ql, qu)
                distribution = st.beta(a, b)
                for i in range(n_outputs):
                    weights[c, i] = distribution.cdf((i+1) / n_outputs) - distribution.cdf(i / n_outputs)

        self.weights = torch.tensor(weights)


class OrdinalStepGradCam(LayerWeightedGradCam):
    def __init__(
        self,
        forward_func: Callable,
        layer: nn.Module,
        n_outputs: int,
        p: int = 1,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        super().__init__(forward_func, layer, device_ids)

        weights = np.zeros((n_outputs, n_outputs))
        weights += np.arange(n_outputs)
        weights = (n_outputs - np.abs(weights - weights.T))**p
        weights = weights / weights.sum(axis=0)[:, None]

        self.weights = torch.tensor(weights).float()


class OrdinalBinomialGradCam(LayerWeightedGradCam):
    def __init__(
        self,
        forward_func: Callable,
        layer: nn.Module,
        n_outputs: int,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        super().__init__(forward_func, layer, device_ids)

        weights = np.zeros((n_outputs, n_outputs), dtype=np.float32)
        
        xs = np.arange(n_outputs)
        for c in range(n_outputs):
            dist = st.binom(n_outputs - 1, (c+0.5) / n_outputs)
            weights[c, :] = dist.pmf(xs)

        self.weights = torch.tensor(weights)


class OrdinalOBDPosNegGradCam(LayerWeightedGradCam):
    def __init__(
        self,
        forward_func: Callable,
        layer: nn.Module,
        n_classes: int,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        super().__init__(forward_func, layer, device_ids)

        n_outputs = n_classes - 1
        target_class = np.ones((n_outputs+1, n_outputs), dtype=np.float32)
        target_class[np.triu_indices(n_outputs+1, 0, n_outputs)] = -1.0
        self.weights = torch.tensor(target_class, dtype=torch.float32)