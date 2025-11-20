import typing as tp
import torch
from torch import nn
from abc import abstractmethod


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: torch.Tensor) -> tp.List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> tp.Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: tp.Any, **kwargs) -> torch.Tensor:
        pass
