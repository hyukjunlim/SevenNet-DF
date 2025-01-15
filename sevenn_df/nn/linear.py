from typing import Callable, List, Optional
import math

import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, Linear, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class IrrepsLinear(nn.Module):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_in: str,
        data_key_out: Optional[str] = None,
        **e3nn_linear_params,
    ):
        super().__init__()
        self.key_input = data_key_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out

        self.linear = Linear(irreps_in, irreps_out, **e3nn_linear_params)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.linear(data[self.key_input])
        return data


@compile_mode('script')
class AtomReduce(nn.Module):
    """
    atomic energy -> total energy
    constant is multiplied to data
    """

    def __init__(
        self,
        data_key_in: str,
        data_key_out: str,
        reduce: str = 'sum',
        constant: float = 1.0,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
        mode: str = 'energy',
    ):
        super().__init__()

        self.key_input = data_key_in
        self.key_output = data_key_out
        self.constant = constant
        self.reduce = reduce
        self.mode = mode
        self.key_cell_volume = data_key_cell_volume

        # controlled by the upper most wrapper 'AtomGraphSequential'
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        volume = data[self.key_cell_volume]
        if self._is_batch_data:
            src = data[self.key_input]
            src_shape = src.shape
            size = int(data[KEY.BATCH].max()) + 1
            output = torch.zeros(
                (size, *src_shape[1:]), dtype=src.dtype, device=src.device
            )
            output.scatter_reduce_(0, data[KEY.BATCH].unsqueeze(-1).expand_as(src), src, reduce='sum')
            if self.mode == 'energy':
                output = output.squeeze(1)
            elif self.mode == 'stress':
                output = torch.einsum('bi,bj->bij', output, output).reshape(size, 9)[:, [0, 4, 8, 1, 5, 2]] / volume.view(-1, 1)
            data[self.key_output] = output * self.constant
        else:
            output = torch.sum(data[self.key_input], dim=0)
            if self.mode == 'stress':
                output = torch.einsum('i,j->ij', output, output).reshape(9)[[0, 4, 8, 1, 5, 2]] / volume
            data[self.key_output] = output * self.constant
            
        return data


@compile_mode('script')
class FCN_e3nn(nn.Module):
    """
    wrapper class of e3nn FullyConnectedNet
    """

    def __init__(
        self,
        irreps_in: Irreps,  # confirm it is scalar & input size
        dim_out: int,
        hidden_neurons: List[int],
        activation: Callable,
        data_key_in: str,
        data_key_out: Optional[str] = None,
        **e3nn_params,
    ):
        super().__init__()
        self.key_input = data_key_in
        self.irreps_in = irreps_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out

        for _, irrep in irreps_in:
            assert irrep.is_scalar()
        inp_dim = irreps_in.dim

        self.fcn = FullyConnectedNet(
            [inp_dim] + hidden_neurons + [dim_out],
            activation,
            **e3nn_params,
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.fcn(data[self.key_input])
        return data
