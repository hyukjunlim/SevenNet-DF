from typing import List, Union

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn_df._keys as KEY
from sevenn_df._const import AtomGraphDataType


@compile_mode('script')
class Rescale(nn.Module):
    """
    Scaling and shifting energy (and automatically force and stress)
    """

    def __init__(
        self,
        shift: float,
        scale: dict,
        data_key_in: list = [KEY.SCALED_ATOMIC_ENERGY, KEY.SCALED_ATOMIC_FORCE, KEY.SCALED_ATOMIC_STRESS],
        data_key_out: list = [KEY.ATOMIC_ENERGY, KEY.PRED_FORCE, KEY.ATOMIC_STRESS],
        train_shift_scale: bool = False,
    ):
        assert isinstance(shift, float) and all(isinstance(_, float) for _ in scale.values())
        super().__init__()
        self.shift = nn.Parameter(
            torch.FloatTensor([shift]), requires_grad=train_shift_scale
        )
        self.scale_force = nn.Parameter(
            torch.FloatTensor([scale['force']]), requires_grad=train_shift_scale
        )
        self.scale_stress = nn.Parameter(
            torch.FloatTensor([scale['stress']]), requires_grad=train_shift_scale
        )
        self.key_input_E = data_key_in[0]
        self.key_input_F = data_key_in[1]
        self.key_input_S = data_key_in[2]
        self.key_output_E = data_key_out[0]
        self.key_output_F = data_key_out[1]
        self.key_output_S = data_key_out[2]

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output_E] = data[self.key_input_E] * self.scale_force + self.shift
        data[self.key_output_F] = data[self.key_input_F] * self.scale_force
        data[self.key_output_S] = data[self.key_input_S] * self.scale_stress

        return data


@compile_mode('script')
class SpeciesWiseRescale(nn.Module):
    """
    Scaling and shifting energy (and automatically force and stress)
    Use as it is if given list, expand to list if one of them is float
    If two lists are given and length is not the same, raise error
    """

    def __init__(
        self,
        shift: Union[List[float], float],
        scale: Union[List[float], float],
        data_key_in: str = KEY.SCALED_ATOMIC_ENERGY,
        data_key_out: str = KEY.ATOMIC_ENERGY,
        data_key_indices: str = KEY.ATOM_TYPE,
        train_shift_scale: bool = False,
    ):
        super().__init__()
        assert isinstance(shift, float) or isinstance(shift, list)
        assert isinstance(scale, float) or isinstance(scale, list)

        if (
            isinstance(shift, list)
            and isinstance(scale, list)
            and len(shift) != len(scale)
        ):
            raise ValueError('List length should be same')

        if isinstance(shift, list):
            num_species = len(shift)
        elif isinstance(scale, list):
            num_species = len(scale)
        else:
            raise ValueError('Both shift and scale is not a list')

        shift = [shift] * num_species if isinstance(shift, float) else shift
        scale = [scale] * num_species if isinstance(scale, float) else scale

        self.shift = nn.Parameter(
            torch.FloatTensor(shift), requires_grad=train_shift_scale
        )
        self.scale = nn.Parameter(
            torch.FloatTensor(scale), requires_grad=train_shift_scale
        )
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.key_indices = data_key_indices

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        indices = data[self.key_indices]
        data[self.key_output] = data[self.key_input] * self.scale[indices].view(
            -1, 1
        ) + self.shift[indices].view(-1, 1)

        return data
