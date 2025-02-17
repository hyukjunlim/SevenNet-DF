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
        data_key_in: list = [KEY.SCALED_ATOMIC_ENERGY, KEY.SCALED_ATOMIC_FORCE, KEY.SCALED_PRED_STRESS],
        data_key_out: list = [KEY.ATOMIC_ENERGY, KEY.PRED_FORCE, KEY.PRED_STRESS],
        train_shift_scale: bool = False,
        mode: str = 'EF',
        data_key_cell_volume: str = KEY.CELL_VOLUME,
    ):
        
        assert all(isinstance(_, float) for _ in [shift['E'], shift['F'], shift['S'], scale['E'], scale['F'], scale['S']])
        super().__init__()
        
        self.scale_energy = nn.Parameter(torch.FloatTensor([scale['E']]), requires_grad=train_shift_scale)
        self.shift_energy = nn.Parameter(torch.FloatTensor([shift['E']]), requires_grad=train_shift_scale)
        
        self.scale_force = nn.Parameter(torch.FloatTensor([scale['F']]), requires_grad=train_shift_scale)
        self.shift_force = nn.Parameter(torch.FloatTensor([shift['F']]), requires_grad=train_shift_scale)
        
        self.scale_stress = nn.Parameter(torch.FloatTensor([scale['S']]), requires_grad=train_shift_scale)
        self.shift_stress = nn.Parameter(torch.FloatTensor([shift['S']]), requires_grad=train_shift_scale)
        
        self.key_input_E, self.key_input_F, self.key_input_S = data_key_in
        self.key_output_E, self.key_output_F, self.key_output_S = data_key_out
        self.mode = mode
        self.key_cell_volume = data_key_cell_volume
        self._is_batch_data = True
        

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        volume = data[self.key_cell_volume].view(-1, 1)
        
        if 'E' in self.mode:
            data[self.key_output_E] = data[self.key_input_E] * self.scale_force + self.shift_energy
        if 'F' in self.mode:
            data[self.key_output_F] = data[self.key_input_F] * self.scale_force
        if 'S' in self.mode:
            x = self.shift_stress
            shift_stress = torch.cat([x, x, x, torch.zeros(3, device=volume.device, dtype=volume.dtype)], dim=-1)
            data[self.key_output_S] = (data[self.key_input_S] * self.scale_stress + shift_stress) # / volume
            if not self._is_batch_data:
                data[self.key_output_S] = data[self.key_output_S].squeeze(0)

        return data


# @compile_mode('script')
# class SpeciesWiseRescale(nn.Module):
#     """
#     Scaling and shifting energy (and automatically force and stress)
#     Use as it is if given list, expand to list if one of them is float
#     If two lists are given and length is not the same, raise error
#     """

#     def __init__(
#         self,
#         shift: Union[List[float], float],
#         scale: Union[List[float], float],
#         data_key_in: str = KEY.SCALED_ATOMIC_ENERGY,
#         data_key_out: str = KEY.ATOMIC_ENERGY,
#         data_key_indices: str = KEY.ATOM_TYPE,
#         train_shift_scale: bool = False,
#     ):
#         super().__init__()
#         assert isinstance(shift, float) or isinstance(shift, list)
#         assert isinstance(scale, float) or isinstance(scale, list)

#         if (
#             isinstance(shift, list)
#             and isinstance(scale, list)
#             and len(shift) != len(scale)
#         ):
#             raise ValueError('List length should be same')

#         if isinstance(shift, list):
#             num_species = len(shift)
#         elif isinstance(scale, list):
#             num_species = len(scale)
#         else:
#             raise ValueError('Both shift and scale is not a list')

#         shift = [shift] * num_species if isinstance(shift, float) else shift
#         scale = [scale] * num_species if isinstance(scale, float) else scale

#         self.shift = nn.Parameter(
#             torch.FloatTensor(shift), requires_grad=train_shift_scale
#         )
#         self.scale = nn.Parameter(
#             torch.FloatTensor(scale), requires_grad=train_shift_scale
#         )
#         self.key_input = data_key_in
#         self.key_output = data_key_out
#         self.key_indices = data_key_indices

#     def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
#         indices = data[self.key_indices]
#         data[self.key_output] = data[self.key_input] * self.scale[indices].view(
#             -1, 1
#         ) + self.shift[indices].view(-1, 1)

#         return data
