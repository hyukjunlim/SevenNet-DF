import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn_df._keys as KEY
from sevenn_df._const import AtomGraphDataType

from .util import _broadcast
import math


@compile_mode('script')
class ForceOutput(nn.Module):
    """
    works when pos.requires_grad_ is True
    """

    def __init__(
        self,
        data_key_pos: str = KEY.POS,
        data_key_energy: str = KEY.PRED_ENERGY,
        data_key_force: str = KEY.PRED_FORCE,
    ):
        super().__init__()
        self.key_pos = data_key_pos
        self.key_energy = data_key_energy
        self.key_force = data_key_force

    def get_grad_key(self):
        return self.key_pos

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        pos_tensor = [data[self.key_pos]]
        energy = [(data[self.key_energy]).sum()]

        grad = torch.autograd.grad(
            energy,
            pos_tensor,
            create_graph=self.training,
        )[0]

        # For torchscript
        if grad is not None:
            data[self.key_force] = torch.neg(grad)
        return data


@compile_mode('script')
class ForceStressOutput(nn.Module):
    """
    Compute stress and force from positions.
    Used in serial torchscipt models
    """
    def __init__(
        self,
        data_key_pos: str = KEY.POS,
        data_key_energy: str = KEY.PRED_ENERGY,
        data_key_force: str = KEY.PRED_FORCE,
        data_key_stress: str = KEY.PRED_STRESS,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
    ):

        super().__init__()
        self.key_pos = data_key_pos
        self.key_energy = data_key_energy
        self.key_force = data_key_force
        self.key_stress = data_key_stress
        self.key_cell_volume = data_key_cell_volume
        self._is_batch_data = True

    def get_grad_key(self):
        return self.key_pos

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        pos_tensor = data[self.key_pos]
        energy = [(data[self.key_energy]).sum()]

        grad = torch.autograd.grad(
            energy,
            [pos_tensor, data['_strain']],
            create_graph=self.training,
        )

        # make grad is not Optional[Tensor]
        fgrad = grad[0]
        if fgrad is not None:
            data[self.key_force] = torch.neg(fgrad)

        sgrad = grad[1]
        volume = data[self.key_cell_volume]
        if sgrad is not None:
            if self._is_batch_data:
                stress = sgrad / volume.view(-1, 1, 1)
                stress = torch.neg(stress)
                virial_stress = torch.vstack((
                    stress[:, 0, 0],
                    stress[:, 1, 1],
                    stress[:, 2, 2],
                    stress[:, 0, 1],
                    stress[:, 1, 2],
                    stress[:, 0, 2],
                ))
                data[self.key_stress] = virial_stress.transpose(0, 1)
            else:
                stress = sgrad / volume
                stress = torch.neg(stress)
                virial_stress = torch.stack((
                    stress[0, 0],
                    stress[1, 1],
                    stress[2, 2],
                    stress[0, 1],
                    stress[1, 2],
                    stress[0, 2],
                ))
                data[self.key_stress] = virial_stress

        return data


@compile_mode('script')
class ForceStressOutputFromEdge(nn.Module):
    """
    Compute stress and force from edge.
    Used in parallel torchscipt models, and training
    """
    def __init__(
        self,
        data_key_edge: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_energy: str = KEY.PRED_ENERGY,
        data_key_force: str = KEY.PRED_FORCE,
        data_key_stress: str = KEY.PRED_STRESS,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
    ):

        super().__init__()
        self.key_edge = data_key_edge
        self.key_edge_idx = data_key_edge_idx
        self.key_energy = data_key_energy
        self.key_force = data_key_force
        self.key_stress = data_key_stress
        self.key_cell_volume = data_key_cell_volume
        self._is_batch_data = True

    def get_grad_key(self):
        return self.key_edge

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        tot_num = torch.sum(data[KEY.NUM_ATOMS])  # ? item?
        rij = data[self.key_edge]
        energy = [(data[self.key_energy]).sum()]
        edge_idx = data[self.key_edge_idx]

        grad = torch.autograd.grad(
            energy,
            [rij],
            create_graph=self.training,
            allow_unused=True
        )

        # make grad is not Optional[Tensor]
        fij = grad[0]

        if fij is not None:
            # compute force
            pf = torch.zeros(tot_num, 3, dtype=fij.dtype, device=fij.device)
            nf = torch.zeros(tot_num, 3, dtype=fij.dtype, device=fij.device)
            _edge_src = _broadcast(edge_idx[0], fij, 0)
            _edge_dst = _broadcast(edge_idx[1], fij, 0)
            pf.scatter_reduce_(0, _edge_src, fij, reduce='sum')
            nf.scatter_reduce_(0, _edge_dst, fij, reduce='sum')
            data[self.key_force] = pf - nf

            # compute virial
            diag = rij * fij
            s12 = rij[..., 0] * fij[..., 1]
            s23 = rij[..., 1] * fij[..., 2]
            s31 = rij[..., 2] * fij[..., 0]
            # cat last dimension
            _virial = torch.cat([
                diag,
                s12.unsqueeze(-1),
                s23.unsqueeze(-1),
                s31.unsqueeze(-1)
            ], dim=-1)

            _s = torch.zeros(tot_num, 6, dtype=fij.dtype, device=fij.device)
            _edge_dst6 = _broadcast(edge_idx[1], _virial, 0)
            _s.scatter_reduce_(0, _edge_dst6, _virial, reduce='sum')

            if self._is_batch_data:
                batch = data[KEY.BATCH]  # for deploy, must be defined first
                nbatch = int(batch.max().cpu().item()) + 1
                sout = torch.zeros(
                    (nbatch, 6), dtype=_virial.dtype, device=_virial.device
                )
                _batch = _broadcast(batch, _s, 0)
                sout.scatter_reduce_(0, _batch, _s, reduce='sum')
            else:
                sout = torch.sum(_s, dim=0)

            data[self.key_stress] =\
                torch.neg(sout) / data[self.key_cell_volume].unsqueeze(-1)

        return data
    
@compile_mode('script')
class DirectEnergyStressOutput(nn.Module):
    """
    Process energy and stress.
    Used in serial torchscipt models
    """
    def __init__(
        self,
        data_key_edge: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_energy: str = KEY.PRED_ENERGY,
        data_key_force: str = KEY.PRED_FORCE,
        data_key_stress: str = KEY.SCALED_PRED_STRESS,
        data_key_atomic_stress_iso: str = KEY.SCALED_ATOMIC_STRESS_ISO,
        data_key_atomic_stress_aniso: str = KEY.SCALED_ATOMIC_STRESS_ANISO,
        data_key_force_drv: str = KEY.PRED_FORCE_DRV,
        data_key_stress_drv: str = KEY.PRED_STRESS_DRV,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
    ):

        super().__init__()
        self.key_edge = data_key_edge
        self.key_edge_idx = data_key_edge_idx
        self.key_energy = data_key_energy
        self.key_force = data_key_force
        self.key_stress = data_key_stress
        self.key_atomic_stress_iso = data_key_atomic_stress_iso
        self.key_atomic_stress_aniso = data_key_atomic_stress_aniso
        self.key_force_drv = data_key_force_drv
        self.key_stress_drv = data_key_stress_drv
        self.key_cell_volume = data_key_cell_volume
        self._is_batch_data = True
        
    def get_grad_key(self):
        return self.key_edge
    
    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        volume = data[self.key_cell_volume]
        data[self.key_energy] = data[self.key_energy].squeeze(-1)
        iso = data[self.key_atomic_stress_iso]
        aniso = data[self.key_atomic_stress_aniso]
        A = torch.tensor([
            [0.0, 0.0, -1/math.sqrt(6), 0.0, -1/math.sqrt(2)],
            [0.0, 0.0,  math.sqrt(2)/math.sqrt(3), 0.0,  0.0],
            [0.0, 0.0, -1/math.sqrt(6), 0.0, 1/math.sqrt(2)],
            [0.0, 1/math.sqrt(2), 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1/math.sqrt(2), 0.0],
            [1/math.sqrt(2), 0.0, 0.0, 0.0, 0.0]
        ], dtype=aniso.dtype, device=aniso.device)
        tot_num = torch.sum(data[KEY.NUM_ATOMS])  # ? item?
        
        if self._is_batch_data:
            ### Derivating method
            rij = data[self.key_edge]
            energy = [(data[self.key_energy]).sum()]
            edge_idx = data[self.key_edge_idx]

            grad = torch.autograd.grad(
                energy,
                [rij],
                create_graph=self.training,
                allow_unused=True
            )

            # make grad is not Optional[Tensor]
            fij = grad[0]

            if fij is not None:
                # compute force
                pf = torch.zeros(tot_num, 3, dtype=fij.dtype, device=fij.device)
                nf = torch.zeros(tot_num, 3, dtype=fij.dtype, device=fij.device)
                _edge_src = _broadcast(edge_idx[0], fij, 0)
                _edge_dst = _broadcast(edge_idx[1], fij, 0)
                pf.scatter_reduce_(0, _edge_src, fij, reduce='sum')
                nf.scatter_reduce_(0, _edge_dst, fij, reduce='sum')
                data[self.key_force_drv] = pf - nf

                # compute virial
                diag = rij * fij
                s12 = rij[..., 0] * fij[..., 1]
                s23 = rij[..., 1] * fij[..., 2]
                s31 = rij[..., 2] * fij[..., 0]
                # cat last dimension
                _virial = torch.cat([
                    diag,
                    s12.unsqueeze(-1),
                    s23.unsqueeze(-1),
                    s31.unsqueeze(-1)
                ], dim=-1)

                _s = torch.zeros(tot_num, 6, dtype=fij.dtype, device=fij.device)
                _edge_dst6 = _broadcast(edge_idx[1], _virial, 0)
                _s.scatter_reduce_(0, _edge_dst6, _virial, reduce='sum')

                batch = data[KEY.BATCH]  # for deploy, must be defined first
                nbatch = int(batch.max().cpu().item()) + 1
                sout = torch.zeros(
                    (nbatch, 6), dtype=_virial.dtype, device=_virial.device
                )
                _batch = _broadcast(batch, _s, 0)
                sout.scatter_reduce_(0, _batch, _s, reduce='sum')
                data[self.key_stress_drv] =\
                    torch.neg(sout) / volume.unsqueeze(-1)
                    
            ### Direct method
            size = volume.shape[0]
            src_andev = aniso @ A.T
            src_dev = torch.cat([iso, iso, iso, torch.zeros(tot_num, 3, dtype=aniso.dtype, device=aniso.device)], dim=-1)
            assert src_andev.shape == src_dev.shape
            
            src_shape = src_andev.shape
            size = int(data[KEY.BATCH].max()) + 1
            output_andev = torch.zeros(
                (size, *src_shape[1:]), dtype=src_andev.dtype, device=src_andev.device
            )
            output_andev.scatter_reduce_(0, data[KEY.BATCH].unsqueeze(-1).expand_as(src_andev), src_andev, reduce='sum')
            output_dev = torch.zeros(
                (size, *src_shape[1:]), dtype=src_dev.dtype, device=src_dev.device
            )
            output_dev.scatter_reduce_(0, data[KEY.BATCH].unsqueeze(-1).expand_as(src_dev), src_dev, reduce='sum')
            
            # output
            data[self.key_stress] = (output_andev, output_dev)
        else:
            ### Direct method (only)
            output_andev = aniso @ A.T
            output_dev = torch.cat([iso, iso, iso, torch.zeros(tot_num, 3, dtype=aniso.dtype, device=aniso.device)], dim=-1)
            
            # output
            data[self.key_stress] = (torch.sum(output_andev, dim=0), torch.sum(output_dev, dim=0))
        
        return data
      
@compile_mode('script')
class SplitEFS(nn.Module):
    """
    Split energy, force and stress.
    Used in serial torchscipt models
    """
    def __init__(
        self,
        data_key_feature: str = KEY.NODE_FEATURE,
        data_key_energy: str = KEY.SCALED_ATOMIC_ENERGY,
        data_key_force: str = KEY.SCALED_ATOMIC_FORCE,
        data_key_stress_iso: str = KEY.SCALED_ATOMIC_STRESS_ISO,
        data_key_stress_aniso: str = KEY.SCALED_ATOMIC_STRESS_ANISO,
    ):

        super().__init__()
        self.key_feature = data_key_feature
        self.key_energy = data_key_energy
        self.key_force = data_key_force
        self.key_stress_iso = data_key_stress_iso
        self.key_stress_aniso = data_key_stress_aniso

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        feature = data[self.key_feature]
        
        data[self.key_energy] = feature[..., :1]
        data[self.key_stress_iso] = feature[..., 1:2]
        data[self.key_force] = feature[..., 2:5]
        data[self.key_stress_aniso] = feature[..., 5:]
        
        return data