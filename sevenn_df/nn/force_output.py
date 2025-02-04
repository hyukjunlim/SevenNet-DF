import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn_df._keys as KEY
from sevenn_df._const import AtomGraphDataType

from .util import _broadcast


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
        data_key_stress: str = KEY.PRED_STRESS,
        data_key_atomic_stress: str = KEY.ATOMIC_STRESS,
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
        self.key_atomic_stress = data_key_atomic_stress
        self.key_force_drv = data_key_force_drv
        self.key_stress_drv = data_key_stress_drv
        self.key_cell_volume = data_key_cell_volume
        self._is_batch_data = True
        
        # self.change_mat = torch.tensor(
        #     [
        #         [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
        #         [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
        #         [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
        #         [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
        #         [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
        #         [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
        #         [-(6 ** (-0.5)), 0, 0, 0, 2 * 6 ** (-0.5), 0, 0, 0, -(6 ** (-0.5))],
        #         [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
        #         [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
        #     ],
        #     device=device
        # )
    
    def get_grad_key(self):
        return self.key_edge
    
    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        volume = data[self.key_cell_volume]
        data[self.key_energy] = data[self.key_energy].squeeze(-1)
        v = data[self.key_atomic_stress]
        
        if self._is_batch_data:
            # # Derivating method
            # tot_num = torch.sum(data[KEY.NUM_ATOMS])  # ? item?
            # rij = data[self.key_edge]
            # energy = [(data[self.key_energy]).sum()]
            # edge_idx = data[self.key_edge_idx]

            # grad = torch.autograd.grad(
            #     energy,
            #     [rij],
            #     create_graph=self.training,
            #     allow_unused=True
            # )

            # # make grad is not Optional[Tensor]
            # fij = grad[0]

            # if fij is not None:
            #     # compute force
            #     pf = torch.zeros(tot_num, 3, dtype=fij.dtype, device=fij.device)
            #     nf = torch.zeros(tot_num, 3, dtype=fij.dtype, device=fij.device)
            #     _edge_src = _broadcast(edge_idx[0], fij, 0)
            #     _edge_dst = _broadcast(edge_idx[1], fij, 0)
            #     pf.scatter_reduce_(0, _edge_src, fij, reduce='sum')
            #     nf.scatter_reduce_(0, _edge_dst, fij, reduce='sum')
            #     data[self.key_force_drv] = pf - nf

            #     # compute virial
            #     diag = rij * fij
            #     s12 = rij[..., 0] * fij[..., 1]
            #     s23 = rij[..., 1] * fij[..., 2]
            #     s31 = rij[..., 2] * fij[..., 0]
            #     # cat last dimension
            #     _virial = torch.cat([
            #         diag,
            #         s12.unsqueeze(-1),
            #         s23.unsqueeze(-1),
            #         s31.unsqueeze(-1)
            #     ], dim=-1)

            #     _s = torch.zeros(tot_num, 6, dtype=fij.dtype, device=fij.device)
            #     _edge_dst6 = _broadcast(edge_idx[1], _virial, 0)
            #     _s.scatter_reduce_(0, _edge_dst6, _virial, reduce='sum')

            #     batch = data[KEY.BATCH]  # for deploy, must be defined first
            #     nbatch = int(batch.max().cpu().item()) + 1
            #     sout = torch.zeros(
            #         (nbatch, 6), dtype=_virial.dtype, device=_virial.device
            #     )
            #     _batch = _broadcast(batch, _s, 0)
            #     sout.scatter_reduce_(0, _batch, _s, reduce='sum')
            #     data[self.key_stress_drv] =\
            #         torch.neg(sout) / volume.unsqueeze(-1)
                    
            # Direct method
            size = volume.shape[0]
            # out = torch.zeros((size, 6), requires_grad=False)
            # out[:, 1:6] = out.view(size, -1)
            # voigt = torch.einsum("ba,cb->ca", self.cg_change_mat, out).reshape(size, 9)[:, [0, 4, 8, 1, 5, 2]]
            
            src = torch.einsum('bi,bj->bij', v, v).reshape(-1, 9)[:, [0, 4, 8, 1, 5, 2]]
            src_shape = src.shape
            size = int(data[KEY.BATCH].max()) + 1
            output = torch.zeros(
                (size, *src_shape[1:]), dtype=src.dtype, device=src.device
            )
            output.scatter_reduce_(0, data[KEY.BATCH].unsqueeze(-1).expand_as(src), src, reduce='sum')
            data[self.key_stress] = output / volume.view(-1, 1)
            print(volume, flush=True)
            print(data[self.key_force])
            print(v, flush=True)
            print(data[self.key_stress], flush=True)
        else:
            voigt = torch.einsum('ni,nj->nij', v, v).reshape(-1, 9)[[0, 4, 8, 1, 5, 2]]
            data[self.key_stress] = torch.sum(voigt, dim=0) / volume
            print(volume, flush=True)
            print(data[self.key_force])
            print(v, flush=True)
            print(data[self.key_stress], flush=True)
        
        return data
      
# @compile_mode('script')
# class DirectEnergyStressOutput(nn.Module):
#     """
#     Process energy and stress.
#     Used in serial torchscipt models
#     """
#     def __init__(
#         self,
#         data_key_energy: str = KEY.PRED_ENERGY,
#         data_key_stress: str = KEY.PRED_STRESS,
#         data_key_cell_volume: str = KEY.CELL_VOLUME,
#     ):

#         super().__init__()
#         self.key_energy = data_key_energy
#         self.key_stress = data_key_stress
#         self.key_cell_volume = data_key_cell_volume
#         self._is_batch_data = True

#     def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
#         volume = data[self.key_cell_volume]
        
#         data[self.key_energy] = data[self.key_energy].squeeze(-1)
#         v = data[self.key_stress]
        
#         if self._is_batch_data:
#             size = volume.shape[0]
#             voigt = torch.einsum('bi,bj->bij', v, v).reshape(size, 9)[:, [0, 4, 8, 1, 5, 2]]
#             data[self.key_stress] = voigt / volume.view(-1, 1)
#         else:
#             voigt = torch.einsum('i,j->ij', v, v).reshape(9)[[0, 4, 8, 1, 5, 2]]
#             data[self.key_stress] = voigt / volume
        
#         return data


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
        data_key_stress: str = KEY.SCALED_ATOMIC_STRESS,
    ):

        super().__init__()
        self.key_feature = data_key_feature
        self.key_energy = data_key_energy
        self.key_force = data_key_force
        self.key_stress = data_key_stress

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        feature = data[self.key_feature]
        
        data[self.key_energy] = feature[..., :1]
        data[self.key_force] = feature[..., 1:4]
        data[self.key_stress] = feature[..., 4:7]
        # data[self.key_force] = feature[..., 2:5]
        # data[self.key_stress] = feature[..., 1:2] + feature[..., 5:10]
        
        return data