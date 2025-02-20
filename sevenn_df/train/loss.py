from typing import Any, Callable, Dict, Optional

import math
import torch
import torch.nn.functional as F

import sevenn_df._keys as KEY


class LossDefinition:
    """
    Base class for loss definition
    weights are defined in outside of the class
    """

    def __init__(
        self,
        name: str,
        unit: Optional[str] = None,
        criterion: Optional[Callable] = None,
        ref_key: Optional[str] = None,
        pred_key: Optional[str] = None
    ):
        self.name = name
        self.unit = unit
        self.criterion = criterion
        self.ref_key = ref_key
        self.pred_key = pred_key

    def __repr__(self):
        return self.name

    def assign_criteria(self, criterion: Callable):
        if self.criterion is not None:
            raise ValueError('Loss uses its own criterion.')
        self.criterion = criterion

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        if self.pred_key is None or self.ref_key is None:
            raise NotImplementedError('LossDefinition is not implemented.')
        return torch.reshape(batch_data[self.pred_key], (-1,)), torch.reshape(
            batch_data[self.ref_key], (-1,)
        )

    def get_loss(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        """
        Function that return scalar
        """
        if self.criterion is None:
            raise NotImplementedError('LossDefinition has no criterion.')
        return self.criterion(*self._preprocess(batch_data, model))


class PerAtomEnergyLoss(LossDefinition):
    """
    Loss for per atom energy
    """

    def __init__(
        self,
        name: str = 'Energy',
        unit: str = 'eV/atom',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.ENERGY,
        pred_key: str = KEY.PRED_ENERGY,
    ):
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key
        )

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        num_atoms = batch_data[KEY.NUM_ATOMS]
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        return (
            batch_data[self.pred_key] / num_atoms,
            batch_data[self.ref_key] / num_atoms,
        )


class ForceLoss(LossDefinition):
    """
    Loss for force
    """

    def __init__(
        self,
        name: str = 'Force',
        unit: str = 'eV/A',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.FORCE,
        pred_key: str = KEY.PRED_FORCE,
    ):
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key
        )

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        return (
            torch.reshape(batch_data[self.pred_key], (-1,)),
            torch.reshape(batch_data[self.ref_key], (-1,)),
        )


class StressLoss(LossDefinition):
    """
    Loss for stress this is kbar
    """

    def __init__(
        self,
        name: str = 'Stress',
        unit: str = 'kbar',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.STRESS,
        pred_key: str = KEY.PRED_STRESS,
    ):
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key
        )
        self.TO_KB = 1602.1766208  # eV/A^3 to kbar

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        return (
            torch.reshape(batch_data[self.pred_key] * self.TO_KB, (-1,)),
            torch.reshape(batch_data[self.ref_key] * self.TO_KB, (-1,)),
        )

class ConsistencyForceLoss(LossDefinition):
    """
    Loss for Consistency of Force
    """

    def __init__(
        self,
        name: str = 'ConsistencyForce',
        unit: str = 'eV/A',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.FORCE,
        pred_key: str = KEY.PRED_FORCE_DRV,
    ):
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key
        )

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        return (
            torch.reshape(batch_data[self.pred_key], (-1,)),
            torch.reshape(batch_data[self.ref_key], (-1,)),
        )

class ConsistencyForceLoss2(LossDefinition):
    """
    Loss for Consistency of Force
    """

    def __init__(
        self,
        name: str = 'ConsistencyForce2',
        unit: str = 'eV/A',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.PRED_FORCE_DRV,
        pred_key: str = KEY.PRED_FORCE,
    ):
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key
        )

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        return (
            torch.reshape(batch_data[self.pred_key], (-1,)),
            torch.reshape(batch_data[self.ref_key], (-1,)),
        )

class ConsistencyStressLoss(LossDefinition):
    """
    Loss for Consistency for Stress this is kbar
    """

    def __init__(
        self,
        name: str = 'ConsistencyStress',
        unit: str = 'kbar',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.STRESS,
        pred_key: str = KEY.PRED_STRESS_DRV,
    ):
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key
        )
        self.TO_KB = 1602.1766208  # eV/A^3 to kbar

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        return (
            torch.reshape(batch_data[self.pred_key] * self.TO_KB, (-1,)),
            torch.reshape(batch_data[self.ref_key] * self.TO_KB, (-1,)),
        )
        
class ConsistencyStressLoss2(LossDefinition):
    """
    Loss for Consistency for Stress this is kbar
    """

    def __init__(
        self,
        name: str = 'ConsistencyStress2',
        unit: str = 'kbar',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.PRED_STRESS_DRV,
        pred_key: str = KEY.PRED_STRESS,
    ):
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key
        )
        self.TO_KB = 1602.1766208  # eV/A^3 to kbar

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        return (
            torch.reshape(batch_data[self.pred_key] * self.TO_KB, (-1,)),
            torch.reshape(batch_data[self.ref_key] * self.TO_KB, (-1,)),
        )
        
class NoiseConsistencyForceLoss2(LossDefinition):
    """
    Loss for Noise Consistency of Force
    """

    def __init__(
        self,
        name: str = 'NoiseConsistencyForce2',
        unit: str = 'eV/A',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.PRED_FORCE_DRV,
        pred_key: str = KEY.PRED_FORCE,
    ):
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key
        )

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        return (
            torch.reshape(batch_data[self.pred_key], (-1,)),
            torch.reshape(batch_data[self.ref_key], (-1,)),
        )
        
class NoiseConsistencyStressLoss2(LossDefinition):
    """
    Loss for Noise Consistency for Stress this is kbar
    """

    def __init__(
        self,
        name: str = 'NoiseConsistencyStress2',
        unit: str = 'kbar',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.PRED_STRESS_DRV,
        pred_key: str = KEY.PRED_STRESS,
    ):
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key
        )
        self.TO_KB = 1602.1766208  # eV/A^3 to kbar

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        return (
            torch.reshape(batch_data[self.pred_key] * self.TO_KB, (-1,)),
            torch.reshape(batch_data[self.ref_key] * self.TO_KB, (-1,)),
        )

# class ArcForce(LossDefinition):
#     """
#     Cosine triplet loss for force
#     """
#     def __init__(
#         self, 
#         name: str = 'Force',
#         unit: str = 'eV/A',
#         criterion: Optional[Callable] = torch.nn.MarginRankingLoss(margin=0.3),
#         # criterion: Optional[Callable] = torch.nn.MarginRankingLoss(margin=math.pi * 0.75),
#         ref_key: str = KEY.FORCE,
#         pred_key: list = [KEY.PRED_FORCE, KEY.PRED_FORCE_DRV],
#     ):
#         super().__init__(
#             name=name,
#             unit=unit,
#             criterion=criterion,
#             ref_key=ref_key,
#             pred_key=pred_key,
#         )
    
#     def _preprocess(
#         self,
#         batch_data: Dict[str, Any],
#         model: Optional[Callable] = None
#     ):
#         assert all(isinstance(i, str) for i in self.pred_key) and isinstance(self.ref_key, str)
        
#         anchor = batch_data[self.ref_key]
#         positive = batch_data[self.pred_key[0]]
#         negative = batch_data[self.pred_key[1]] * (-1)
        
#         a_p_cos = 1 - F.cosine_similarity(anchor, positive)
#         # a_p_cos = torch.acos(F.cosine_similarity(anchor, positive))
#         a_n_cos = 1 - F.cosine_similarity(anchor, negative)
#         # a_n_cos = torch.acos(F.cosine_similarity(anchor, negative))
        
#         return (
#             torch.reshape(a_p_cos, (-1,)),
#             torch.reshape(a_n_cos, (-1,)),
#             torch.reshape(torch.tensor([-1], device=a_p_cos.device), (-1,))
        
def get_loss_functions_from_config(config: Dict[str, Any]):
    from sevenn.train.optim import loss_dict

    loss_functions = []  # list of tuples (loss_definition, weight)

    loss = loss_dict[config[KEY.LOSS].lower()]
    try:
        loss_param = config[KEY.LOSS_PARAM]
    except KeyError:
        loss_param = {}
    criterion = loss(**loss_param)
    
    consistency_weight = 0.5
    consistency2_weight = 0.1
    noiseconsistency2_weight = 0.05
    loss_functions.append((PerAtomEnergyLoss(), 1.0))
    loss_functions.append((ForceLoss(), config[KEY.FORCE_WEIGHT]))
    loss_functions.append((ConsistencyForceLoss(), config[KEY.FORCE_WEIGHT] * consistency_weight))
    loss_functions.append((ConsistencyForceLoss2(), config[KEY.FORCE_WEIGHT] * consistency2_weight))
    loss_functions.append((NoiseConsistencyForceLoss2(), config[KEY.FORCE_WEIGHT] * noiseconsistency2_weight))
    if config[KEY.IS_TRAIN_STRESS]:
        loss_functions.append((StressLoss(), config[KEY.STRESS_WEIGHT]))
        loss_functions.append((ConsistencyStressLoss(), config[KEY.STRESS_WEIGHT] * consistency_weight))
        loss_functions.append((ConsistencyStressLoss2(), config[KEY.STRESS_WEIGHT] * consistency2_weight))
        loss_functions.append((NoiseConsistencyStressLoss2(), config[KEY.STRESS_WEIGHT] * noiseconsistency2_weight))
        
    for loss_function, _ in loss_functions:
        if loss_function.criterion is None:
            loss_function.assign_criteria(criterion)

    return loss_functions
