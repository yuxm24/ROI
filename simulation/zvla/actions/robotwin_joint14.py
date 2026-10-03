"""Absolute drive-target Joint14 in a masked 20-column transport envelope.

No EEF/Rot6D conversion is performed. Joint observations are the pinned
simulator's drive targets, NOT its separately exposed achieved joint positions.
"""
import hashlib
import json

import torch

from .mixed_native7 import MinMaxMixedActionCodec, MINMAX_MIN_SPAN
from .state_trajectory import _fp32_robot_boundary, _strict_matrix

from zvla_runtime.robotwin_joint14 import ROBOT_TYPE,CONTRACT,FORMAT,LAYOUT


class Joint14MinMaxCodec(MinMaxMixedActionCodec):
    def __init__(self, stats=None, *, embodiments=None):
        selected=tuple(embodiments or (ROBOT_TYPE,))
        if selected!=(ROBOT_TYPE,):
            raise ValueError('Joint14 requires its own embodiment registry')
        super().__init__(embodiments=selected)
        self.contract_version.fill_(7)
        self.native7_embodiments.zero_()
        for modality in ('state','action'):
            getattr(self,modality+'_stat_valid')[:,14:]=False
        if stats is not None:
            expected=dict(format=FORMAT,target_contract=CONTRACT,embodiments=list(selected),
                          state_layout=list(LAYOUT),action_layout=list(LAYOUT),
                          normalization='fixed_per_dim_minmax_shared_across_horizon')
            if any(stats.get(k)!=v for k,v in expected.items()):
                raise ValueError('Joint14 statistics semantic contract mismatch')
            for modality in ('state','action'):
                lo=_strict_matrix(stats,modality+'_min',rows=1,dtype=torch.float64)
                hi=_strict_matrix(stats,modality+'_max',rows=1,dtype=torch.float64)
                valid=_strict_matrix(stats,modality+'_stat_valid',rows=1,dtype=torch.bool)
                degenerate=_strict_matrix(stats,modality+'_degenerate',rows=1,dtype=torch.bool)
                if (not valid[:,:14].all() or valid[:,14:].any() or not torch.isfinite(lo).all()
                        or not torch.isfinite(hi).all() or (hi<lo).any()
                        or (valid & (degenerate!=((hi-lo)<MINMAX_MIN_SPAN))).any()
                        or (valid & ~degenerate & (hi.float()<=lo.float())).any()):
                    raise ValueError('invalid Joint14 bounds, validity or constant dimensions')
                for suffix,value in (('min',lo),('max',hi),('stat_valid',valid),('degenerate',degenerate)):
                    getattr(self,modality+'_'+suffix).copy_(value)
            digest=hashlib.sha256(json.dumps(dict(stats),sort_keys=True,separators=(',',':')).encode()).digest()
            self.stats_sha256.copy_(torch.tensor(list(digest),dtype=torch.uint8))
        digest=hashlib.sha256(json.dumps((CONTRACT,LAYOUT),separators=(',',':')).encode()).digest()
        for name in ('action_layout_sha256','state_layout_sha256','coordinate_convention_sha256'):
            getattr(self,name).copy_(torch.tensor(list(digest),dtype=torch.uint8))

    def _encode(self,value,valid,ids,modality):
        if value.shape!=valid.shape or value.shape[-1]!=20:
            raise ValueError('Joint14 requires matching masked 20-column tensors')
        lo,hi,sv,dg=self._select_statistics(ids,batch=value.shape[0],
            q01=getattr(self,modality+'_min'),q99=getattr(self,modality+'_max'),
            valid=getattr(self,modality+'_stat_valid'),degenerate=getattr(self,modality+'_degenerate'),
            add_time_axis=value.ndim==3)
        mask=valid.bool() & sv
        # Invalid/sentinel values are never allowed to produce NaN gradients.
        out,mask=self._normalize(torch.where(mask,value.float(),0),mask,lo,hi,sv,dg)
        return out.float(),mask

    @_fp32_robot_boundary
    def encode_inference_state(self,current_state_abs,current_state_abs_valid,embodiment_id):
        return self._encode(current_state_abs,current_state_abs_valid,embodiment_id,'state')

    @_fp32_robot_boundary
    def encode_training_targets(self,current_state_abs,current_state_abs_valid,
                                target_state_abs,target_state_abs_valid,embodiment_id):
        state,sv=self.encode_inference_state(current_state_abs,current_state_abs_valid,embodiment_id)
        action,av=self._encode(target_state_abs,target_state_abs_valid,embodiment_id,'action')
        return action,av,state,sv

    @_fp32_robot_boundary
    def inference_action_valid(self,current_state_abs,current_state_abs_valid,embodiment_id,horizon):
        return (self.action_stat_valid.index_select(0,embodiment_id.long()) & current_state_abs_valid.bool())[:,None].expand(-1,horizon,-1)

    @_fp32_robot_boundary
    def decode_predictions(self,normalized_action,current_state_abs,current_state_abs_valid,
                           embodiment_id,action_valid=None):
        valid=self.inference_action_valid(current_state_abs,current_state_abs_valid,embodiment_id,normalized_action.shape[1])
        if action_valid is not None:
            valid=valid & action_valid.bool()
        lo,hi,sv,dg=self._select_statistics(embodiment_id,batch=normalized_action.shape[0],
            q01=self.action_min,q99=self.action_max,valid=self.action_stat_valid,
            degenerate=self.action_degenerate,add_time_axis=True)
        value=self._denormalize(torch.where(valid,normalized_action.float(),0),lo,hi,dg)
        return torch.where(valid,value,0).float(),valid

    def encode_raw_training_targets(self,*args,**kwargs):
        raise RuntimeError('Joint14 must not enter the EEF geometry codec; use raw_action_targets')

    @_fp32_robot_boundary
    def raw_action_targets(self,current,current_valid,target,target_valid,embodiment_id):
        valid=target_valid.bool() & self.action_stat_valid.index_select(0,embodiment_id.long())[:,None]
        return torch.where(valid,target,0).float(),valid
