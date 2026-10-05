"""Identity applied-command4 and observed xyz/opening State20 transport.

Missing wrist rotation must NOT mask observed position. No achieved-pose,
relative-rotation, gripper binarization or per-chunk normalization is used.
"""
import torch
from .mixed_native7 import MinMaxMixedActionCodec
from .state_trajectory import _fp32_robot_boundary, _strict_matrix
from metaworld_benchmark.contract import (ROBOT, PROTOCOL, STATS_FORMAT,
    STATE_SLOTS, ACTION_LAYOUT, digest)


class MetaWorldNative4Codec(MinMaxMixedActionCodec):
    def __init__(self, stats=None, *, embodiments=None):
        selected = tuple(embodiments or (ROBOT,))
        if selected != (ROBOT,): raise ValueError('Native4 registry differs')
        super().__init__(embodiments=selected)
        self.contract_version.fill_(8)
        self.state_stat_valid.zero_(); self.state_stat_valid[:, list(STATE_SLOTS)] = True
        self.action_stat_valid.zero_(); self.action_stat_valid[:, :4] = True
        self.native7_embodiments.zero_()
        if stats is not None:
            expected = dict(format=STATS_FORMAT, target_contract=PROTOCOL,
                embodiments=[ROBOT], state_slots=list(STATE_SLOTS), action_layout=list(ACTION_LAYOUT),
                normalization='state_minmax_action_applied4_identity')
            if any(stats.get(k) != v for k, v in expected.items()):
                raise ValueError('MetaWorld statistics semantic contract differs')
            for mode in ('state', 'action'):
                lo = _strict_matrix(stats, mode+'_min', rows=1, dtype=torch.float64)
                hi = _strict_matrix(stats, mode+'_max', rows=1, dtype=torch.float64)
                valid = _strict_matrix(stats, mode+'_stat_valid', rows=1, dtype=torch.bool)
                degenerate = _strict_matrix(stats, mode+'_degenerate', rows=1, dtype=torch.bool)
                if (not torch.equal(valid, getattr(self, mode+'_stat_valid'))
                        or not torch.isfinite(lo).all() or not torch.isfinite(hi).all()
                        or (hi < lo).any() or (valid & (degenerate != ((hi-lo) < 1e-6))).any()
                        or (valid & ~degenerate & (hi.float() <= lo.float())).any()):
                    raise ValueError('MetaWorld bounds/validity are invalid')
                if mode == 'action' and (not (lo[:, :4] == -1).all() or not (hi[:, :4] == 1).all()):
                    raise ValueError('Native4 actions must remain identity [-1,1]')
                for suffix, value in (('min',lo),('max',hi),('stat_valid',valid),('degenerate',degenerate)):
                    getattr(self,mode+'_'+suffix).copy_(value)
            self.stats_sha256.copy_(torch.tensor(list(bytes.fromhex(digest(stats))), dtype=torch.uint8))
        for key in ('action_layout_sha256', 'state_layout_sha256', 'coordinate_convention_sha256'):
            getattr(self,key).copy_(torch.tensor(list(bytes.fromhex(digest((PROTOCOL,STATE_SLOTS,ACTION_LAYOUT)))),dtype=torch.uint8))

    @_fp32_robot_boundary
    def encode_inference_state(self,current_state_abs,current_state_abs_valid,embodiment_id):
        self._validate_shapes(current_state_abs,current_state_abs_valid)
        lo,hi,sv,dg = self._select_statistics(embodiment_id,batch=current_state_abs.shape[0],
            q01=self.state_min,q99=self.state_max,valid=self.state_stat_valid,
            degenerate=self.state_degenerate,add_time_axis=False)
        mask=current_state_abs_valid.bool() & sv
        value,mask=self._normalize(torch.where(mask,current_state_abs.float(),0),mask,lo,hi,sv,dg)
        return value.float(),mask

    @_fp32_robot_boundary
    def encode_training_targets(self,current_state_abs,current_state_abs_valid,
                                target_state_abs,target_state_abs_valid,embodiment_id):
        state,sv=self.encode_inference_state(current_state_abs,current_state_abs_valid,embodiment_id)
        av=target_state_abs_valid.bool() & self.action_stat_valid[embodiment_id.long(),None]
        return torch.where(av,target_state_abs.float(),0),av,state,sv

    @_fp32_robot_boundary
    def inference_action_valid(self,current_state_abs,current_state_abs_valid,embodiment_id,horizon):
        available=current_state_abs_valid[:,list(STATE_SLOTS)].all(-1,keepdim=True)
        return (self.action_stat_valid[embodiment_id.long()] & available)[:,None].expand(-1,horizon,-1)

    @_fp32_robot_boundary
    def decode_predictions(self,normalized_action,current_state_abs,current_state_abs_valid,
                           embodiment_id,action_valid=None):
        mask=self.inference_action_valid(current_state_abs,current_state_abs_valid,embodiment_id,normalized_action.shape[1])
        if action_valid is not None: mask=mask & action_valid.bool()
        return torch.where(mask,normalized_action.float(),0),mask

    def encode_raw_training_targets(self,*args,**kwargs):
        raise RuntimeError('Native4 commands must not enter the relative EEF codec')

    @_fp32_robot_boundary
    def raw_action_targets(self,current,current_valid,target,target_valid,embodiment_id):
        valid=target_valid.bool() & self.action_stat_valid[embodiment_id.long(),None]
        return torch.where(valid,target,0).float(),valid
