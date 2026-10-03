from .megatron_fsdp_strategy import MegatronFSDPStrategy

__all__ = ['MegatronFSDPStrategy']

STRATEGY_REGISTRY = {'megatron_fsdp': MegatronFSDPStrategy}


def build_strategy(strategy_type: str, device_id: int = 0, **kwargs):
    """Factory for building a strategy from its type name."""
    return STRATEGY_REGISTRY[strategy_type](device_id=device_id, **kwargs)
