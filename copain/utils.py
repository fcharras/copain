import torch.nn as nn


class WeightInitializer:
    def __init__(self, initialize_fn, initialize_fn_kwargs):
        self.initialize_fn = initialize_fn
        self.initialize_fn_kwargs = initialize_fn_kwargs

    def __call__(self, module):
        if hasattr(module, "weight"):
            self.initialize_fn(module.weight, **self.initialize_fn_kwargs)
        if hasattr(module, "bias"):
            nn.init.constant_(module.bias, 0)
