import warnings

device = "cpu"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torch
    if torch.cuda.is_available():
        device = "cuda"
