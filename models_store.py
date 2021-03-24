"""
A store for lazy loading and/or making single-instance
of all the AI models used throughout the code base.
"""

import gc
from torch_device import device as default_device

class LazyModel:
    def __init__(self, name: str, loading_code):
        self.loaded_model = None
        self.loaded_device = None
        self.loading_device = default_device
        self.name = name
        self.loading_code = loading_code
    def load(self):
        device = self.loading_device
        self.loaded_model = self.loading_code(device)
        self.loaded_device = device
    def unload(self):
        self.loaded_model = None
        gc.collect()
    def is_loaded(self) -> bool:
        return self.loaded_model is not None
    def get(self):
        if not self.is_loaded():
            self.load()
        return self.loaded_model
    def __str__(self):
        t = type(self)
        return f"<{t.__module__}.{t.__name__} object with name={self.name!r}, is_loaded()={self.is_loaded()}>"

class EagerModel(LazyModel):
    def __init__(self, name: str, loading_code):
        super().__init__(name, loading_code)
        self.load()

class Store:
    def __init__(self):
        self.models = {}
    def register(self, model: LazyModel):
        if not isinstance(model, LazyModel):
            raise TypeError(f"Argument model needs to be compatible with LazyModel, but got {type(model)}")
        if not isinstance(model.name, str):
            raise TypeError(f"Model name needs to be compatible with str, but got {type(model.name)}")
        if len(model.name) == 0:
            raise ValueError("Model name missing")
        if model.name in self.models:
            raise RuntimeError(f"Model name {model.name!r} already registered")
        self.models[model.name] = model
        return model
    def register_lazy(self, name: str, loading_code):
        return self.register(LazyModel(name, loading_code))
    def register_eager(self, name: str, loading_code):
        return self.register(EagerModel(name, loading_code))
    def __getitem__(self, name):
        if name not in self.models:
            raise KeyError(f"Model name {name!r} not registered")
        return self.models[name]

    def register_lazy_or_getitem(self, name: str, loading_code):
        if name in self.models:
            return self.__getitem__(name)
        return self.register_lazy(name, loading_code)
    def register_eager_or_getitem(self, name: str, loading_code):
        if name in self.models:
            return self.__getitem__(name)
        return self.register_eager(name, loading_code)

    def __str__(self):
        t = type(self)
        ms = []
        for k in self.models:
            v = self.models[k]
            ms.append((k, "loaded" if v.is_loaded() else "not loaded"))
        return f"<{t.__module__}.{t.__name__} object being store for {ms}>"

store = Store()
