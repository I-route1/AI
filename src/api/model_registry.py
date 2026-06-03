_registry: dict = {}

def register(name: str, obj) -> None:
    _registry[name] = obj

def get(name: str):
    return _registry.get(name)

def is_available(name: str) -> bool:
    return _registry.get(name) is not None
