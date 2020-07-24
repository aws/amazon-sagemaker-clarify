def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
