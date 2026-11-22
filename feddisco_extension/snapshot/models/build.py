from .cnn import SmallCNN


def build_model(cfg: dict):
    name = cfg["model"].get("name", "SmallCNN")
    if name != "SmallCNN":
        raise ValueError(f"Unsupported model: {name}")
    return SmallCNN(
        num_classes=cfg["dataset"].get("num_classes", 10),
        input_channels=cfg["dataset"].get("input_channels", 3),
        use_aib=cfg["model"].get("use_aib", True),
    )
