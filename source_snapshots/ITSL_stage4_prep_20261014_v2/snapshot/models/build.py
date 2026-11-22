from .cnn import SmallCNN
from .resnet import ResNet18SmallImage


def build_model(cfg: dict):
    name = cfg["model"].get("name", "SmallCNN")
    if name == "SmallCNN":
        return SmallCNN(
            num_classes=cfg["dataset"].get("num_classes", 10),
            input_channels=cfg["dataset"].get("input_channels", 3),
            use_aib=cfg["model"].get("use_aib", True),
        )
    if name == "ResNet18SmallImage":
        # Stage-4: AIB must be off; the constructor also enforces it.
        return ResNet18SmallImage(
            num_classes=cfg["dataset"].get("num_classes", 10),
            input_channels=cfg["dataset"].get("input_channels", 3),
            use_aib=cfg["model"].get("use_aib", False),
        )
    raise ValueError(f"Unsupported model: {name}")
