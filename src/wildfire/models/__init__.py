from wildfire.models.resnet import ResNet18Classifier
from wildfire.models.small_cnn import SmallCNN


def build_model(config: dict):
    name = config["model"]["name"]
    if name == "small_cnn":
        return SmallCNN(
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        )
    if name == "resnet18":
        return ResNet18Classifier(
            num_classes=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"],
            dropout=config["model"]["dropout"],
        )
    raise ValueError(f"Unsupported model: {name}")
