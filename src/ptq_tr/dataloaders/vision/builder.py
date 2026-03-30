"""Vision dataloader builders."""

from ptq_tr.dataloaders.vision.imagenet import build_imagenet_datasets, build_imagenet_loaders


def build_vision_dataloaders(dataset_name, **kwargs):
    if dataset_name == "imagenet":
        return build_imagenet_loaders(**kwargs)
    raise ValueError(f"Unsupported vision dataset: {dataset_name}")


def build_vision_datasets(dataset_name, **kwargs):
    if dataset_name == "imagenet":
        return build_imagenet_datasets(**kwargs)
    raise ValueError(f"Unsupported vision dataset: {dataset_name}")
