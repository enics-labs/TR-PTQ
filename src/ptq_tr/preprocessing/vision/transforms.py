"""Vision transform definitions."""

from torchvision import transforms

from ptq_tr.preprocessing.vision.image_processing import ensure_rgb


def build_imagenet_preprocess(image_size=224, resize_size=256):
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: ensure_rgb(img)),
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
