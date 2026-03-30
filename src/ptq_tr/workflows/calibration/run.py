"""Calibration workflow entrypoints."""

import urllib.request

import torch


def collect_image_batches(loader, device, limit=32, desc="Building calibration set..."):
    print(desc)
    image_list = []
    for i, info in enumerate(loader):
        data = torch.stack(info["image"])
        data = data.to(device)

        if i == limit:
            print("break")
            break
        print(i, end="|")
        data = data.to(device)
        image_list.append(data)

    print("done")
    return image_list


def download_imagenet_classes():
    print("Download ImageNet class index")
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = urllib.request.urlopen(url)
    imagenet_classes = [line.strip() for line in response.readlines()]
    print("done")
    return imagenet_classes


def calibrate_model(model, image_list):
    print("Quantizing model...")
    with torch.no_grad():
        model.set_calibration_flag()
        print()
        for i, image in enumerate(image_list):
            print(i, end="|")
            _ = model(image)
        model.unset_calibration_flag()
        print("Quantization parameter were set")
    print("All done")


def prepare_calibration_artifacts(
    train_loader,
    scale_opt_loader,
    device,
    calib_limit=32,
    scale_opt_limit=32,
    download_classes=True,
):
    image_list = collect_image_batches(
        train_loader,
        device,
        limit=calib_limit,
        desc="Building calibration set...",
    )
    scale_opt_image_list = collect_image_batches(
        scale_opt_loader,
        device,
        limit=scale_opt_limit,
        desc="Building fine tuning set...",
    )
    imagenet_classes = download_imagenet_classes() if download_classes else None
    print("Calibration set is ready")

    return {
        "image_list": image_list,
        "scale_opt_image_list": scale_opt_image_list,
        "imagenet_classes": imagenet_classes,
    }


def run_calibration(*args, **kwargs):
    if "model" not in kwargs or "image_list" not in kwargs:
        raise NotImplementedError("Calibration workflow wiring has not been migrated yet.")
    return calibrate_model(kwargs["model"], kwargs["image_list"])
