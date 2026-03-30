"""Image preprocessing helpers."""


def ensure_rgb(img):
    return img.convert("RGB") if img.mode != "RGB" else img


def apply_preprocess(example, preprocess):
    # Some dataset variants may expose raw bytes instead of a PIL image.
    example["image"] = preprocess(example["image"])
    example["label"] = example["label"]
    return example
