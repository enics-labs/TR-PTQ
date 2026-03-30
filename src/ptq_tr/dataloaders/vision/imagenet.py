"""ImageNet dataset helpers."""

from datasets import load_dataset
from huggingface_hub.errors import LocalTokenNotFoundError

from ptq_tr.preprocessing.vision.image_processing import apply_preprocess
from ptq_tr.preprocessing.vision.transforms import build_imagenet_preprocess

def load_imagenet_stream(
    split,
    *,
    token=True,
    trust_remote_code=True,
    shuffle_seed=None,
    take=None,
):
    dataset = load_dataset(
        "imagenet-1k",
        split=split,
        streaming=True,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)

    if take is not None:
        dataset = dataset.take(take)

    return dataset


def map_imagenet_preprocess(dataset, preprocess):
    def fit_image(example):
        return apply_preprocess(example, preprocess)

    return dataset.map(fit_image)


def build_imagenet_datasets(
    *,
    calib_take=100,
    scale_opt_take=100,
    infer_shuffle_seed=42,
    calib_shuffle_seed=42,
    scale_opt_shuffle_seed=12,
    preprocess=None,
    token=True,
    trust_remote_code=True,
):
    preprocess = preprocess or build_imagenet_preprocess()

    try:
        dset_calib = load_imagenet_stream(
            "train",
            token=token,
            trust_remote_code=trust_remote_code,
            shuffle_seed=calib_shuffle_seed,
            take=calib_take,
        )
    except LocalTokenNotFoundError as tok_err:
        print("This project accessing huggingface. you need to provide token to apply it")
        print("You can use either one of to options:")
        print("Go to https://huggingface.co/settings/tokens and get the token (you might need to sign-in)")
        print('a. [CLI 1] - add env variable in PowerShell: $env:HF_TOKEN="hf_your_token_here" or add a token flag to run command --hf-token "hf_your_token_here"')
        print('b. [CLI 2] - run in terminal: hf auth login')
        print('c. insert the token here')
        print('d. Close run and let me handle it')
        ans = input("insert your answer here:").strip().lower()
        while ans not in {"a", "b", "c", "d"}:
            ans = input("did not understand please type one of the options (a/b/c/d):").strip().lower()
        if ans == 'a':
            raise SystemExit("provide the token and rerun")
        if ans == 'b':
            raise SystemExit("run `hf auth login` and rerun")
        if ans == 'c':
            token = input("insert the token: ").strip()
        if ans == 'd':
            raise SystemExit("exiting")

        dset_calib = load_imagenet_stream(
            "train",
            token=token,
            trust_remote_code=trust_remote_code,
            shuffle_seed=calib_shuffle_seed,
            take=calib_take,
        )

    dset_infer = load_imagenet_stream(
        "validation",
        token=token,
        trust_remote_code=trust_remote_code,
        shuffle_seed=infer_shuffle_seed,
    )
    dset_scale_opt = load_imagenet_stream(
        "train",
        token=token,
        trust_remote_code=trust_remote_code,
        shuffle_seed=scale_opt_shuffle_seed,
        take=scale_opt_take,
    )

    dset_infer_updated = map_imagenet_preprocess(dset_infer, preprocess)
    dset_calib_updated = map_imagenet_preprocess(dset_calib, preprocess)
    dset_scale_opt_updated = map_imagenet_preprocess(dset_scale_opt, preprocess)

    return {
        "infer": dset_infer,
        "calib": dset_calib,
        "scale_opt": dset_scale_opt,
        "infer_processed": dset_infer_updated,
        "calib_processed": dset_calib_updated,
        "scale_opt_processed": dset_scale_opt_updated,
    }


def build_imagenet_loaders(
    *,
    batch_size,
    calib_batch_size=1,
    scale_opt_batch_size=1,
    preprocess=None,
    token=True,
    trust_remote_code=True,
    calib_take=100,
    scale_opt_take=100,
):
    datasets = build_imagenet_datasets(
        calib_take=calib_take,
        scale_opt_take=scale_opt_take,
        preprocess=preprocess,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    val_loader = datasets["infer_processed"].iter(batch_size=batch_size)
    train_loader = datasets["calib_processed"].iter(batch_size=calib_batch_size)
    scale_opt_loader = datasets["scale_opt_processed"].iter(batch_size=scale_opt_batch_size)
    single_image_iter = iter(datasets["calib"])

    return {
        "val_loader": val_loader,
        "train_loader": train_loader,
        "scale_opt_loader": scale_opt_loader,
        "single_image_iter": single_image_iter,
        "datasets": datasets,
    }
