"""Validation workflow entrypoints."""

import time
import os

import torch.nn as nn
import torch

from ptq_tr.common.device import get_default_device
from ptq_tr.metrics.classification import AverageMeter, accuracy
from ptq_tr.preprocessing.nlp import extract_answer_from_span, tokenize_squad_batch


def validate(val_loader, model, criterion, device, samples=-1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    val_start_time = end = time.time()
    processed_samples = 0

    for i, info in enumerate(val_loader):
        data = torch.stack(info["image"])
        target = torch.tensor(info["label"])

        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        processed_samples += data.size(0)

        if i % 100 == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    processed_samples,
                    50000,
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )
        if samples > 0 and i >= samples:
            print(f"validation is finished (samples {samples})")
            break
    val_end_time = time.time()
    print(
        " * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}".format(
            top1=top1, top5=top5, time=val_end_time - val_start_time
        )
    )

    return losses.avg, top1.avg, top5.avg


def validate_nlp(val_loader, model, device, samples=-1):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for i, batch in enumerate(val_loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch["labels"] if "labels" in batch else batch["label"]
        if "label" in batch:
            batch["labels"] = batch.pop("label")

        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)

        if i % 100 == 0:
            print(
                "Eval: [{0}]\tLoss {1:.4f}\tAcc {2:.3f}".format(
                    total_examples,
                    total_loss / max(total_examples, 1),
                    100.0 * total_correct / max(total_examples, 1),
                )
            )

        if samples > 0 and total_examples >= samples:
            print(f"validation is finished (samples {samples})")
            break

    avg_loss = total_loss / max(total_examples, 1)
    accuracy_value = 100.0 * total_correct / max(total_examples, 1)
    print(" * Acc {0:.3f} Loss {1:.4f}".format(accuracy_value, avg_loss))
    return {"loss": avg_loss, "accuracy": accuracy_value}


def predict_squad_batch(model, tokenizer, batch, device, max_length=384, doc_stride=128, max_answer_length=30):
    encoded = tokenize_squad_batch(
        batch,
        tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
    )

    overflow_to_sample_mapping = encoded["overflow_to_sample_mapping"].tolist()
    offset_mapping = encoded["offset_mapping"]
    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key not in {"offset_mapping", "overflow_to_sample_mapping"}
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    start_logits = outputs.start_logits.detach().cpu()
    end_logits = outputs.end_logits.detach().cpu()

    best_answers = [{"text": "", "score": float("-inf")} for _ in batch["id"]]
    top_k = min(20, start_logits.size(1))

    for feature_index, sample_index in enumerate(overflow_to_sample_mapping):
        sequence_ids = encoded.sequence_ids(feature_index)
        context = batch["context"][sample_index]
        offsets = offset_mapping[feature_index].tolist()
        start_indexes = torch.topk(start_logits[feature_index], k=top_k).indices.tolist()
        end_indexes = torch.topk(end_logits[feature_index], k=top_k).indices.tolist()

        for start_index in start_indexes:
            if sequence_ids[start_index] != 1:
                continue
            for end_index in end_indexes:
                if sequence_ids[end_index] != 1:
                    continue
                if end_index < start_index or (end_index - start_index + 1) > max_answer_length:
                    continue

                answer_text = extract_answer_from_span(context, offsets, start_index, end_index)
                if not answer_text:
                    continue

                score = start_logits[feature_index, start_index].item() + end_logits[feature_index, end_index].item()
                if score > best_answers[sample_index]["score"]:
                    best_answers[sample_index] = {"text": answer_text, "score": score}

    predictions = []
    references = []
    for sample_index, sample_id in enumerate(batch["id"]):
        predictions.append(
            {
                "id": sample_id,
                "prediction_text": best_answers[sample_index]["text"],
            }
        )
        references.append(
            {
                "id": sample_id,
                "answers": batch["answers"][sample_index],
            }
        )

    return predictions, references


def validate_squad(
    val_loader,
    model,
    tokenizer,
    device,
    samples=-1,
    max_length=384,
    doc_stride=128,
    max_answer_length=30,
):
    import evaluate

    model.eval()
    metric = evaluate.load("squad")
    predictions = []
    references = []
    total_examples = 0

    for i, batch in enumerate(val_loader):
        batch_predictions, batch_references = predict_squad_batch(
            model,
            tokenizer,
            batch,
            device,
            max_length=max_length,
            doc_stride=doc_stride,
            max_answer_length=max_answer_length,
        )

        predictions.extend(batch_predictions)
        references.extend(batch_references)
        total_examples += len(batch_predictions)

        if i % 10 == 0:
            partial_results = metric.compute(predictions=predictions, references=references)
            print(
                "Eval: [{0}]\tEM {1:.3f}\tF1 {2:.3f}".format(
                    total_examples,
                    partial_results["exact_match"],
                    partial_results["f1"],
                )
            )

        if samples > 0 and total_examples >= samples:
            print(f"validation is finished (samples {samples})")
            break

    results = metric.compute(predictions=predictions, references=references)
    print(" * Exact Match {0:.3f} F1 {1:.3f}".format(results["exact_match"], results["f1"]))
    return results


def validate_model(model, val_loader, criterion=None, device=None, samples=-1):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    device = device or get_default_device()
    criterion = criterion or nn.CrossEntropyLoss().to(device)
    model = model.to(device)

    try:
        print("Validating...")
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, device, samples=samples)
    except KeyboardInterrupt:
        print("stopped by a Keyboard Interrupt")
        raise
    finally:
        model = model.to(torch.device("cpu"))

    return {
        "loss": val_loss,
        "top1": val_prec1,
        "top5": val_prec5,
    }


def validate_nlp_model(model, val_loader, device=None, samples=-1):
    device = device or get_default_device()
    model = model.to(device)

    try:
        print("Validating...")
        return validate_nlp(val_loader, model, device, samples=samples)
    except KeyboardInterrupt:
        print("stopped by a Keyboard Interrupt")
        raise
    finally:
        model = model.to(torch.device("cpu"))


def validate_squad_model(
    model,
    val_loader,
    tokenizer,
    device=None,
    samples=-1,
    max_length=384,
    doc_stride=128,
    max_answer_length=30,
):
    device = device or get_default_device()
    model = model.to(device)

    try:
        print("Validating...")
        return validate_squad(
            val_loader,
            model,
            tokenizer,
            device,
            samples=samples,
            max_length=max_length,
            doc_stride=doc_stride,
            max_answer_length=max_answer_length,
        )
    except KeyboardInterrupt:
        print("stopped by a Keyboard Interrupt")
        raise
    finally:
        model = model.to(torch.device("cpu"))


def run_validation(*args, **kwargs):
    if kwargs.get("task") == "nlp":
        if kwargs.get("task_type") == "question_answering":
            if "model" not in kwargs or "val_loader" not in kwargs or "tokenizer" not in kwargs:
                raise NotImplementedError("NLP question answering workflow wiring has not been migrated yet.")
            return validate_squad_model(
                kwargs["model"],
                kwargs["val_loader"],
                kwargs["tokenizer"],
                device=kwargs.get("device"),
                samples=kwargs.get("samples", -1),
                max_length=kwargs.get("max_length", 384),
                doc_stride=kwargs.get("doc_stride", 128),
                max_answer_length=kwargs.get("max_answer_length", 30),
            )
        if "model" not in kwargs or "val_loader" not in kwargs:
            raise NotImplementedError("NLP validation workflow wiring has not been migrated yet.")
        return validate_nlp_model(
            kwargs["model"],
            kwargs["val_loader"],
            device=kwargs.get("device"),
            samples=kwargs.get("samples", -1),
        )
    if "model" not in kwargs or "val_loader" not in kwargs:
        raise NotImplementedError("Validation workflow wiring has not been migrated yet.")
    return validate_model(
        kwargs["model"],
        kwargs["val_loader"],
        criterion=kwargs.get("criterion"),
        device=kwargs.get("device"),
        samples=kwargs.get("samples", -1),
    )
