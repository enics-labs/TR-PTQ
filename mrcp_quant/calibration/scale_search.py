import numpy as np

from .metrics import cosine_similarity, nrmse


def update_scale_factor(module, observer, base_min, base_max, scale_factor):
    observer.min_val = base_min * scale_factor
    observer.max_val = base_max * scale_factor
    observer.calculate_qparams()
    module.is_weights_quantized = False


def _line_search(scales, score_fn, better_fn, initial_score, initial_scale=1.0):
    best_score = initial_score
    best_scale = initial_scale

    for scale_factor in scales:
        score = score_fn(scale_factor)
        if better_fn(score, best_score):
            best_score = score
            best_scale = scale_factor

    return best_scale, best_score


def find_optimal_scale_factor(
    update_scale_fn,
    forward_pass,
    inputs,
    reference_output,
    alpha,
    beta,
    num_scales,
):
    """Find the scale factor that maximizes output cosine similarity."""

    def score_scale(scale_factor):
        update_scale_fn(scale_factor)
        candidate_output = forward_pass(inputs)
        return cosine_similarity(reference_output, candidate_output)

    optimal_scale, _ = _line_search(
        scales=np.linspace(alpha, beta, num=num_scales),
        score_fn=score_scale,
        better_fn=lambda score, best: score > best,
        initial_score=-float("inf"),
    )
    return optimal_scale


def calc_cosine(layer, reference_output, inputs, alpha=0.9, beta=1.1, num_scales=10):
    """Optimize LayerNorm input and weight scale factors with cosine similarity."""
    optimal_weight_scale = find_optimal_scale_factor(
        layer.update_w_scale_factor,
        layer.forward_pass,
        inputs,
        reference_output,
        alpha,
        beta,
        num_scales,
    )
    optimal_input_scale = find_optimal_scale_factor(
        layer.update_in_scale_factor,
        layer.forward_pass,
        inputs,
        reference_output,
        alpha,
        beta,
        num_scales,
    )

    layer.set_optimal_w_scale_factor(optimal_weight_scale)
    layer.set_optimal_in_scale_factor(optimal_input_scale)


def find_activation_scale(
    inputs,
    module,
    observer,
    alpha=0.5,
    beta=2.0,
    num_scales=10,
    metric=cosine_similarity,
):
    base_min = observer.min_val
    base_max = observer.max_val

    def score_scale(scale_factor):
        update_scale_factor(module, observer, base_min, base_max, scale_factor)
        candidate_output = module.forward_pass(inputs)
        reference_output = module.float_forward_pass(inputs)
        return metric(reference_output, candidate_output)

    optimal_scale, _ = _line_search(
        scales=np.linspace(alpha, beta, num=num_scales),
        score_fn=score_scale,
        better_fn=lambda score, best: score > best,
        initial_score=-1,
    )
    update_scale_factor(module, observer, base_min, base_max, 1)
    return optimal_scale


def find_binary_input_scale(
    first_input,
    second_input,
    module,
    observer,
    alpha=0.5,
    beta=2.0,
    num_scales=10,
):
    base_min = observer.min_val
    base_max = observer.max_val

    def score_scale(scale_factor):
        update_scale_factor(module, observer, base_min, base_max, scale_factor)
        candidate_output = module.forward_pass(first_input, second_input)
        reference_output = module.float_forward_pass(first_input, second_input)
        return cosine_similarity(reference_output, candidate_output)

    optimal_scale, _ = _line_search(
        scales=np.linspace(alpha, beta, num=num_scales),
        score_fn=score_scale,
        better_fn=lambda score, best: score > best,
        initial_score=-1,
    )
    update_scale_factor(module, observer, base_min, base_max, 1)
    return optimal_scale


def find_layernorm_weight_scale(module, observer, alpha=0.5, beta=2.0, num_scales=10, verbose=True):
    base_min = observer.min_val
    base_max = observer.max_val

    if verbose:
        print("_______________________________________________")

    def score_scale(scale_factor):
        update_scale_factor(module, observer, base_min, base_max, scale_factor)
        quantized_weight = observer.quantizer(module.weight)
        score = nrmse(observer.dequantizer(quantized_weight), module.weight)
        if verbose:
            print(f"scale_factor:{scale_factor}, nrmse:{score}")
        return score

    optimal_scale, _ = _line_search(
        scales=np.linspace(alpha, beta, num=num_scales),
        score_fn=score_scale,
        better_fn=lambda score, best: (best - score) > 0.1,
        initial_score=100,
    )

    if verbose:
        print("_______________________________________________")

    update_scale_factor(module, observer, base_min, base_max, 1)
    return optimal_scale


def optimal_factor_search(
    x,
    mdl,
    obs,
    is_weight=False,
    alpha=0.5,
    betta=2.0,
    itr=10,
    cosine_similarity=cosine_similarity,
):
    return find_activation_scale(
        inputs=x,
        module=mdl,
        observer=obs,
        alpha=alpha,
        beta=betta,
        num_scales=itr,
        metric=cosine_similarity,
    )


def optimal_factor_search2(x1, x2, mdl, obs, is_weight=False, alpha=0.5, betta=2.0, itr=10):
    return find_binary_input_scale(
        first_input=x1,
        second_input=x2,
        module=mdl,
        observer=obs,
        alpha=alpha,
        beta=betta,
        num_scales=itr,
    )


def optimal_factor_search_lnorm(mdl, obs, is_weight=False, alpha=0.5, betta=2.0, itr=10):
    return find_layernorm_weight_scale(
        module=mdl,
        observer=obs,
        alpha=alpha,
        beta=betta,
        num_scales=itr,
    )
