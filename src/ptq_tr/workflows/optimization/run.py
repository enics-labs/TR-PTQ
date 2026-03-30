"""Optimization workflow entrypoints."""

import numpy as np
import torch

from ptq_tr.metrics.similarity import cosine_similarity, nrmse
from ptq_tr.quantization.modules.q_layernorm import QLayerNorm
from ptq_tr.quantization.modules.quant_linear import QuantizedLinear
from ptq_tr.quantization.modules.quant_matmul import QuantizedMatmul


def calc_cosine(layer, original_output, x, alpha=0.9, beta=1.1, num_scales=10):
    """
    Optimizes the scale factors for a layer normalization module using cosine similarity.

    Args:
        layer: The LayerNorm module whose scale factors will be optimized.
        original_output: A reference output for comparison.
        x: Input tensor to compute the cosine similarity.
        alpha: Lower bound for scale factor search.
        beta: Upper bound for scale factor search.
        num_scales: Number of scale factors to test within the [alpha, beta] range.

    Returns:
        None. Updates the layer with the optimal scale factors for weights and inputs.
    """
    optimal_weight_scale = find_optimal_scale_factor(
        layer.update_w_scale_factor,
        layer.forward_pass,
        x,
        original_output,
        alpha,
        beta,
        num_scales,
    )

    optimal_in_scale = find_optimal_scale_factor(
        layer.update_in_scale_factor,
        layer.forward_pass,
        x,
        original_output,
        alpha,
        beta,
        num_scales,
    )

    layer.set_optimal_w_scale_factor(optimal_weight_scale)
    layer.set_optimal_in_scale_factor(optimal_in_scale)


def find_optimal_scale_factor(update_scale_fn, forward_pass, x, original_output, alpha, beta, num_scales):
    """
    Finds the optimal scale factor that maximizes cosine similarity.

    Args:
        update_scale_fn: Function to update the scale factor.
        forward_pass: The forward pass function of the layer being optimized.
        x: Input tensor.
        original_output: The reference output for comparison.
        alpha: Lower bound for scale factor search.
        beta: Upper bound for scale factor search.
        num_scales: Number of scale factors to test.

    Returns:
        optimal_scale (float): The scale factor that gives the highest cosine similarity.
    """
    best_cosine = -float("inf")
    optimal_scale = 1.0

    for scale_factor in np.linspace(alpha, beta, num=num_scales):
        update_scale_fn(scale_factor)
        approx_output = forward_pass(x)
        cosine_sim = cosine_similarity(original_output, approx_output)

        if cosine_sim > best_cosine:
            best_cosine = cosine_sim
            optimal_scale = scale_factor

    return optimal_scale


def update_scale_factor(mdl, obs, cur_min, cur_max, search_scale):
    obs.min_val = cur_min * search_scale
    obs.max_val = cur_max * search_scale
    obs.calculate_qparams()
    mdl.is_weights_quantized = False


def optimal_factor_search(x, mdl, obs, is_weight=False, alpha=0.5, betta=2.0, itr=10, similarity_fn=cosine_similarity):
    cur_min = obs.min_val
    cur_max = obs.max_val

    optimal_cosine = -1
    optimal_scale = 1

    for search_scale in np.linspace(alpha, betta, num=itr):
        update_scale_factor(mdl, obs, cur_min, cur_max, search_scale)

        approx_output = mdl.forward_pass(x)
        original_output = mdl.float_forward_pass(x)
        cosine_sim = similarity_fn(original_output, approx_output)

        if cosine_sim > optimal_cosine:
            optimal_cosine = cosine_sim
            optimal_scale = search_scale

    update_scale_factor(mdl, obs, cur_min, cur_max, 1)

    return optimal_scale


def optimal_factor_search2(x1, x2, mdl, obs, is_weight=False, alpha=0.5, betta=2.0, itr=10):
    cur_min = obs.min_val
    cur_max = obs.max_val

    optimal_cosine = -1
    optimal_scale = 1

    for search_scale in np.linspace(alpha, betta, num=itr):
        update_scale_factor(mdl, obs, cur_min, cur_max, search_scale)

        approx_output = mdl.forward_pass(x1, x2)
        original_output = mdl.float_forward_pass(x1, x2)
        cosine_sim = cosine_similarity(original_output, approx_output)

        if cosine_sim > optimal_cosine:
            optimal_cosine = cosine_sim
            optimal_scale = search_scale

    update_scale_factor(mdl, obs, cur_min, cur_max, 1)

    return optimal_scale


def optimal_factor_search_lnorm(mdl, obs, is_weight=False, alpha=0.5, betta=2.0, itr=10):
    cur_min = obs.min_val
    cur_max = obs.max_val

    optimal_cosine = 100
    optimal_scale = 1

    print("_______________________________________________")
    for search_scale in np.linspace(alpha, betta, num=itr):
        update_scale_factor(mdl, obs, cur_min, cur_max, search_scale)
        weight_integer = obs.quantizer(mdl.weight)

        cosine_sim = nrmse(obs.dequantizer(weight_integer), mdl.weight)

        print(f"search_scale:{search_scale}, optimal_cosine:{optimal_cosine}, cosine_sim:{cosine_sim}")
        if (optimal_cosine - cosine_sim) > 0.1:
            optimal_cosine = cosine_sim
            optimal_scale = search_scale
    print("_______________________________________________")

    update_scale_factor(mdl, obs, cur_min, cur_max, 1)
    print("-----------------")
    return optimal_scale


def optimize_model(
    model,
    scale_opt_image_list,
    q_module_list,
    alpha=0.5,
    betta=2.0,
    itr=10,
    cosine_threshold=0.01,
    lnorm_threshold=0.03,
    lnorm_itr=50,
):
    print("optimizing model scale-factors...")
    with torch.no_grad():
        model.set_scale_opt()
        print()
        for i, image in enumerate(scale_opt_image_list):
            print(i, end="|")
            _ = model(image)
            for m in model.modules():
                if not type(m) in q_module_list:
                    continue
                if type(m) in [QuantizedLinear]:
                    x = m.opt_input
                    dq_output = m.forward_pass(x)
                    output = m.float_forward_pass(x)
                    cs = cosine_similarity(output, dq_output)
                    if cs < (1 - cosine_threshold):
                        print("---------------")
                        print("QuantizedLinear: cosine similarity:", f"{float(cs):.2f}")
                        search_scale_w = optimal_factor_search(
                            x,
                            m,
                            m.w_obs,
                            is_weight=True,
                            alpha=alpha,
                            betta=betta,
                            itr=itr,
                        )
                        update_scale_factor(m, m.w_obs, m.w_obs.min_val, m.w_obs.max_val, search_scale_w)
                        search_scale_in = optimal_factor_search(
                            x,
                            m,
                            m.in_obs,
                            is_weight=True,
                            alpha=alpha,
                            betta=betta,
                            itr=itr,
                        )
                        update_scale_factor(m, m.in_obs, m.in_obs.min_val, m.in_obs.max_val, search_scale_in)

                if type(m) in [QuantizedMatmul]:
                    x1 = m.opt_input1
                    x2 = m.opt_input2
                    dq_output = m.forward_pass(x1, x2)
                    output = m.float_forward_pass(x1, x2)
                    cs = cosine_similarity(output, dq_output)
                    if cs < (1 - cosine_threshold):
                        print("---------------")
                        print("QuantizedMatmul: cosine similarity:", f"{float(cs):.2f}")
                        search_scale_in1 = optimal_factor_search2(
                            x1,
                            x2,
                            m,
                            m.in1_obs,
                            is_weight=False,
                            alpha=alpha,
                            betta=betta,
                            itr=itr,
                        )
                        update_scale_factor(m, m.in1_obs, m.in1_obs.min_val, m.in1_obs.max_val, search_scale_in1)
                        search_scale_in2 = optimal_factor_search2(
                            x1,
                            x2,
                            m,
                            m.in2_obs,
                            is_weight=False,
                            alpha=alpha,
                            betta=betta,
                            itr=itr,
                        )
                        update_scale_factor(m, m.in2_obs, m.in2_obs.min_val, m.in2_obs.max_val, search_scale_in2)

                if type(m) in [QLayerNorm] and i == 0:
                    x = m.opt_input
                    _ = m.forward_pass(x)
                    cs = nrmse(m.w_obs.dequantizer(m.weight_integer), m.weight)

                    mean_val = m.normalize(x)
                    mean_val_f = m.float_foward_pass_normalize(x)
                    dq_output = m.betta_gamma_forward_pass(mean_val)
                    dq_output_f = m.float_betta_gamma_forward_pass(mean_val)

                    if cs > lnorm_threshold:
                        search_scale_w = optimal_factor_search_lnorm(
                            m,
                            m.w_obs,
                            is_weight=False,
                            alpha=alpha,
                            betta=betta,
                            itr=lnorm_itr,
                        )
                        print("search_scale_w", search_scale_w)
                        update_scale_factor(m, m.w_obs, m.w_obs.min_val, m.w_obs.max_val, search_scale_w)

                    print("---------------")

        model.unset_scale_opt()
        print()


def run_optimization(*args, **kwargs):
    if "model" not in kwargs or "scale_opt_image_list" not in kwargs or "q_module_list" not in kwargs:
        raise NotImplementedError("Optimization workflow wiring has not been migrated yet.")
    return optimize_model(kwargs["model"], kwargs["scale_opt_image_list"], kwargs["q_module_list"])
