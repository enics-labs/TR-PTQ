import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mrcp_quant.quantized_layers import QLayerNorm


sampled_layers = {}


def configure_plot_style():
    sns.set(style="whitegrid", font_scale=1.4)


def sample_layers(model, dataset, max_samples=12, sample_batches=3, var_name="weight"):
    sampled_layers.clear()

    for i, sample in enumerate(dataset):
        if i == sample_batches:
            break

        for j, m in enumerate(model.modules()):
            if isinstance(m, QLayerNorm):
                if j not in sampled_layers and len(sampled_layers) < max_samples:
                    if var_name == "weight":
                        sampled_layers[j] = m.w_obs.dequantizer(m.weight_integer).detach().clone().cpu()
        break

    print(f"Sampled {len(sampled_layers)} QLayerNorm.weight_integer matrices")
    return sampled_layers


def plot_distributions_icml(bins=200, normalize=False, after_layers=None, layers=None):
    layers = sampled_layers if layers is None else layers
    n = len(layers)
    if n == 0:
        print("No sampled layers to plot.")
        return

    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(6.5 * cols, 4.5 * rows))

    for idx, (j, weights) in enumerate(layers.items()):
        plt.subplot(rows, cols, idx + 1)
        original = weights.view(-1).float().numpy()

        if normalize and after_layers and j in after_layers:
            modified = after_layers[j].view(-1).float().numpy()
            sns.histplot(modified, bins=bins, kde=True, color="crimson", label="After", stat="density", element="step")
            sns.histplot(original, bins=bins, kde=True, color="skyblue", label="Before", stat="density", element="step")
            plt.legend()
        else:
            sns.histplot(original, bins=bins, kde=True, color="navy", stat="density", element="step")

        plt.title(f"Layer {j}", fontsize=16)
        plt.xlabel("Weight Value", fontsize=14)
        plt.ylabel("Density", fontsize=14)

    plt.tight_layout()
    if normalize:
        plt.suptitle("Weight Distribution Before vs After Normalization", fontsize=18, y=1.02)
    else:
        plt.suptitle("QLayerNorm Weight Distributions", fontsize=18, y=1.02)

    plt.show()


def run_mean(layer, s_factor, verbose, metric, metric_name):
    mean_scale = layer.s1 * s_factor
    layer.scale_input_normed, layer.zero_point_input_normed = layer.in_obs_normalize.calculate_qparams()

    x_q = layer.in_obs_normalize.quantizer(layer.x_test)
    x_q = (x_q - layer.zero_point_input_normed).int()

    xq_sum = x_q.sum(dim=-1, keepdim=True)
    xq_scaled_mean = (xq_sum * mean_scale) / x_q.shape[-1]
    xq_mean = xq_scaled_mean.trunc().int()

    xdq_mean = xq_mean * layer.in_obs_normalize.scale / mean_scale
    x_mean = layer.x_test.float().mean(dim=-1, keepdim=True)

    local_metric_res = metric(xdq_mean, x_mean)

    if verbose:
        print("-----------------------------")
        print(f"[SCALE: {s_factor}]xdq_mean {xdq_mean-x_mean}")
        print(f"[SCALE: {s_factor}] {metric_name} {local_metric_res}")

    return local_metric_res, x_q, xq_mean


def run_ver(m, s_factor, x_q, xq_mean, verbose, metric, metric_name):
    var_scale = m.s2 * s_factor

    q_e_2_x = xq_mean * xq_mean
    q_e_x_2 = (((x_q**2).sum(dim=-1, keepdim=True) * var_scale) / x_q.shape[-1]).trunc().int()
    q_var_x = q_e_x_2 - q_e_2_x

    qdq_x_var = ((m.scale_input_normed) ** 2) * (q_var_x.float()) / var_scale
    x_var = m.x_test.var(dim=-1, unbiased=False, keepdim=True)

    local_metric_res = metric(qdq_x_var, x_var)

    if verbose:
        print("-----------------------------")
        print(f"[SCALE: {s_factor}] xdq_mean {qdq_x_var}")
        print(f"[SCALE: {s_factor}] x_var {x_var}")
        print(f"[SCALE: {s_factor}] {metric_name} {local_metric_res}")

    return local_metric_res


def mean_scale_lineseach(m, metric, metric_name, min_max="min", alpha=0.5, betta=2):
    global_metric_res, _, _ = run_mean(m, 1.0, False, metric, metric_name)
    opt_mean_scale = 1.0

    for s1_factor in np.linspace(alpha, betta, num=10):
        local_metric_res, _, _ = run_mean(m, s1_factor, False, metric, metric_name)
        print(f"[SCALE:{s1_factor}] {global_metric_res} vs. {local_metric_res}")

        is_update = global_metric_res > local_metric_res if min_max == "min" else global_metric_res < local_metric_res
        if is_update:
            global_metric_res = local_metric_res
            opt_mean_scale = s1_factor

    print(f"[OPT SCALE:{opt_mean_scale}] {global_metric_res} metric={metric_name}")
    local_metric_res, x_q, xq_mean = run_mean(m, opt_mean_scale, False, metric, metric_name)

    return local_metric_res, x_q, xq_mean, opt_mean_scale


def var_scale_linesearch(m, x_q, xq_mean, metric, metric_name, min_max="min", alpha=0.5, betta=2):
    global_metric_res = run_ver(m, 1.0, x_q, xq_mean, False, metric, metric_name)
    opt_var_scale = 1.0

    for s2_factor in np.linspace(alpha, betta, num=10):
        local_metric_res = run_ver(m, s2_factor, x_q, xq_mean, False, metric, metric_name)
        print(f"[SCALE:{s2_factor}] {global_metric_res} vs. {local_metric_res}")

        is_update = global_metric_res > local_metric_res if min_max == "min" else global_metric_res < local_metric_res
        if is_update:
            global_metric_res = local_metric_res
            opt_var_scale = s2_factor

    print(f"[OPT SCALE:{opt_var_scale}] {global_metric_res} metric={metric_name}")
    run_ver(m, opt_var_scale, x_q, xq_mean, False, metric, metric_name)

    return opt_var_scale
