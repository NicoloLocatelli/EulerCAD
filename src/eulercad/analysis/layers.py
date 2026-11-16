"""The hidden states of every layer of a GraphGPT-based encoder, for analyses and frozen probes.

The encoder runs once; forward hooks keep the rows before the first layer, the output of each
layer, and the output of the final normalization, which is what `GraphGPTModel.encode`
returns. The hooks are removed before returning, whatever happens.
"""

import torch

from eulercad.model.model import GraphGPTModel


def layer_names(model: GraphGPTModel) -> list[str]:
    """The names `hidden_states` gives, in depth order.

    Returns:
        `["rows", "layer_1", ..., "layer_L", "final_norm"]`.
    """
    return ["rows", *(f"layer_{index}" for index in range(1, len(model.backbone.layers) + 1)), "final_norm"]


def hidden_states(
    model: GraphGPTModel, input_ids: torch.Tensor, mask: torch.Tensor, causal: bool = True
) -> dict[str, torch.Tensor]:
    """Run the model's encoder and keep the rows before the first layer, after each layer, and after the final norm.

    Args:
        model: the network; only its encoder runs.
        input_ids: `(B, T, F)` right-padded token rows.
        mask: `(B, T)` bool, true on real rows.
        causal: whether a row reads only the rows before it, as in pretraining.

    Returns:
        `{"rows", "layer_1", ..., "layer_L", "final_norm"}`, each `(B, T, H)`.

    Raises:
        ValueError: If the model mixes its layers with attention residuals, whose layers are
            not called one after the other.
    """
    backbone = model.backbone
    if backbone.depth_attention is not None:
        raise ValueError("models with attention residuals are not supported")
    states: dict[str, torch.Tensor] = {}
    handles = [backbone.layers[0].register_forward_pre_hook(lambda _, inputs: states.update(rows=inputs[0]))]
    for index, layer in enumerate(backbone.layers, start=1):
        name = f"layer_{index}"
        handles.append(layer.register_forward_hook(lambda _, __, output, name=name: states.update({name: output[0]})))
    handles.append(backbone.norm.register_forward_hook(lambda _, __, output: states.update(final_norm=output)))
    try:
        model.encode(input_ids, mask, causal)
    finally:
        for handle in handles:
            handle.remove()
    return states
