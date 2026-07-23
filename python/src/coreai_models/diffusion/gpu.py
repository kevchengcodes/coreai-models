# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Stateless GPU export for diffusion components.

Simpler than the LLM export path: no KV cache, no dynamic shapes.
Each component is a single fixed-shape forward pass.

When `externalize=True`, composite ops (SDPA, RoPE, RMSNorm) are emitted
as named MLIR ops — enables compiler-level fusion (flash attention, etc.).
"""

import logging

import coreai_torch
import coreai_torch.composite_ops
import torch
from coreai.authoring import AIProgram

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO

logger = logging.getLogger(__name__)

_DIFFUSION_EXTERNALIZE_SPECS = [
    coreai_torch.ExternalizeSpec(
        target_class=coreai_torch.composite_ops.SDPA,
        composite_op_name="scaled_dot_product_attention",
        composite_attrs=["scale", "is_causal", "window_size"],
    ),
]


def export_stateless(
    wrapper: torch.nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
    externalize: bool = False,
    dynamic_shapes: dict | None = None,
    static_shape_configs: dict[str, dict[str, tuple[int, ...]]] | None = None,
) -> AIProgram:
    """Export a stateless model to a CoreAI AIProgram.

    Args:
        wrapper: A thin torch.nn.Module that wraps a HF model component.
        dummy_inputs: Reference input tensors (positional) for tracing.
        input_names: Names for the exported model's inputs.
        output_names: Names for the exported model's outputs.
        include_debug_info: When True, the converter runs in ``DEBUG`` mode and embeds debug
            information in the exported ``.aimodel``. Defaults to ``RELEASE`` mode,
            which embeds minimum debug information and makes the exported asset smaller.
        externalize: If True, emit composite ops (SDPA, RoPE) as named
            MLIR ops for compiler-level fusion.
        dynamic_shapes: Optional dynamic shape specs for torch.export.
        static_shape_configs: Optional shape specialization configs. When
            provided, the compiler generates optimized function variants for
            each config entry (e.g. main_full, main_half). Requires
            dynamic_shapes to be set.

    Returns:
        An optimized AIProgram ready for saving/compilation.
    """
    wrapper.eval()

    def export_fn(module: torch.nn.Module) -> torch.export.ExportedProgram:
        with torch.no_grad():
            exported = torch.export.export(module, args=dummy_inputs, dynamic_shapes=dynamic_shapes)
        coreai_decomp_table = coreai_torch.get_decomp_table()
        decomposed: torch.export.ExportedProgram = exported.run_decompositions(coreai_decomp_table)
        return decomposed

    converter = coreai_torch.TorchConverter(
        mode=(
            coreai_torch.TorchConverter.Mode.DEBUG
            if include_debug_info
            else coreai_torch.TorchConverter.Mode.RELEASE
        )
    )
    converter.add_pytorch_module(
        wrapper,
        export_fn=export_fn,
        input_names=input_names,
        output_names=output_names,
        externalize_modules=_DIFFUSION_EXTERNALIZE_SPECS if externalize else None,
    )
    if externalize:
        from coreai_models.export.mlir_ops import register_custom_torch_lowering

        register_custom_torch_lowering(converter)
    program = converter.to_coreai()

    if static_shape_configs is not None:
        logger.info(f"Setting static shape configs: {list(static_shape_configs.keys())}")
        program.set_static_shape_config("main", static_shape_configs)

    program.optimize()
    return program


def export_multifunction(
    functions: list[tuple[str, torch.nn.Module, tuple[torch.Tensor, ...]]],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
) -> AIProgram:
    """Export multiple function variants into a single .aimodel with shared weights.

    Each function is a separate static trace of the same model at different input
    shapes. Weights are shared automatically — disk size equals one copy.

    Args:
        functions: List of (entrypoint_name, wrapper, dummy_inputs) tuples.
            The wrapper should be the same nn.Module instance (or share weights).
        input_names: Names for the exported model's inputs (same for all functions).
        output_names: Names for the exported model's outputs (same for all functions).

    Returns:
        An optimized AIProgram with multiple named functions.
    """
    converter = coreai_torch.TorchConverter()
    coreai_decomp_table = coreai_torch.get_decomp_table()

    for name, wrapper, dummy_inputs in functions:
        wrapper.eval()
        logger.info(f"Tracing function '{name}' (seq dims: {[t.shape for t in dummy_inputs[:2]]})")

        with torch.no_grad():
            exported = torch.export.export(wrapper, args=dummy_inputs)
        decomposed = exported.run_decompositions(coreai_decomp_table)

        converter.add_exported_program(
            decomposed,
            input_names=input_names,
            output_names=output_names,
            entrypoint_name=name,
        )

    program = converter.to_coreai()
    program.optimize()
    return program
