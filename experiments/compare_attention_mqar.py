"""Run three hand-written variants using the shared base MQAR settings.

From the repository root:
    python -m zoology.launch experiments/compare_attention_mqar.py
"""

from zoology.config import ModuleConfig

from experiments.linear_attention_mqar import config as base_config


variants = [
    ("basic-la", "attention_variants.linear_attention.CausalLinearAttention"),
    ("delta-net", "attention_variants.delta_net.DeltaNet"),
    ("gated-delta-net", "attention_variants.gated_delta_net.GatedDeltaNet"),
]

configs = []
for run_name, module_path in variants:
    # Independent copies keep every other setting identical to the base experiment.
    config = base_config.model_copy(deep=True)
    config.run_id = f"{run_name}-seed-{config.seed}"
    config.model.sequence_mixer = ModuleConfig(name=module_path)
    configs.append(config)
