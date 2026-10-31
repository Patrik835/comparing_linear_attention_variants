"""Small integration run

Run from the repository with conda env active:
python -m zoology.launch experiments/linear_attention_mqar.py
"""

from zoology.config import DataConfig, ModelConfig, ModuleConfig, TrainConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig


config = TrainConfig(
    logger=LoggerConfig(
        project_name="linear-attention-mqar",
        entity="patrik835",
    ),
    max_epochs=10,
    data=DataConfig(
        batch_size=4,
        train_configs=[
            MQARConfig(
                num_examples=128,
                vocab_size=256,
                input_seq_len=64,
                num_kv_pairs=4,
            )
        ],
        test_configs=[
            MQARConfig(
                num_examples=64,
                vocab_size=256,
                input_seq_len=64,
                num_kv_pairs=4,
            )
        ],
    ),
    model=ModelConfig(
        d_model=32,
        n_layers=2,
        vocab_size=256,
        max_position_embeddings=64,
        sequence_mixer=ModuleConfig(
            name="attention_variants.linear_attention.CausalLinearAttention",
        ),
    ),
)

configs = [config]
