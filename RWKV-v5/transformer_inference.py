
from transformers import AutoConfig, AutoModel
from transformers import GPT2Config, GPT2LMHeadModel

# Create a custom configuration
config = AutoConfig.from_pretrained(
    "bert-base-uncased",  # Use a base model config as a template
    hidden_size=256,      # Equivalent to n_embd
    num_hidden_layers=6,  # Number of transformer layers
    num_attention_heads=8 # Ensure heads fit into hidden_size
)

# Initialize the model with the custom configuration
model = AutoModel.from_config(config)


# Create a GPT-like configuration
config2 = GPT2Config(
    n_embd=256,            # Embedding size
    n_layer=6,             # Number of layers
    n_head=8,              # Number of attention heads
    vocab_size=50257,      # Vocabulary size
    max_position_embeddings=512 # Maximum sequence length
)

# Initialize the GPT model with the custom configuration
model2 = GPT2LMHeadModel(config2)


