from lab.config import Registry

# Whole model families, chosen by `model.type`. Each config class has
# `build(vocab_size) -> nn.Module`.
MODELS = Registry("model")
