from lab.config import Registry

# Whole model families, chosen by `model.type`. Each is a module whose constructor is its
# config, with the vocabulary size passed in from `tokenizer.vocab_size`.
MODELS = Registry("model", provided=("vocab_size",))
