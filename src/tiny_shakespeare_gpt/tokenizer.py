"""
Byte-Pair Encoding (BPE) tokenizer for the GPT model.
"""
import tiktoken

class BPETokenizer:
    """
    A Byte-Pair Encoding (BPE) tokenizer wrapping OpenAI's tiktoken.
    Defaults to 'gpt2' encoding which is standard for from-scratch implementations,
    but can use 'cl100k_base' (GPT-4) as well.
    """
    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integer tokens."""
        return self.encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Convert a list of integer tokens back to a string."""
        return self.encoding.decode(tokens)
