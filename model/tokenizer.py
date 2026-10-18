"""Character-level tokenizer for arithmetic expressions.

The vocabulary is fixed because the arithmetic dataset only contains a small
set of characters.  The interface follows minbpe's ``encode``/``decode``
convention, while ``train`` is intentionally unnecessary.
"""


class ArithmeticTokenizer:
    """Encode digits, arithmetic operators, parentheses, ``=`` and ``#``."""

    # Keep ids deterministic so the same tokenizer can always be used by GPT.
    # ``#`` is the end-of-text token.
    tokens = "0123456789+-*/().=#"

    def __init__(self):
        self.stoi = {token: i for i, token in enumerate(self.tokens)}
        self.itos = {i: token for token, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, text):
        """Convert an arithmetic expression into a list of token ids."""
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"unsupported character(s): {unknown}")
        return [self.stoi[token] for token in text]

    def decode(self, ids):
        """Convert token ids back into an arithmetic expression."""
        try:
            return "".join(self.itos[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unsupported token id: {error.args[0]}") from None


Tokenizer = ArithmeticTokenizer

