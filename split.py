from base import Tokenizer, get_stats, merge
import regex as re

GPT_4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):

    def __init__(self, pattern = None):
        super().__init__()
        self.pattern = GPT_4_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        regexed = re.findall(self.compiled_pattern, text)
        ids = [list(chunk.encode("utf-8")) for chunk in regexed]
        merges = {}  # (int, int) -> int
        for i in range(num_merges):
            stats = {}
            for chunk in ids:
                get_stats(chunk, stats) # go through chunks and add up pair counts
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk, pair, idx) for chunk in ids] # do merge within each chunk
            merges[pair] = idx
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx}")
        self.merges = merges
        self.vocab = self._build_vocab()

    def encode_processed(self, text):
        utf_bytes = text.encode("utf-8")
        ids = list(utf_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        regexed = re.findall(self.compiled_pattern, text)
        ids = []
        for piece in regexed:
            ids += self.encode_processed(piece) # encode each chunk independently and concatenate results
        return ids

    def decode(self, ids):
        raw_bytes = b"".join(self.vocab[id] for id in ids)
        text = raw_bytes.decode("utf-8", errors="replace")
        return text


with open("taylorswift.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = RegexTokenizer()
tokenizer.train(text, 276)
ids = tokenizer.encode("hello world!!!? (안녕하세요!) lol123 😉")
print(tokenizer.decode(ids))
