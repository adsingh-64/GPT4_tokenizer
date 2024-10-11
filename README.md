# GPT4_tokenizer
We build a mini version of the GPT4 tokenizer (tiktoken) from scratch using BPE and the GPT 4 regex pattern. This is an implementation of the exercise flow given in https://github.com/karpathy/minbpe/blob/master/exercise.md. The BPE is trained on the Taylor Swift Wikipedia page (taylorswift.txt). The vocab size is expanded to 276, and vocab shows the 20 merges after the 256 initial bytes. The key methods implemented in the regex tokenizer in split.py are training the model vocab via BPE,
encoding strings to tokens, and decoding tokens to strings.
