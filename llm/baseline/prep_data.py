"""Tokenize TinyStories with cl100k_base -> uint32 binaries. Stories are separated by <|endoftext|> (id 100257);
the source files already use <|endoftext|> as the separator, which tiktoken maps to the special id."""
import sys, os, numpy as np, tiktoken
enc = tiktoken.get_encoding("cl100k_base")
EOT = enc.eot_token
def run(src, dst, truncate_partial):
    txt = open(src, encoding="utf-8", errors="ignore").read()
    parts = txt.split("<|endoftext|>")
    if truncate_partial: parts = parts[:-1]  # last story of a byte-range prefix may be cut off
    out = []
    for i in range(0, len(parts), 20000):
        for ids in enc.encode_batch([p.strip() for p in parts[i:i+20000]], num_threads=16):
            out.append(np.array(ids + [EOT], dtype=np.uint32))
    a = np.concatenate(out); a.tofile(dst)
    print(dst, "stories", len(parts), "tokens", a.size, "max id", int(a.max()), flush=True)
d = os.path.expanduser("~/data")
run(f"{d}/TinyStories-valid.txt", f"{d}/tinystories_cl100k_valid.u32", False)
run(f"{d}/TinyStories-train-prefix.txt", f"{d}/tinystories_cl100k_train.u32", True)
