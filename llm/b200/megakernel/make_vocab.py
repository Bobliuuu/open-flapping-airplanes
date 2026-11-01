#!/usr/bin/env python3
"""Writes the cl100k detokenizer table (flat binary: u32 n, u32 offsets[n+1], bytes) and the prompt token ids for ofa.
usage (on the box, with tiktoken):  ~/venv-baseline/bin/python make_vocab.py ~/data/cl100k_vocab.bin ~/data/prompt_ids.txt "Once upon a time" """
import sys, struct, tiktoken
out, prompt_out, prompt = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "Once upon a time"
enc = tiktoken.get_encoding("cl100k_base")
N = 100352
offs, buf = [0], bytearray()
for i in range(N):
    try: b = enc.decode_single_token_bytes(i)
    except Exception: b = b""
    if i == 100257: b = b"\n\n"                       # <|endoftext|> separates stories
    buf += b; offs.append(len(buf))
with open(out, "wb") as f: f.write(struct.pack("<I", N)); f.write(struct.pack("<%dI" % (N + 1), *offs)); f.write(bytes(buf))
ids = enc.encode(prompt)
open(prompt_out, "w").write(" ".join(map(str, ids)) + "\n")
print("vocab", N, "tokens,", len(buf), "bytes; prompt", prompt, "->", ids)
