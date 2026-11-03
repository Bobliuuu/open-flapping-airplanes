#!/usr/bin/env python3
# ptxmerge.py main.ptx doom.ptx out.ptx : merge Doom's PTX module into the (non-rdc) main PTX so ptxas sees ONE whole-program module.
import re, sys
m = open(sys.argv[1]).read(); d = open(sys.argv[2]).read()
# main: drop the extern Doom entry-point declarations and the two Doom globals nvcc emitted as definitions
m = re.sub(r'\.extern \.func[^;]*?gpu_doom_\w+[^;]*;\n?', '', m)
m = re.sub(r'\.global \.align \d+ \.u64 I_VideoBuffer;\n', '', m)
m = re.sub(r'\.global \.align \d+ \.b8 colors\[1024\];\n', '', m)
mh = re.search(r'\.address_size \d+\n', m).end()
dh = re.search(r'\.address_size \d+\n', d).end()
open(sys.argv[3], 'w').write(m[:mh] + d[dh:] + m[mh:])
