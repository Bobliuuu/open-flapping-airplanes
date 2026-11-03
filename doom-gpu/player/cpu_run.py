"""Run the scripted player on the CPU build (cpu_build.sh) and report what it did, second by second.

  python3 cpu_run.py [--seconds 56] [--skill 3] [--wad ../doom1.wad] [--mp4 play.mp4]    (--mp4 needs imageio[ffmpeg] and numpy)

The megakernel runs one game tic per frame at 35 frames per second, so second s here is second s of the GPU run's game.
"""
import argparse
import ctypes
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=56)
ap.add_argument("--skill", type=int, default=3)
ap.add_argument("--wad", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "doom1.wad"))
ap.add_argument("--mp4", default="")
a = ap.parse_args()

lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libdoom_cpu.so"))
lib.cpu_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.cpu_screen.restype = ctypes.POINTER(ctypes.c_uint32)
S = (ctypes.c_int * 11)()
saved_stdout = os.dup(1)                            # the engine prints its start-up log to stdout: keep it out of the report
os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
lib.cpu_init(a.wad.encode(), a.skill)

writer = None
if a.mp4:
    import imageio
    import numpy as np
    writer = imageio.get_writer(a.mp4, fps=35, codec="libx264", quality=8, macro_block_size=8)

per_second, deaths, was_dead, finished_at = [], 0, False, None
for t in range(int(a.seconds * 35)):
    lib.cpu_tic()
    lib.cpu_state(S)
    if S[7] and not was_dead:
        deaths += 1
    was_dead = bool(S[7])
    if S[0] == 1 and finished_at is None:
        finished_at = round((t + 1) / 35, 2)
    if t % 35 == 34:
        per_second.append({"s": (t + 1) // 35, "kills": S[3], "health": S[1], "waypoint": S[10], "x": S[8], "y": S[9]})
    if writer:
        f = np.ctypeslib.as_array(lib.cpu_screen(), shape=(200 * 320,)).view(np.uint8).reshape(200, 320, 4)[:, :, [2, 1, 0]]
        writer.append_data(np.repeat(np.repeat(f, 2, 0), 2, 1))
if writer:
    writer.close()
ctypes.CDLL(None).fflush(None)
os.dup2(saved_stdout, 1)
lib.cpu_state(S)
print(json.dumps({"skill": a.skill, "seconds": a.seconds, "kills": S[3], "monsters": S[4], "items": S[5], "secrets": S[6],
                  "deaths": deaths, "health": S[1], "armor": S[2], "level_finished_at_s": finished_at, "per_second": per_second}))
