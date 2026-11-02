# doom-gpu

Builds the C Doom engine (doomgeneric) as a relocatable GPU object with clang's NVPTX backend. The engine source and the WAD are not stored here; `setup_box.sh` fetches them.

| File | Role |
|---|---|
| `setup_box.sh` | One-time: installs clang, clones doomgeneric, fetches the shareware WAD (md5 checked), builds |
| `build_doom.sh` | Compiles engine + tinylibc + glue to `doom.o`, applying the PTX-compatibility patches. `ARCH=sm_100 ./build_doom.sh` builds for a B200 (default `sm_80`) |
| `tinylibc/`, `tinylibc.c` | Minimal libc headers/implementation for device code |
| `fix_varargs.py` | LLVM IR pass lowering varargs (clang 14 cannot select `va_start` on NVPTX) |
| `gpu_glue.c`, `gpu_par.c` | Engine entry points; queued parallel column/span drawing for 255 workers |
| `doomtest.cu` | Standalone test: run Doom in one GPU thread for N frames and dump a frame |
| `player/player.c` | The scripted player: builds the player's command every game tic (hooked into `G_BuildTiccmd` by `build_doom.sh`) |
| `player/route_e1m1.h`, `player/plan_route.py` | The E1M1 route it follows, planned from the WAD's map data |
| `player/cpu_build.sh`, `player/cpu_platform.c`, `player/cpu_run.py` | The same engine and player on the CPU, to check what the player does |

## The player

Doom is played by a scripted player, without cheats. Every game tic the engine asks it for the player's command, the same command the
keyboard would produce, and it fills it in from the game state:

- **Fight:** it picks the nearest live monster it can see within 1,400 map units (the engine's own sight check), turns toward it at
  most about 6 degrees per tic, fires when its aim is within 6 degrees, and side-steps while firing. A monster it cannot kill within
  4 seconds is left alone for 5.
- **Move:** otherwise it runs along a route through E1M1, steering at a point 56 map units ahead on the current leg. After a fight it
  rejoins the route at the nearest leg ahead. If it stops making progress it side-steps and rejoins.
- **Doors and exit:** it presses USE at the doors on the route only when that opens them (USE on an opening door would close it again),
  and flips the exit switch after the last waypoint.

`plan_route.py` plans the route from the map data in `doom1.wad`: grid A* over floor that does not hurt, with room for the player,
away from walls and solid objects, never up a step higher than 24. The megakernel runs one game tic per frame at 35 Hz, so the game
is the same tic for tic on the CPU. At skill 3 (the demo's) the CPU check gives 6 of 6 monsters killed, no deaths, and the level
finished at 42.9 s:

```bash
cd doom-gpu/player && ./cpu_build.sh && python3 cpu_run.py --seconds 56 --skill 3     # JSON: kills, items, deaths, level_finished_at_s
python3 cpu_run.py --mp4 play.mp4                                                    # also a video of the game (imageio[ffmpeg])
python3 plan_route.py --png route.png                                                # re-plan the route (numpy; opencv for the picture)
```
