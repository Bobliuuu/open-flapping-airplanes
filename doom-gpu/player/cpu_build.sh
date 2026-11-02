#!/usr/bin/env bash
# Build the same Doom engine (doomgeneric) and scripted player for the CPU, as a shared library cpu_run.py drives one tic at a time.
# Uses the checkout and WAD that setup_box.sh puts next to build_doom.sh (doom-gpu/doomgeneric, doom-gpu/doom1.wad).
#   ./cpu_build.sh            -> ./libdoom_cpu.so
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; SRC="$HERE/../doomgeneric/doomgeneric"; B="$HERE/build_cpu"
[ -d "$SRC" ] || { echo "no $SRC: run doom-gpu/setup_box.sh first (or clone doomgeneric there)"; exit 1; }
rm -rf "$B" && mkdir -p "$B" && cp -r "$SRC"/* "$B"/
# the player fills in the command at the end of G_BuildTiccmd, the same hook build_doom.sh adds for the GPU
python3 - "$B/g_game.c" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "    // special buttons\n"
assert s.count(old) == 1
s = s.replace(old, "    { extern void player_ticcmd(ticcmd_t*); player_ticcmd(cmd); }\n" + old, 1)
open(p, "w").write(s)
PY
FILES="dummy am_map doomdef doomstat dstrings d_event d_items d_iwad d_loop d_main d_mode d_net f_finale f_wipe g_game hu_lib hu_stuff info i_cdmus i_endoom i_joystick i_scale i_sound i_system i_timer memio m_argv m_bbox m_cheat m_config m_controls m_fixed m_menu m_misc m_random p_ceilng p_doors p_enemy p_floor p_inter p_lights p_map p_maputl p_mobj p_plats p_pspr p_saveg p_setup p_sight p_spec p_switch p_telept p_tick p_user r_bsp r_data r_draw r_main r_plane r_segs r_sky r_things sha1 sounds statdump st_lib st_stuff s_sound tables v_video wi_stuff w_checksum w_file w_main w_wad z_zone w_file_stdc i_input i_video doomgeneric"
cd "$B"
gcc -O2 -fPIC -shared -w -DNORMALUNIX -DLINUX -D_DEFAULT_SOURCE -DDOOMGENERIC_RESX=320 -DDOOMGENERIC_RESY=200 -I. -I"$HERE" \
  $(for f in $FILES; do echo "$f.c"; done) "$HERE/cpu_platform.c" "$HERE/player.c" -o "$HERE/libdoom_cpu.so" -lm
ls -la "$HERE/libdoom_cpu.so"
