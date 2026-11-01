#!/usr/bin/env bash
# Compile the Doom engine (doomgeneric) + tinylibc + glue to a relocatable GPU object (doom.o) with clang's NVPTX backend.
set -e
cd "$(dirname "$0")"
SRC=doomgeneric/doomgeneric
RES=$(clang -print-resource-dir)
FILES="dummy am_map doomdef doomstat dstrings d_event d_items d_iwad d_loop d_main d_mode d_net f_finale f_wipe g_game hu_lib hu_stuff info i_cdmus i_endoom i_joystick i_scale i_sound i_system i_timer memio m_argv m_bbox m_cheat m_config m_controls m_fixed m_menu m_misc m_random p_ceilng p_doors p_enemy p_floor p_inter p_lights p_map p_maputl p_mobj p_plats p_pspr p_saveg p_setup p_sight p_spec p_switch p_telept p_tick p_user r_bsp r_data r_draw r_main r_plane r_segs r_sky r_things sha1 sounds statdump st_lib st_stuff s_sound tables v_video wi_stuff w_checksum w_file w_main w_wad z_zone w_file_stdc i_input i_video doomgeneric"
CF="--target=nvptx64-nvidia-cuda -march=sm_80 -O2 -ffreestanding -fno-stack-protector -fno-jump-tables -w -nostdinc -isystem tinylibc -isystem $RES/include -I$SRC -DNORMALUNIX -DLINUX -D_DEFAULT_SOURCE -DDOOMGENERIC_RESX=320 -DDOOMGENERIC_RESY=200"
rm -rf src_gpu && cp -r $SRC src_gpu && SRC=src_gpu
# sounds are disabled: self-referential initializers (S_sfx links into itself) cannot be emitted as PTX globals
sed -i -E 's/&S_sfx\[[A-Za-z0-9_]+\]/NULL/g' src_gpu/sounds.c
# LLVM turns these constant string tables into self-referential relative lookup tables that PTX cannot emit; volatile blocks it
sed -i -E 's/static char \*(detailNames|msgNames)\[2\]/static char * volatile \1[2]/' src_gpu/m_menu.c
# PTX cannot statically initialise a pointer to an array element; set these three at startup instead
sed -i -E 's/^const fixed_t \*finecosine = &finesine\[FINEANGLES\/4\];/const fixed_t *finecosine;/' src_gpu/tables.c
echo 'void gpu_init_tables(void) { finecosine = &finesine[FINEANGLES/4]; }' >> src_gpu/tables.c
sed -i -E 's/^static boolean \*mousebuttons = &mousearray\[1\];.*/static boolean *mousebuttons;/; s/^static boolean \*joybuttons = &joyarray\[1\];.*/static boolean *joybuttons;/' src_gpu/g_game.c
echo 'void gpu_init_game_ptrs(void) { mousebuttons = &mousearray[1]; joybuttons = &joyarray[1]; }' >> src_gpu/g_game.c

# ---- parallel drawing hooks (see gpu_par.c)
python3 - <<'PYEOF'
import re
def patch(path, fn):
    s = open(path).read(); s2 = fn(s); assert s2 != s, path; open(path, 'w').write(s2)
decl = 'extern volatile int gpu_par_active; void gpu_flush(void); void gpu_q_col(byte*, const byte*, const byte*, int, int, int, int); void gpu_q_span(byte*, const byte*, const byte*, unsigned, unsigned, int);\n'
def rdraw(s):
    s = s.replace('#include "r_local.h"', '#include "r_local.h"\n' + decl, 1) if '#include "r_local.h"' in s else decl + s
    s = s.replace('    frac = dc_texturemid + (dc_yl-centery)*fracstep; \n', '    frac = dc_texturemid + (dc_yl-centery)*fracstep; \n    if (gpu_par_active) { gpu_q_col(dest, dc_source, dc_colormap, frac, fracstep, count, dc_x); return; }\n', 1)
    s = s.replace('    count = ds_x2 - ds_x1;\n', '    count = ds_x2 - ds_x1;\n    if (gpu_par_active) { gpu_q_span(dest, ds_source, ds_colormap, position, step, count); return; }\n', 1)
    s = re.sub(r'(void R_DrawFuzzColumn \(void\)\s*\{)', r'\1 gpu_flush();', s, count=1)
    s = re.sub(r'(void R_DrawTranslatedColumn \(void\)\s*\{)', r'\1 gpu_flush();', s, count=1)
    return s
patch('src_gpu/r_draw.c', rdraw)
def rmain(s):
    s = s.replace('    R_DrawPlanes ();\n', '    R_DrawPlanes ();\n    { extern void gpu_flush(void); gpu_flush(); }\n', 1)
    s = s.replace('    R_DrawMasked ();\n', '    R_DrawMasked ();\n    { extern void gpu_flush(void); gpu_flush(); }\n', 1)
    return s
patch('src_gpu/r_main.c', rmain)
def ivid(s):
    s = s.replace('static struct color colors[256];', 'struct color colors[256];')
    s = s.replace('void I_FinishUpdate (void)\n{', 'void I_FinishUpdate (void)\n{\n    DG_DrawFrame(); return;   /* palette conversion happens in the megakernel copy tasks (parallel) */', 1)
    return s
patch('src_gpu/i_video.c', ivid)
PYEOF
mkdir -p obj; rm -f obj/*.bc
fail=0
for f in $FILES; do OPT="-O2"; [ $f = m_menu ] && OPT="-O1"; ( clang $CF $OPT -c -emit-llvm $SRC/$f.c -o obj/$f.bc 2> obj/$f.err || { echo "FAIL $f"; head -12 obj/$f.err; } ) & done; wait
clang $CF -c -emit-llvm tinylibc.c -o obj/tinylibc.bc 2> obj/tinylibc.err || { echo FAIL tinylibc; head -20 obj/tinylibc.err; fail=1; }
clang $CF -c -emit-llvm gpu_par.c -o obj/gpu_par.bc 2> obj/gpu_par.err || { echo FAIL par; head -20 obj/gpu_par.err; fail=1; }
clang $CF -c -emit-llvm gpu_glue.c -o obj/gpu_glue.bc 2> obj/gpu_glue.err || { echo FAIL glue; head -20 obj/gpu_glue.err; fail=1; }
ls obj/*.bc | wc -l
[ $fail = 0 ] || exit 1
llvm-link obj/*.bc -o doom.bc
llvm-dis doom.bc -o doom.ll
python3 fix_varargs.py doom.ll doom_nv.ll
llc -march=nvptx64 -mcpu=sm_80 -mattr=+ptx75 -O2 doom_nv.ll -o doom.ptx
ptxas -arch=sm_80 -c doom.ptx -o doom.cubin 2>&1 | grep -v "Potential out of bound" || true
ls -la doom.cubin doom.ptx
# assemble the PTX into a relocatable device object that nvcc can link; cap registers so the megakernel keeps 2 blocks per SM
nvcc -arch=sm_80 -dc -maxrregcount=112 doom.ptx -o doom_dev.o 2>&1 | grep -v 'Potential out of bound' || true
ls -la doom_dev.o
