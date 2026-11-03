// Glue between doomgeneric and the megakernel: platform hooks, in-memory WAD. The player is player/player.c.
#include <string.h>
#include <stdint.h>
#include "doomgeneric.h"

extern const unsigned char* tl_wad; extern long tl_wad_len; extern volatile char* tl_dbgbuf; extern long tl_dbglen; extern volatile long tl_dbgn;
static unsigned long long g_now_ns;
static unsigned long long g_t0_ns;
static int g_created;

void tl_heap_reset(void);
void gpu_doom_setup(const unsigned char* wad, long wad_len, volatile char* dbg, long dbglen) { tl_heap_reset(); tl_wad = wad; tl_wad_len = wad_len; tl_dbgbuf = dbg; tl_dbglen = dbglen; tl_dbgn = 0; }
void gpu_init_tables(void); void gpu_init_game_ptrs(void);
extern volatile int gpu_par_active; void gpu_par_init(void); void gpu_par_begin(void); void gpu_par_end(void); void gpu_par_worker(int tid);
static int g_par_enabled;
void gpu_doom_enable_par(int on) { g_par_enabled = on; }
void gpu_doom_create(unsigned long long now_ns) {
  gpu_init_tables(); gpu_init_game_ptrs(); gpu_par_init();
  static char* argv[] = {"doom", "-iwad", "doom1.wad", "-warp", "1", "1", "-skill", "3", "-nomusic", "-nosfx", "-nosound", 0};
  g_now_ns = now_ns; g_t0_ns = now_ns;
  doomgeneric_Create(11, argv); g_created = 1;
}

void DG_Init(void) {}
void DG_DrawFrame(void) {}
void DG_SleepMs(uint32_t ms) { g_now_ns += (unsigned long long)ms * 1000000ull; }   // virtual clock: sleeping advances time, so Doom never spins waiting for a tic
uint32_t DG_GetTicksMs(void) { return (uint32_t)(g_now_ns / 1000000ull); }
void DG_SetWindowTitle(const char* t) { (void)t; }

// No key input: the scripted player (player/player.c) builds the player's command each tic from the game state.
int DG_GetKey(int* pressed, unsigned char* key) { (void)pressed; (void)key; return 0; }
// Thread 0 runs the tic. If parallel drawing is enabled, the other 255 threads of the block must call gpu_doom_worker() for the same tic.
void gpu_doom_frame(unsigned long long now_ns) {
  if (now_ns > g_now_ns) g_now_ns = now_ns;
  if (g_par_enabled) gpu_par_begin();
  doomgeneric_Tick();
  if (g_par_enabled) gpu_par_end();
}
void gpu_doom_worker(int tid) { if (g_par_enabled) gpu_par_worker(tid); }
void gpu_doom_worker_always(int tid) { gpu_par_worker(tid); }      // the megakernel always drives all 256 threads of the block
