// Glue between doomgeneric and the megakernel: platform hooks, in-memory WAD, scripted player.
#include <string.h>
#include <stdint.h>
#include "doomgeneric.h"
#include "doomkeys.h"

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

// ---- scripted player: god mode + all weapons, then walk, turn, strafe, shoot and use doors on a pseudo-random schedule
#define NKEYS 8
static const unsigned char K_FWD = KEY_UPARROW, K_TL = KEY_LEFTARROW, K_TR = KEY_RIGHTARROW, K_FIRE = KEY_FIRE, K_USE = KEY_USE, K_SL = KEY_STRAFE_L, K_SR = KEY_STRAFE_R;
static unsigned char s_keys[NKEYS]; static unsigned char s_down[NKEYS];
static unsigned char q_key[64], q_pr[64]; static int q_h, q_t;
static void qpush(unsigned char key, int pr) { int n = (q_t + 1) & 63; if (n == q_h) return; q_key[q_t] = key; q_pr[q_t] = (unsigned char)pr; q_t = n; }
static unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
static int cheat_pos; static unsigned long long cheat_next_ms;
static const char CHEAT[] = "iddqd\0idfa";

void gpu_doom_bot(void) {
  const unsigned long long ms = (g_now_ns - g_t0_ns) / 1000000ull;
  if (ms > 2500 && cheat_pos < (int)sizeof(CHEAT) - 1 && ms >= cheat_next_ms) {                 // type the cheat codes one key at a time
    if (CHEAT[cheat_pos]) { qpush((unsigned char)CHEAT[cheat_pos], 1); qpush((unsigned char)CHEAT[cheat_pos], 0); cheat_next_ms = ms + 90; }
    else cheat_next_ms = ms + 400;
    cheat_pos++;
  }
  if (ms < 5000) return;
  const unsigned seg = (unsigned)(ms / 650), r = hash32(seg * 2654435761u + 5u);
  unsigned char want[NKEYS];
  want[0] = 1;                                              // forward
  want[1] = (r % 6 == 0) || (r % 17 == 3);                  // turn left
  want[2] = (r % 6 == 1) || (r % 13 == 5);                  // turn right
  want[3] = (hash32(seg * 7u + 1u) % 3) == 0;               // fire
  want[4] = ((ms / 450) % 3) == 0;                          // use (open doors)
  want[5] = (r % 11 == 7); want[6] = (r % 11 == 8);         // strafe
  want[7] = 1;                                              // always run
  if (want[1] && want[2]) want[2] = 0;
  const unsigned char keys[NKEYS] = {KEY_UPARROW, KEY_LEFTARROW, KEY_RIGHTARROW, KEY_FIRE, KEY_USE, KEY_STRAFE_L, KEY_STRAFE_R, KEY_RSHIFT};
  for (int i = 0; i < NKEYS; i++) if (want[i] != s_down[i]) { s_down[i] = want[i]; qpush(keys[i], want[i]); }
  (void)s_keys; (void)K_FWD; (void)K_TL; (void)K_TR; (void)K_FIRE; (void)K_USE; (void)K_SL; (void)K_SR;
}
int DG_GetKey(int* pressed, unsigned char* key) {
  if (q_h == q_t) return 0;
  *key = q_key[q_h]; *pressed = q_pr[q_h]; q_h = (q_h + 1) & 63; return 1;
}
// Thread 0 runs the tic. If parallel drawing is enabled, the other 255 threads of the block must call gpu_doom_worker() for the same tic.
void gpu_doom_frame(unsigned long long now_ns) {
  if (now_ns > g_now_ns) g_now_ns = now_ns;
  gpu_doom_bot();
  if (g_par_enabled) gpu_par_begin();
  doomgeneric_Tick();
  if (g_par_enabled) gpu_par_end();
}
void gpu_doom_worker(int tid) { if (g_par_enabled) gpu_par_worker(tid); }
void gpu_doom_worker_always(int tid) { gpu_par_worker(tid); }      // the megakernel always drives all 256 threads of the block
