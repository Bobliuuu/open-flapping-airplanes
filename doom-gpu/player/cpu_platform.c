// Headless doomgeneric platform for the CPU check: no window, no input, and a clock that advances exactly one game tic per
// cpu_tic() call, like the megakernel's Doom job (one tic per frame). Same arguments as gpu_glue.c.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "doomgeneric.h"
#include "doomstat.h"
#include "d_player.h"

void doomgeneric_Tick(void);
int player_waypoint(void);

static uint64_t g_tic, g_slept_ms;
void DG_Init(void) {}
void DG_DrawFrame(void) {}
void DG_SleepMs(uint32_t ms) { g_slept_ms += ms; }          // the engine sleeps only at start-up and level changes: let that time pass
uint32_t DG_GetTicksMs(void) { return (uint32_t)((g_tic * 1000 + 34) / 35 + g_slept_ms); }
int DG_GetKey(int* pressed, unsigned char* key) { (void)pressed; (void)key; return 0; }
void DG_SetWindowTitle(const char* t) { (void)t; }

void cpu_init(const char* wad, int skill) {
  static char wadbuf[512], skbuf[8];
  static char* argv[16];
  strncpy(wadbuf, wad, sizeof wadbuf - 1);
  snprintf(skbuf, sizeof skbuf, "%d", skill);
  char* a[] = {"doom", "-iwad", wadbuf, "-warp", "1", "1", "-skill", skbuf, "-nomusic", "-nosfx", "-nosound"};
  int n = (int)(sizeof a / sizeof a[0]);
  for (int i = 0; i < n; i++) argv[i] = a[i];
  argv[n] = 0;
  doomgeneric_Create(n, argv);
}
void cpu_tic(void) { g_tic++; doomgeneric_Tick(); }
uint32_t* cpu_screen(void) { return DG_ScreenBuffer; }        // 320x200 XRGB
// [0] game state (0 level, 1 intermission), [1] health, [2] armor, [3] kills, [4] monsters, [5] items, [6] secrets, [7] dead,
// [8] x, [9] y (map units), [10] waypoint
void cpu_state(int* o) {
  player_t* p = &players[consoleplayer];
  o[0] = (int)gamestate; o[1] = p->health; o[2] = p->armorpoints; o[3] = p->killcount; o[4] = totalkills;
  o[5] = p->itemcount; o[6] = p->secretcount; o[7] = p->playerstate == PST_DEAD;
  o[8] = p->mo ? p->mo->x >> 16 : 0; o[9] = p->mo ? p->mo->y >> 16 : 0; o[10] = player_waypoint();
}
