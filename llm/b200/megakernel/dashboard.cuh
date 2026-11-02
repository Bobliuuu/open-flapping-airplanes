// dashboard.cuh : host-only HTTP dashboard for the Triad kernel. Serves the three-panel page (index.html next to the binary), a JSON /state,
// and a persistent /stream of 65,024-byte Doom frames. A 10 Hz sampler thread reads the pinned counters the kernel updates (loss, steps,
// frames, sampled token ids) and detokenizes the generated token ids with the cl100k vocabulary (flat file from make_vocab.py).
#pragma once
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <deque>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

namespace dash {
inline std::mutex mu;
inline std::string text, prompt, note, index_path = "index.html";
inline std::vector<float> curve_t, curve_l;
inline std::atomic<int> phase{0};                                   // 0 starting, 1 running, 3 finished
inline double t0 = 0, kt = 0, steps_s = 0, toks_s = 0, fps = 0;
inline int sm_train = 0, sm_infer = 0, sm_doom = 0, nsm = 0, tokens_per_step = 4096;
inline unsigned char* fb_host = nullptr; inline Live* live = nullptr;
inline const volatile int* tok_host = nullptr; inline size_t tok_cap = 0; inline int skip_tokens = 0;
inline std::vector<std::string> vocab;

inline size_t tok_count_pub();
inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
inline bool load_vocab(const std::string& path) {                    // u32 n, u32 off[n+1], bytes
  std::ifstream f(path, std::ios::binary); if (!f) return false;
  unsigned n = 0; f.read((char*)&n, 4); std::vector<unsigned> off(n + 1); f.read((char*)off.data(), 4 * (n + 1));
  std::string bytes((size_t)off[n], '\0'); f.read(&bytes[0], bytes.size());
  vocab.resize(n); for (unsigned i = 0; i < n; i++) vocab[i] = bytes.substr(off[i], off[i + 1] - off[i]);
  return true;
}
inline std::string jesc(const std::string& s) {
  std::string o; for (unsigned char c : s) { if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n"; else if (c == '\t') o += ' '; else if (c >= 32) o += (char)c; }
  return o;
}
inline void stream_frames(int c) {
  const char* h = "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
  if (send(c, h, strlen(h), MSG_NOSIGNAL) < 0) { close(c); return; }
  unsigned last = 0xffffffffu; std::vector<unsigned char> buf(FBSZ);
  for (;;) {
    const unsigned f = live->frames;
    if (f == last) { usleep(1500); continue; }
    last = f; memcpy(buf.data(), fb_host, FBSZ);
    size_t off = 0; while (off < buf.size()) { ssize_t w = send(c, buf.data() + off, buf.size() - off, MSG_NOSIGNAL); if (w <= 0) { close(c); return; } off += w; }
  }
}
inline void serve_client(int c) {
  char req[2048]; const int n = (int)recv(c, req, sizeof req - 1, 0);
  if (n <= 0) { close(c); return; }
  req[n] = 0; std::string path = "/"; { char* sp = strchr(req, ' '); if (sp) { char* e = strchr(sp + 1, ' '); if (e) path.assign(sp + 1, e); } }
  if (path == "/stream") { stream_frames(c); return; }
  std::string body, ctype = "text/html; charset=utf-8";
  if (path == "/frame") { body.assign((const char*)fb_host, FBSZ); ctype = "application/octet-stream"; }
  else if (path == "/state") {
    ctype = "application/json";
    std::lock_guard<std::mutex> lk(mu);
    std::ostringstream o;
    o << "{\"phase\":" << phase.load() << ",\"t\":" << kt << ",\"steps\":" << live->steps << ",\"loss\":" << live->loss << ",\"chars\":" << tok_count_pub()
      << ",\"frames\":" << live->frames << ",\"steps_s\":" << steps_s << ",\"chars_s\":" << toks_s << ",\"fps\":" << fps
      << ",\"sm\":[" << sm_train << "," << sm_infer << "," << sm_doom << "],\"nsm\":" << nsm << ",\"batch\":" << tokens_per_step << ",\"note\":\"" << jesc(note) << "\",\"prompt\":\"" << jesc(prompt) << "\"";
    const size_t tl = text.size(); o << ",\"text\":\"" << jesc(text.substr(tl > 900 ? tl - 900 : 0)) << "\",\"curve\":[";
    for (size_t i = 0; i < curve_t.size(); i++) o << (i ? "," : "") << "[" << curve_t[i] << "," << curve_l[i] << "]";
    o << "]}"; body = o.str();
  } else if (path == "/" || path == "/index.html") { std::ifstream f(index_path); std::stringstream ss; ss << f.rdbuf(); body = ss.str(); if (body.empty()) body = "index.html not found next to the binary"; }
  else { std::string h = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"; send(c, h.data(), h.size(), 0); close(c); return; }
  std::ostringstream h; h << "HTTP/1.1 200 OK\r\nContent-Type: " << ctype << "\r\nContent-Length: " << body.size() << "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
  const std::string hs = h.str(); send(c, hs.data(), hs.size(), MSG_NOSIGNAL);
  size_t off = 0; while (off < body.size()) { ssize_t w = send(c, body.data() + off, body.size() - off, MSG_NOSIGNAL); if (w <= 0) break; off += w; }
  close(c);
}
inline void server_thread(int port) {
  int s = socket(AF_INET, SOCK_STREAM, 0); int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(port);
  if (bind(s, (sockaddr*)&a, sizeof a) < 0 || listen(s, 64) < 0) { perror("dashboard bind/listen"); exit(1); }
  for (;;) { int c = accept(s, nullptr, nullptr); if (c >= 0) std::thread(serve_client, c).detach(); }
}
inline size_t tokens_seen = 0;                                      // number of token ids consumed from the pinned ring (guarded by mu)
inline size_t tok_count_pub() { return tokens_seen; }
// 10 Hz: consume new token ids, rates over a ~1 s window, loss curve
inline void sampler_thread() {
  struct Smp { double t; unsigned steps, frames; size_t toks; }; std::deque<Smp> win; double last_curve = -1;
  for (;;) {
    usleep(100000);
    if (phase.load() == 0) continue;
    const double t = now_s(); std::lock_guard<std::mutex> lk(mu);
    if (phase.load() == 1) kt = t - t0;
    if (tok_host) while (tokens_seen < tok_cap) {
      const int id = tok_host[tokens_seen]; if (id < 0) break;
      if ((int)tokens_seen >= skip_tokens) {
        if (id < (int)vocab.size()) text += vocab[id];
        else if (id == decode::OUT_RESTART) text += "\n\n> " + prompt + "\n";   // answer mode: the prompt is asked again
      }
      tokens_seen++;
    }
    const unsigned st = live->steps, fr = live->frames; const float loss = live->loss;
    win.push_back({t, st, fr, tokens_seen}); while (win.size() > 12 && t - win.front().t > 1.05) win.pop_front();
    const Smp& a = win.front(); const double dt = t - a.t;
    if (dt > 0.3) { steps_s = (st - a.steps) / dt; fps = (fr - a.frames) / dt; toks_s = (double)(tokens_seen - a.toks) / dt; }
    if (st > 0 && kt - last_curve > 0.5) { curve_t.push_back((float)kt); curve_l.push_back(loss); last_curve = kt;
      if (curve_t.size() > 800) { std::vector<float> a2, b2; for (size_t i = 0; i < curve_t.size(); i += 2) { a2.push_back(curve_t[i]); b2.push_back(curve_l[i]); } curve_t = a2; curve_l = b2; } }
  }
}
inline void start(int port) { std::thread(server_thread, port).detach(); std::thread(sampler_thread).detach(); }
}  // namespace dash
