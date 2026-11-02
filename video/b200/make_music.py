#!/usr/bin/env python3
"""Background music for the demo video, composed by this script: no samples and no outside audio, so it is covered by the repository's
license. A calm electronic track at 100 BPM (pad, arpeggio, bass, soft drums) whose sections follow the video's scenes: quiet over the
live recording, building through the zoom and the code, full during the side by side, lighter over the cards, fading out on the end card.
Only numpy is needed; filters and reverb are done with FFTs. The same inputs give the same file, bit for bit.

  python3 make_music.py --info ../out/demo_info.json --out ../out/music.wav     (demo_info.json: written by render_demo.js)
"""
import argparse
import json
import wave

import numpy as np

SR = 44100
BPM = 100
BEAT = 60.0 / BPM
BAR = 4 * BEAT
# two bars per chord: Fmaj9, G6/9, Em9, Am9 (voiced around middle C) and their bass notes
CHORDS = [([53, 57, 60, 64, 67], 41), ([55, 59, 62, 64, 69], 43), ([52, 55, 59, 62, 66], 40), ([57, 60, 64, 67, 71], 45)]
# how much of each part plays in each scene: pad, arpeggio, sustained bass, pulsing bass, kick on 1 and 3, kick on every beat,
# hats, snare on 2 and 4, brightness (0 dark .. 1 open)
LEVELS = {
    "live":  dict(pad=.9, arp=.25, bass_sus=0, bass_pulse=0, kick13=0, kick4=0, hats=0, snare=0, bright=0),
    "zoom":  dict(pad=1, arp=.6, bass_sus=.7, bass_pulse=0, kick13=0, kick4=0, hats=0, snare=0, bright=.2),
    "code":  dict(pad=1, arp=.75, bass_sus=0, bass_pulse=.7, kick13=.55, kick4=0, hats=.35, snare=0, bright=.35),
    "race":  dict(pad=1, arp=.9, bass_sus=0, bass_pulse=1, kick13=0, kick4=1, hats=.7, snare=.6, bright=1),
    "cards": dict(pad=1, arp=.7, bass_sus=.7, bass_pulse=0, kick13=.4, kick4=0, hats=.3, snare=0, bright=.5),
    "vsfa":  dict(pad=1, arp=.75, bass_sus=0, bass_pulse=.6, kick13=.5, kick4=0, hats=.35, snare=0, bright=.55),
    "fair":  dict(pad=1, arp=.55, bass_sus=.6, bass_pulse=0, kick13=0, kick4=0, hats=.15, snare=0, bright=.3),
    "end":   dict(pad=1, arp=.3, bass_sus=.5, bass_pulse=0, kick13=0, kick4=0, hats=0, snare=0, bright=.15),
}


def hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def controls(scenes, n):
    """Per-sample level of every part: each scene's levels from its first beat on, smoothed over one beat."""
    order = sorted(scenes.items(), key=lambda kv: kv[1][0])
    curves = {k: np.zeros(n) for k in LEVELS["live"]}
    for name, (a, b) in order:
        lv = LEVELS.get(name, LEVELS["cards"])
        i0 = int(round(round(a / BEAT) * BEAT * SR)) if name != order[0][0] else 0
        for k in curves:
            curves[k][i0:] = lv[k]
    w = int(BEAT * SR)
    for k, c in curves.items():
        cs = np.cumsum(np.concatenate([np.full(w, c[0]), c]))
        curves[k] = (cs[w:] - cs[:-w]) / w
    return curves


def place(dst, src, at):
    """Add the stereo or mono clip `src` into `dst` starting at sample `at`, cropped to fit."""
    if at >= dst.shape[-1]:
        return
    m = min(src.shape[-1], dst.shape[-1] - at)
    dst[..., at:at + m] += src[..., :m]


def fft_filter(x, sr, lo=None, hi=None, slope=600.0):
    """Zero-phase band filter with soft edges, applied with one FFT (x: (..., n))."""
    n = x.shape[-1]
    f = np.fft.rfftfreq(n, 1.0 / sr)
    g = np.ones_like(f)
    if hi is not None:
        g *= 1.0 / (1.0 + np.exp((f - hi) / slope))
    if lo is not None:
        g *= 1.0 / (1.0 + np.exp((lo - f) / (slope * .25)))
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * g, n, axis=-1)


def pad(n, dur):
    """Five-voice detuned additive saw pad, one chord per two bars; returns a dark and a bright version (stereo)."""
    dark, bright = np.zeros((2, n)), np.zeros((2, n))
    seg, att, rel = 2 * BAR, 1.0, 1.6
    k = 0
    while k * seg < dur:
        notes, _ = CHORDS[k % len(CHORDS)]
        t0 = k * seg
        m = int((seg + rel) * SR)
        t = np.arange(m) / SR
        env = np.clip(t / att, 0, 1) ** 2 * (3 - 2 * np.clip(t / att, 0, 1))          # smooth attack
        env *= np.where(t < seg, 1.0, np.clip(1 - (t - seg) / rel, 0, 1) ** 2)        # release into the next chord
        env *= 1 + .06 * np.sin(2 * np.pi * .11 * (t + t0))                          # slow swell
        d, b = np.zeros((2, m)), np.zeros((2, m))
        for note in notes:
            for cents, (gl, gr) in ((-7, (.8, .4)), (0, (.6, .6)), (7, (.4, .8))):
                f0 = hz(note) * 2 ** (cents / 1200)
                ph = 2 * np.pi * f0 * t + note * .7 + cents
                for h in range(1, int(4200 // f0) + 1):
                    s = np.sin(h * ph) / h
                    fh = h * f0
                    wd = 1.0 / (1.0 + np.exp((fh - 900) / 250))
                    wb = 1.0 / (1.0 + np.exp((fh - 2800) / 500))
                    d[0] += gl * wd * s; d[1] += gr * wd * s
                    b[0] += gl * wb * s; b[1] += gr * wb * s
        place(dark, d * env, int(t0 * SR))
        place(bright, b * env, int(t0 * SR))
        k += 1
    return dark / 14.0, bright / 14.0


def arp(n, dur):
    """Plucked arpeggio in sixteenths over the chord tones, an octave up every other bar, with a ping-pong delay (stereo)."""
    out = np.zeros((2, n))
    step = BEAT / 4
    note_len = int(.6 * SR)
    t = np.arange(note_len) / SR
    pattern = [0, 2, 4, 2, 1, 3, 4, 3]
    i = 0
    while i * step < dur:
        tt = i * step
        bar = int(tt // BAR)
        notes, _ = CHORDS[(bar // 2) % len(CHORDS)]
        m = notes[pattern[i % len(pattern)]] + (12 if bar % 2 else 0) + 12
        f = hz(m)
        env = (1 - np.exp(-t / .002)) * np.exp(-t / .17)
        s = (np.sin(2 * np.pi * f * t) + .22 * np.sin(4 * np.pi * f * t) + .07 * np.sin(6 * np.pi * f * t)) * env
        vel = 1.0 if i % 4 == 0 else .72
        pan = .38 if i % 2 else .62
        place(out, np.stack([s * (1 - pan), s * pan]) * vel, int(tt * SR))
        i += 1
    wet = np.zeros_like(out)
    d = int(3 * step * SR)
    for r, g in enumerate((.42, .26, .15, .08), start=1):                            # ping-pong: repeats alternate sides
        src = out[::-1] if r % 2 else out
        place(wet, src * g, r * d)
    return (out + fft_filter(wet, SR, hi=3500)) * .5


def bass(n, dur):
    """Sub bass: one sustained note per chord, and a pulsing eighth-note line (each stem mono)."""
    sus, pulse = np.zeros(n), np.zeros(n)
    seg = 2 * BAR
    k = 0
    while k * seg < dur:
        _, b = CHORDS[k % len(CHORDS)]
        m = int((seg + .3) * SR)
        t = np.arange(m) / SR
        env = np.clip(t / .08, 0, 1) * np.where(t < seg, 1.0, np.clip(1 - (t - seg) / .3, 0, 1))
        f = hz(b)
        s = np.sin(2 * np.pi * f * t) + .25 * np.sin(4 * np.pi * f * t)
        place(sus, np.tanh(1.4 * s) * env, int(k * seg * SR))
        for e in range(16):                                                           # eighth notes over the two bars
            tt = k * seg + e * BEAT / 2
            ln = int(.27 * SR)
            u = np.arange(ln) / SR
            fe = f * (2 if e % 8 == 7 else 1)
            pe = np.sin(2 * np.pi * fe * u) + .3 * np.sin(4 * np.pi * fe * u)
            place(pulse, np.tanh(1.6 * pe) * np.clip(u / .006, 0, 1) * np.exp(-u / .14), int(tt * SR))
        k += 1
    return sus * .3, pulse * .27


def drums(n, dur, rng):
    """Kick on 1 and 3, kick on every beat, hats on the off-beats, snare on 2 and 4 (mono stems), and each kick pattern's envelope for
    ducking the pad and bass under it."""
    k13, k4, hats, snare, duck13, duck4 = (np.zeros(n) for _ in range(6))
    t = np.arange(int(.45 * SR)) / SR
    freq = 46 + 115 * np.exp(-t / .032)
    kick = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t / .21) * np.clip(t / .002, 0, 1)
    hl = int(.06 * SR)
    hat = fft_filter(rng.standard_normal(hl), SR, lo=7000) * np.exp(-np.arange(hl) / SR / .014) * np.clip(np.arange(hl) / SR / .0015, 0, 1)
    sl = int(.25 * SR)
    ts = np.arange(sl) / SR
    sn = fft_filter(rng.standard_normal(sl), SR, lo=1200, hi=5000) * np.exp(-ts / .06) + .5 * np.sin(2 * np.pi * 190 * ts) * np.exp(-ts / .04)
    env = np.exp(-np.arange(int(.35 * SR)) / SR / .12)
    b = 0
    while b * BEAT < dur:
        at = int(b * BEAT * SR)
        place(k4, kick, at); place(duck4, env, at)
        if b % 2 == 0:
            place(k13, kick, at); place(duck13, env, at)
        if b % 2 == 1:
            place(snare, sn, at)
        place(hats, hat, at + int(BEAT / 2 * SR))
        b += 1
    return k13 * .55, k4 * .55, hats * .3, snare * .35, duck13, duck4


def reverb(x, rng, seconds=2.6, decay=.85):
    """Stereo convolution reverb: exponentially decaying noise, darkened, different on each side (x: (2, n))."""
    ln = int(seconds * SR)
    ir = rng.standard_normal((2, ln)) * np.exp(-np.arange(ln) / SR / decay * 2.3)
    ir = fft_filter(ir, SR, hi=5000)
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    n = x.shape[1] + ln
    nf = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nf, axis=1) * np.fft.rfft(ir, nf, axis=1), nf, axis=1)[:, :x.shape[1]]
    return y


def render(dur, scenes):
    n = int(round(dur * SR))
    rng = np.random.default_rng(7)
    c = controls(scenes, n)
    dark, bright = pad(n, dur)
    padmix = dark * (1 - c["bright"]) + bright * c["bright"]
    a = arp(n, dur)
    sus, pulse = bass(n, dur)
    k13, k4, hats, snare, duck13, duck4 = drums(n, dur, rng)
    ducking = 1 - .3 * np.clip(duck13 * c["kick13"] + duck4 * c["kick4"], 0, 1)   # pad and bass dip under the kick
    music = padmix * c["pad"] * ducking + a * c["arp"]
    music += (sus * c["bass_sus"] + pulse * c["bass_pulse"]) * ducking
    music = music + .3 * reverb(padmix * c["pad"] * .7 + a * c["arp"] * 1.1 + snare * c["snare"] * .5, rng)
    drum_bus = k13 * c["kick13"] + k4 * c["kick4"] + hats * c["hats"] + snare * c["snare"]
    music = music + np.stack([drum_bus * .95, drum_bus * .95])
    t = np.arange(n) / SR
    music *= np.clip(t / 1.2, 0, 1) * np.clip((dur - t) / 3.5, 0, 1) ** 1.5          # fade in and out
    rms = np.sqrt(np.mean(music ** 2))
    music *= 10 ** (-18 / 20) / max(rms, 1e-9)                                       # about -18 dBFS RMS: a background level
    music = np.tanh(music * 1.2) / 1.2                                               # gentle peak control
    peak = np.abs(music).max()
    if peak > .89:
        music *= .89 / peak                                                          # no higher than -1 dBFS
    return music


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", required=True, help="the video's timeline (render_demo.js writes it next to the MP4)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = json.load(open(a.info))
    music = render(float(info["duration"]), {k: v for k, v in info["scenes"].items()})
    pcm = (np.clip(music, -1, 1) * 32767).astype("<i2").T.copy()
    with wave.open(a.out, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"wrote {a.out}: {info['duration']:.2f} s, peak {np.abs(music).max():.3f}, rms {np.sqrt(np.mean(music ** 2)):.4f}")


if __name__ == "__main__":
    main()
