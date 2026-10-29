---
title: AcoustiGuard
emoji: 🔊
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: cc-by-sa-4.0
short_description: Acoustic fault screening for pumps and fans (MSc coursework)
---

# AcoustiGuard – acoustic fault screening (coursework demo)

Upload a 10-second WAV/FLAC of a **pump** or **fan**. A trained CNN (EfficientNet-B0 on log-mel
spectrograms, MIMII dataset) returns a fault score at request time, with honest uncertainty
measures, input/out-of-distribution checks and a Grad-CAM explanation that is checked live.

**This is a coursework demonstration, not a certified maintenance tool.** On the held-out test
machine the model was no better than chance; see the limitations panel in the app.

Code: https://github.com/pasiuvi/mimii-machine-fault-detection · Data: MIMII (Purohit et al., 2019), CC BY-SA 4.0.
Runs on the free CPU tier; the first request after the Space wakes up takes ~30–60 s.
