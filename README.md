# GTA VI Twitch CCU forecast: comparable-launch log-ratio forecaster

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kainze29/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kainze29/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day, 2026-11-19 (UTC), from the first hour's print, as
five quantiles per hour. Entry for
https://make.fun/bounty/gta6-twitch-ccu-forecast-01.

## How it works

For every past launch in the case's `comparables`, take its release-day
hourly curve and form the log ratio `ln(viewers_h / viewers_1)` for each
hour 2 to 24. The forecast centre at each hour is a local-linear
(Gaussian-kernel, bandwidth 1.5 in log viewers) regression of that log
ratio on `ln(hour-1 viewers)`. The spread is the kernel-weighted standard
deviation of the local-linear residuals at that hour, floored at 0.40 log
units, times 0.6, times the standard normal quantiles.

Two facts from the data drive this design:

- The shape is strongly scale-dependent. Across the comparables, the
  correlation between `ln(hour 1)` and the evening log ratio is about
  -0.94. Launches with a huge first hour (ELDEN RING, Cyberpunk 2077,
  about 850k) stay near flat into the evening (ratio about 0.9), while
  launches with a tiny first hour explode (Hollow Knight: Silksong, hour 1
  of 164 viewers, evening ratio over 1000). A pooled median misses both
  ends; the kernel-local fit follows the steep slope at small scales and
  the flat slope at large scales.
- The unlock rolls at local midnight (New Zealand first, the UK at 00:00Z,
  US East 05:00Z, US West 08:00Z), so the curve ramps through the morning
  UTC hours into an evening prime-time peak and decays late at night. The
  per-hour regressions learn that shape from the comparables instead of
  assuming it.

Early-access-contaminated launches and weak-tier launches (free-to-play,
slow-burn) are left out of the fit: their release-day curves do not
describe a clean rolling-midnight launch like GTA VI's. The centre is
clamped to the training ratios' range so an hour-1 print outside the
training range cannot extrapolate wildly. Zero viewers are floored at 1.

## Practice results (local evaluation)

Practice answers (hours 2 to 24) fetched from SullyGnome's public JSON API
exactly as the kit's `input/NOTICE.txt` documents, one month request per
game; hour-1 values match the case files exactly. Scored with the pinball
loss from RULES.txt (`2 * max(tau*d, (tau-1)*d)`, `d = ln(y) - ln(q)`).

| case | game | hour-1 print | loss | typical miss | 90% coverage |
|---|---|---:|---:|---:|---:|
| p00 | Marvel's Wolverine | 23,357 | 0.1379 | 17.2% | 1.00 |
| p01 | Red Dead Redemption 2 | 116,550 | 0.1367 | 25.6% | 0.96 |
| p02 | God of War Ragnarok | 269,517 | 0.1052 | 19.0% | 1.00 |
| p03 | Marvel's Spider-Man 2 | 119,087 | 0.1809 | 30.1% | 0.83 |
| **mean** | | | **0.1402** | | **0.95** |

House persistence baseline on the same cases: mean loss 0.3871 (per case
0.7437, 0.2868, 0.2146, 0.3033). This entry beats it on every practice
case. Runtime is about 0.09 seconds per case, deterministic (two runs
produce byte-identical output).

## Running it

Python 3.13, standard library only, no network:

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` for each.

## Files

- `main.py` — the whole entry (single file).
- `LICENSE` — MIT, copied from the kit starter.
