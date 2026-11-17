"""Forecast GTA VI launch-day Twitch viewers, hour by hour, from hour 1.

Model: a comparable-launch log-ratio forecaster anchored on the case's
hour-1 print.

For every past launch in the case's ``comparables`` we take its release-day
hourly curve and form, for each hour h = 2..24, the log ratio
``ln(viewers_h / viewers_1)``.  The ratio, not the level, is what transfers
across launches: hour 1 is 00:00-01:00 UTC, before most of the world has
unlocked (New Zealand first, the UK at 00:00Z, US East 05:00Z, US West
08:00Z), so the day's shape is a ramp into the evening prime time and the
level of hour 1 says a lot about how much of that ramp is still to come.

The centre of the forecast at each hour is a local-linear (Gaussian-kernel)
regression of the log ratio on ``ln(hour-1 viewers)`` across the
comparables.  This matters because the relationship is strongly
scale-dependent: across the 38 comparables the correlation between
``ln(hour 1)`` and the evening log ratio is about -0.94.  Launches whose
first hour is already huge (ELDEN RING, Cyberpunk 2077, about 850k) stay
near flat into the evening (ratio about 0.9), while launches with a tiny
first hour explode (Hollow Knight: Silksong, hour 1 of 164 viewers, evening
ratio over 1000).  A single pooled median misses both ends; a global
straight line fits neither end well either, because the slope is about -0.94
overall but only about -0.25 among launches with hour 1 above 50k.  The
kernel-local fit (bandwidth 1.5 in log viewers) follows the steep slope at
small scales and the flat slope at large scales.

Two groups of comparables are left out of the fit, because their release-day
curves do not describe a clean rolling-midnight launch like GTA VI's:
launches whose ``unlock`` note says ``early-access-contaminated`` (their
audience had already peaked days before release day), and launches tiered
``weak`` (free-to-play or slow-burn launches).  Simultaneous-global launches
are kept: they are what teach the fit the small-first-hour end of the
scale, and the kernel downweights them automatically for a large first hour.

The spread at each hour comes from the kernel-weighted standard deviation
of the local-linear residuals at that hour, floored at 0.40 log units (the
floor also guards the extrapolation when hour 1 is larger than anything in
the training set), times 0.6, times the standard normal quantiles
(-1.645, -0.674, 0, 0.674, 1.645).  On top of that, if the case's hour-1
print lies outside the training range, the spread is widened in quadrature
by 0.5 log units per log unit of distance outside the range (capped at
1.0), because the kernel fit is then extrapolating from far-away launches.
The 0.6 multiplier and the floor were chosen on the practice cases: the
resulting 90% band covers about 90% of practice hours.  The centre is clamped to the training log ratios' range
(plus a 0.1 margin) at each hour, so an hour-1 print outside the training
range cannot produce a wild extrapolation.

Edge cases: a comparable is used only if its release day has an hour-1
point and at least 20 of the 24 hours (GTA VI's own pre-release history,
whose day start is 2026-08-27, has no hour 1 on its own day start and so
drops out by itself).  Missing hours of a usable comparable are skipped for
that hour only.  Zero viewers are floored at 1 before taking logs.  If too
few comparables survive the filters, the filters are relaxed, and if there
are none at all the forecast falls back to a flat persistence curve with a
widening spread, so the program always writes a valid file.

Standard library only, no network, no randomness, deterministic: the same
case file always yields the same bytes.  Runs in well under a second a case.

Input: every ``*.in`` file in $MAKEFUN_INPUT_DIR (else /input).
Output: ``<id>.out`` in $MAKEFUN_OUTPUT_DIR (else /output), one JSON object
``{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}`` for h 2..24.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Gaussian-kernel bandwidth (log-viewer units) for the local-linear fit.
BANDWIDTH = 1.5
# Multiplier on the residual spread; 0.6 puts practice 90% coverage near 0.9.
SPREAD_MULT = 0.6
# Floor on the per-hour log spread, also a guard for extrapolated hour-1 prints.
SPREAD_FLOOR = 0.40
# Extra spread per log unit the hour-1 print sits outside the training
# range, added in quadrature and capped; the final case's hour-1 print may
# exceed anything in the training set, and the kernel fit then extrapolates
# from far-away launches, so the band must widen with the distance.
SPREAD_EXTRA_PER_LOG = 0.5
SPREAD_EXTRA_CAP = 1.0
# A comparable must have this many release-day hours to be used.
MIN_HOURS = 20


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def is_contaminated(note):
    return "early-access-contaminated" in note.lower()


def is_weak(note):
    return "tier: weak" in note.lower()


def release_day_curve(comp):
    """Hourly viewers on the comparable's release day, {hour: viewers}."""
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    curve = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        try:
            at = parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            curve[hour] = float(viewers)
    return curve


def collect_comparables(case):
    """Usable (log hour-1, {hour: log ratio}) pairs, clean launches first."""
    raw = []
    for comp in case.get("comparables") or []:
        curve = release_day_curve(comp)
        if 1 not in curve or len(curve) < MIN_HOURS:
            continue
        first = curve[1]
        ratios = {}
        for hour, viewers in curve.items():
            if hour != 1:
                ratios[hour] = math.log(max(viewers, 1.0) / max(first, 1.0))
        raw.append((math.log(max(first, 1.0)), ratios, comp.get("unlock") or ""))
    for drop_weak in (True, False):
        kept = [r for r in raw
                if not is_contaminated(r[2]) and (not drop_weak or not is_weak(r[2]))]
        if len(kept) >= 10:
            return kept
    return raw


def local_linear(rows, x0):
    """Gaussian-kernel local-linear fit of y on x at x0.

    rows is a list of (x, y).  Returns (prediction, residuals, weights).
    """
    weights = [math.exp(-((x - x0) ** 2) / (2 * BANDWIDTH * BANDWIDTH))
               for x, _ in rows]
    total = sum(weights)
    if total <= 1e-12:
        middle = statistics.median([y for _, y in rows])
        return middle, [y - middle for _, y in rows], weights
    mean_x = sum(w * x for w, (x, _) in zip(weights, rows)) / total
    mean_y = sum(w * y for w, (_, y) in zip(weights, rows)) / total
    var_x = sum(w * (x - mean_x) ** 2 for w, (x, _) in zip(weights, rows))
    if var_x <= 1e-12:
        return mean_y, [y - mean_y for _, y in rows], weights
    slope = (sum(w * (x - mean_x) * (y - mean_y)
                 for w, (x, y) in zip(weights, rows)) / var_x)
    intercept = mean_y - slope * mean_x
    prediction = intercept + slope * x0
    residuals = [y - (intercept + slope * x) for x, y in rows]
    return prediction, residuals, weights


def weighted_std(residuals, weights):
    total = sum(weights)
    if total <= 0 or not residuals:
        return SPREAD_FLOOR
    mean = sum(w * r for w, r in zip(weights, residuals)) / total
    var = sum(w * (r - mean) ** 2 for w, r in zip(weights, residuals)) / total
    return math.sqrt(max(var, 0.0))


def round2(value):
    """Round to 2 decimals, half up, returning an int for whole numbers."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100
    return int(rounded) if rounded.is_integer() else rounded


def persistence_quantiles(p1, hour):
    """Fallback: flat at the hour-1 print with a widening spread."""
    sigma = 0.15 * math.sqrt(hour - 1)
    return [round2(p1 * math.exp(z * sigma)) for z in Z]


def forecast_case(case):
    known = case.get("known") or []
    first_row = next((k for k in known if k.get("hour") == 1), None)
    if first_row is None and known:
        first_row = known[0]
    p1 = 1.0
    if first_row is not None:
        try:
            p1 = max(1.0, float(first_row.get("viewers") or 1))
        except (TypeError, ValueError):
            p1 = 1.0
    log_p1 = math.log(p1)

    pairs = collect_comparables(case)
    hours = [row["hour"] for row in case.get("forecast") or []]
    if not hours:
        hours = list(range(2, 25))

    if not pairs:
        return {"hours": [{"hour": h, "q": persistence_quantiles(p1, h)}
                          for h in hours]}

    centres = {}
    spreads = {}
    for hour in hours:
        rows = [(log_first, ratios[hour]) for log_first, ratios, _ in pairs
                if hour in ratios]
        if not rows:
            centres[hour] = 0.0
            spreads[hour] = SPREAD_FLOOR
            continue
        centre, residuals, weights = local_linear(rows, log_p1)
        # Clamp the centre to the training ratios' range (plus a margin) so
        # an hour-1 print outside the training range cannot extrapolate wild.
        lo = min(y for _, y in rows) - 0.1
        hi = max(y for _, y in rows) + 0.1
        centres[hour] = min(max(centre, lo), hi)
        spreads[hour] = max(weighted_std(residuals, weights), SPREAD_FLOOR)
        # Extrapolation guardrail: if the case's hour-1 print lies outside
        # the training range, the kernel fit is extrapolating from distant
        # launches, so widen the spread in proportion to the distance
        # (in quadrature, capped). In-range cases are unaffected.
        xs = [x for x, _ in rows]
        outside = max(min(xs) - log_p1, log_p1 - max(xs), 0.0)
        extra = min(SPREAD_EXTRA_PER_LOG * outside, SPREAD_EXTRA_CAP)
        spreads[hour] = math.sqrt(spreads[hour] ** 2 + extra ** 2)

    out_hours = []
    for hour in hours:
        centre = centres.get(hour, 0.0)
        sigma = spreads.get(hour, SPREAD_FLOOR) * SPREAD_MULT
        quantiles = []
        previous = 0.0
        for z in Z:
            value = p1 * math.exp(centre + z * sigma)
            value = min(max(value, 1.0), 1e9)
            value = round2(value)
            if value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        out_hours.append({"hour": hour, "q": quantiles})
    return {"hours": out_hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            p1 = 1.0
            try:
                known = case.get("known") or []
                row = next((k for k in known if k.get("hour") == 1), known[0] if known else None)
                if row is not None:
                    p1 = max(1.0, float(row.get("viewers") or 1))
            except Exception:
                pass
            forecast = {"hours": [{"hour": h, "q": persistence_quantiles(p1, h)}
                                  for h in range(2, 25)]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
