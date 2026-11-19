import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Persistence-anchored forecaster (v2).
#
# Design, driven by the server's practice verdict on v1: on the recorded
# histories the earn rates barely move between reads, and the scoring is
# pinball loss on the logit scale, where a small percentage-point miss near
# p = 0 or p = 1 explodes. v1 moved its centre toward fitted growth /
# dilution models and paid for that move at both ends. v2 therefore:
#   * centres every trophy on its LAST OBSERVED rate, applying at most 5%
#     of the recency-weighted fitted move (and exactly zero drift when the
#     observed history is flat);
#   * takes its spread from the history's own read-to-read logit noise,
#     scaled like a random walk over the forecast gap, shrunk toward the
#     cross-trophy pooled noise, inflated slightly for coverage, and held
#     to a floor so flat histories still get tight, non-degenerate bands.

EPS = 1e-6               # probability clip before any logit transform
DRIFT_KEEP = 0.05        # fraction of the fitted move applied to q50
DRIFT_CAP = 0.10         # absolute cap on the applied drift (logit units)
SIGMA_FLOOR = 0.04       # minimum predictive spread (logit units)
SIGMA_INFLATE = 1.15     # coverage inflation on the noise-based spread
POOL_K = 4.0             # shrinkage weight toward the pooled noise estimate
HALF_LIFE_H = 12.0       # recency half-life for the trend fit (hours)
MOVE_WIDEN = 0.85        # band also covers the fitted move we chose not to centre
MOVE_CAP = 2.0           # cap on the fitted move used for widening (logit units)
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

_NORM = NormalDist()


def clip_prob(p):
    return min(1.0 - EPS, max(EPS, p))


def logit_prob(p):
    p = clip_prob(p)
    return math.log(p / (1.0 - p))


def inv_logit_pct(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000.0
    return int(v) if float(v).is_integer() else v


def finish_pct(x):
    """Round a percent for output; snap the unreachable tails to exactly
    0 / 100 (the source page prints 2 decimals, so a quantile within
    0.0001 of a bound is that bound — and a rate printed as 100.00 is
    only covered by a 95% quantile that reaches 100)."""
    if x >= 99.9999:
        return 100.0
    if x <= 0.0001:
        return 0.0
    return round4(min(100.0, max(0.0, x)))


def parse_ts(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (slope, slope SE) or None."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    s2 = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    return b, math.sqrt(max(s2, 0.0) / sxx)


def trophy_obs(case):
    """Per-trophy list of (hours since first read, rate in percent), nulls dropped."""
    history = case.get("history") or []
    times = []
    for read in history:
        try:
            times.append(parse_ts(read["at"]))
        except Exception:
            times.append(None)
    valid = [t for t in times if t is not None]
    t0 = valid[0] if valid else None
    n_tr = len(case.get("trophies") or [])
    obs = [[] for _ in range(n_tr)]
    for read, t in zip(history, times):
        if t is None or t0 is None:
            continue
        rates = read.get("rates") or []
        h = (t - t0).total_seconds() / 3600.0
        for i in range(min(n_tr, len(rates))):
            r = rates[i]
            if r is not None:
                obs[i].append((h, float(r)))
    return obs, times, t0


def forecast_case(case):
    trophies = case["trophies"]
    levels = case.get("levels") or DEFAULT_LEVELS
    zs = [_NORM.inv_cdf(float(tau)) for tau in levels]
    obs, times, t0 = trophy_obs(case)

    try:
        target_ts = parse_ts(case["target"])
    except Exception:
        target_ts = None
    if target_ts is not None and t0 is not None:
        target_h = (target_ts - t0).total_seconds() / 3600.0
    elif case.get("horizon_minutes") is not None and obs and any(obs):
        last_h = max(o[-1][0] for o in obs if o)
        target_h = last_h + float(case["horizon_minutes"]) / 60.0
    else:
        target_h = 0.0

    # Per-trophy read-to-read logit noise, normalised per sqrt(hour) so it
    # scales like a random walk over gaps of any length.
    sigmas = []
    for o in obs:
        units = []
        for (h1, r1), (h2, r2) in zip(o, o[1:]):
            dt = h2 - h1
            if dt > 1e-9:
                d = logit_prob(r2 / 100.0) - logit_prob(r1 / 100.0)
                units.append(d / math.sqrt(dt))
        if units:
            sigmas.append((math.sqrt(sum(u * u for u in units) / len(units)), len(units)))
        else:
            sigmas.append((None, 0))
    pool_vals = [s for s, _ in sigmas if s is not None]
    pool_sigma = median(pool_vals) if pool_vals else 0.0

    last_rates = [o[-1][1] if o else None for o in obs]
    known = [r for r in last_rates if r is not None]
    global_last = median(known) if known else 1.0

    rows = []
    for i, trophy in enumerate(trophies):
        o = obs[i] if i < len(obs) else []
        r_last = last_rates[i] if i < len(last_rates) else None
        base_rate = r_last if r_last is not None else global_last
        mu = logit_prob(base_rate / 100.0)

        last_h = o[-1][0] if o else (max((oo[-1][0] for oo in obs if oo), default=0.0))
        gap_h = max(0.0, target_h - last_h)

        # Drift: at most DRIFT_KEEP of the recency-weighted fitted move,
        # capped, and exactly zero when the observed history is flat.
        drift = 0.0
        full_move = 0.0
        se_move = 0.0
        if len(o) >= 3 and len({r for _, r in o}) >= 2:
            xs = [h for h, _ in o]
            ys = [logit_prob(r / 100.0) for _, r in o]
            ws = [0.5 ** ((last_h - h) / HALF_LIFE_H) for h in xs]
            fit = wls_line(xs, ys, ws)
            if fit is not None:
                slope, se_slope = fit
                full_move = slope * gap_h
                se_move = se_slope * gap_h
                drift = DRIFT_KEEP * full_move
                drift = min(DRIFT_CAP, max(-DRIFT_CAP, drift))
        mu += drift

        # Spread: own noise shrunk toward the pooled noise, random-walk
        # scaled over the forecast gap and inflated; plus, in quadrature,
        # the trend fit's own uncertainty over the gap and most of the
        # fitted move we deliberately did not centre — so a genuinely
        # trending series widens instead of being missed, while a flat
        # history (both terms exactly zero) stays at the floor.
        s_i, n_i = sigmas[i] if i < len(sigmas) else (None, 0)
        if s_i is not None:
            s_comb = (n_i * s_i + POOL_K * pool_sigma) / (n_i + POOL_K)
        else:
            s_comb = pool_sigma
        var = (SIGMA_INFLATE * s_comb * math.sqrt(gap_h)) ** 2
        var += se_move * se_move
        var += (MOVE_WIDEN * min(abs(full_move), MOVE_CAP)) ** 2
        sigma = math.sqrt(var)
        if sigma < SIGMA_FLOOR:
            sigma = SIGMA_FLOOR

        qs = []
        for z in zs:
            qs.append(finish_pct(inv_logit_pct(mu + z * sigma)))
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_case(case):
    """Last-resort persistence rows: never crash, never emit an invalid case."""
    trophies = case.get("trophies") or []
    levels = case.get("levels") or DEFAULT_LEVELS
    zs = [_NORM.inv_cdf(float(tau)) for tau in levels]
    history = case.get("history") or []
    rows = []
    for i, trophy in enumerate(trophies):
        r = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                r = float(rates[i])
                break
        if r is None:
            r = 1.0
        mu = logit_prob(r / 100.0)
        qs = []
        for z in zs:
            qs.append(finish_pct(inv_logit_pct(mu + z * SIGMA_FLOOR)))
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            out = forecast_case(case)
        except Exception:
            out = fallback_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
