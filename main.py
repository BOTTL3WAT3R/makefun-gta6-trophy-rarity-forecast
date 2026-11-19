import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
HALF_LIFE_H = 18.0
SHRINK_M = 6.0
DAMP = 0.15
MOVE_SPREAD = 0.3
RW = 0.3
SPREAD_FLOOR = 0.08


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
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


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def wls(xs, ys, ws):
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0, mx, my, sxx)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return (my - b * mx, b, mx, my, sxx)


def resid_std(xs, ys, ws, a, b):
    sw = sum(ws)
    if sw <= 0 or len(xs) < 3:
        return None
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    return math.sqrt(max(var, 0.0))


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("empty history")
    try:
        origin = datetime.fromisoformat(case["game"]["release_date"]).replace(tzinfo=timezone.utc)
    except Exception:
        origin = parse_ts(history[0]["at"]).replace(hour=0, minute=0, second=0, microsecond=0)

    def hours(ts):
        return (ts - origin).total_seconds() / 3600.0

    def u_of(ts):
        return math.log(1.0 + max(0.0, hours(ts)))

    target_ts = parse_ts(case["target"])
    u_t = u_of(target_ts)
    horizon_days = case["horizon_minutes"] / 1440.0

    states = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(-1.0 if r is None else r for r in read["rates"]))
        if key != prev_key:
            states.append(read)
            prev_key = key
    if not states:
        states = [history[-1]]

    times = [parse_ts(r["at"]) for r in states]
    us = [u_of(t) for t in times]
    t_last = times[-1]
    weights = []
    for t in times:
        age = (t_last - t).total_seconds() / 3600.0
        weights.append(max(0.02, 0.5 ** (age / HALF_LIFE_H)))
    players = [float(r.get("players_tracked") or 0) for r in states]
    last_read = history[-1]

    n_fit = None
    if len(states) >= 3 and all(p > 0 for p in players):
        n_fit = wls(us, [math.log(p) for p in players], weights)
    if n_fit is not None:
        log_n_t = n_fit[0] + n_fit[1] * u_t
        log_n_t = min(log_n_t, math.log(players[-1]) + 2.5)
        n_t = math.exp(log_n_t)
    else:
        n_t = players[-1] if players[-1] > 0 else 1.0

    n_tr = len(trophies)
    series = []
    for i in range(n_tr):
        pts = []
        for k, read in enumerate(states):
            r = read["rates"][i]
            if r is not None:
                pts.append((us[k], float(r), players[k], weights[k]))
        series.append(pts)

    fits_a = []
    slopes_a = []
    slopes_b = []
    raw_b = []
    for pts in series:
        if len(pts) >= 3 and len({p[1] for p in pts}) >= 2:
            xs = [p[0] for p in pts]
            ys = [logit(p[1]) for p in pts]
            ws = [p[3] for p in pts]
            f = wls(xs, ys, ws)
            fits_a.append(f)
            if f is not None:
                slopes_a.append(f[1])
            hb = wls([p[0] for p in pts], [math.log(max(p[1] * p[2] / 100.0, 0.35)) for p in pts], ws)
            raw_b.append(hb)
            if hb is not None:
                slopes_b.append(hb[1])
        else:
            fits_a.append(None)
            raw_b.append(None)
    pool_a = median(slopes_a) if slopes_a else 0.0
    pool_b = median(slopes_b) if slopes_b else 0.0
    if n_fit is not None:
        pool_b_net = pool_b - n_fit[1]
    else:
        pool_b_net = pool_b

    resids = []
    for i, pts in enumerate(series):
        f = fits_a[i]
        if f is not None and len(pts) >= 4:
            rs = resid_std([p[0] for p in pts], [logit(p[1]) for p in pts], [p[3] for p in pts], f[0], f[1])
            if rs is not None:
                resids.append(rs)
    pool_resid = median(resids) if resids else 0.25
    pool_resid = max(0.12, pool_resid)

    last_rates = []
    for i in range(n_tr):
        r = last_read["rates"][i]
        if r is None:
            for read in reversed(history):
                if read["rates"][i] is not None:
                    r = read["rates"][i]
                    break
        last_rates.append(r)
    known = [r for r in last_rates if r is not None]
    global_mu = logit(median(known)) if known else logit(1.0)

    rows = []
    du = max(0.0, u_t - us[-1])
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        mu_last = logit(r_last) if r_last is not None else global_mu
        pts = series[i]
        m = float(len(pts))
        shrink = m / (m + SHRINK_M) if m > 0 else 0.0
        preds = []
        f = fits_a[i]
        if f is not None:
            b = pool_a + (f[1] - pool_a) * shrink
            a = f[3] - b * f[2]
            preds.append(a + b * u_t)
        hb = raw_b[i]
        if hb is not None and n_fit is not None:
            b2 = pool_b + (hb[1] - pool_b) * shrink
            a2 = hb[3] - b2 * hb[2]
            log_h = a2 + b2 * u_t
            log_h = min(log_h, math.log(n_t) - 0.0001)
            rate_b = 100.0 * math.exp(log_h) / n_t
            preds.append(logit(rate_b))
        if preds:
            mu = sum(mu_last + DAMP * (p - mu_last) for p in preds) / len(preds)
        else:
            mu = mu_last
        mu = min(mu_last + 3.5, max(mu_last - 3.5, mu))
        mu = min(logit(99.99), max(logit(0.01), mu))

        rs = None
        if f is not None and len(pts) >= 4:
            rs = resid_std([p[0] for p in pts], [logit(p[1]) for p in pts], [p[3] for p in pts], f[0], f[1])
        r_i = rs if rs is not None else pool_resid
        sigma = math.sqrt((RW * math.sqrt(horizon_days)) ** 2 + r_i * r_i + SPREAD_FLOOR ** 2 + (MOVE_SPREAD * abs(mu - mu_last)) ** 2)
        qs = []
        for z in Z:
            q = inv_logit(mu + z * sigma)
            q = min(100.0, max(0.0, q))
            qs.append(round4(q))
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") != LEVELS:
            raise ValueError("unexpected levels")
        out = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
