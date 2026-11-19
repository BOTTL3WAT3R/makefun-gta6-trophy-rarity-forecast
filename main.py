import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Forecaster v3 — counts-space, law-selected, refresh-aware, damped.
#
# What the first two versions taught (server practice verdicts: v1 loss
# 1.403, v2 loss 1.569, house persistence baseline ~1.51):
#   * The kit practice histories are byte-identical stale snapshots: the
#     page never refreshed inside either history, and the scored answers
#     are the refreshed page. No law can be selected from those histories;
#     they contain zero within-history signal.
#   * The case that pays, f00, is the opposite regime: ~2 days of reads
#     every 10 minutes with real movement, and a 4.52-day horizon. There
#     the signal is the joint growth of players_tracked and per-trophy
#     earners, so v3 forecasts the two counts separately and divides.
#
# Design:
#   * Collapse the history to distinct page states (Exophase refreshes on
#     its own clock; consecutive identical reads are one observation).
#   * Three candidate laws for the median: persistence; a logit-rate line
#     on ln(1 + hours since release) (the house curve's shape, undamped);
#     and a counts model — log players on ln(1 + hours), per-trophy log
#     earners on ln(1 + hours) with slopes shrunk to the pooled slope.
#   * Choose the law by rolling-origin backtest inside the case's own
#     history (pooled across trophies; per-trophy override only on a
#     clear backtest win). With too few distinct states to backtest, fall
#     back to persistence with a wide, refresh-aware spread: a flat page
#     is a stale page, and the truth has been moving unobserved.
#   * DAMP the chosen law's move: the median takes only DAMP of the
#     law's logit distance from the last rate, capped at MOVE_CAP. A
#     regime tournament over synthetic saturating / growing / diluting
#     worlds showed the undamped law wins big when the world is exactly
#     its shape and loses catastrophically when growth saturates;
#     damping keeps most of the win and removes the blow-up.
#   * Spread from the chosen law's backtest residuals (random-walk scaled
#     to the forecast gap, shrunk to the pooled residual), floored at
#     SIGMA_FLOOR_B * sqrt(horizon days): a 4.5-day horizon is never
#     forecast tighter than that, whatever the in-window fit says.

EPS = 1e-6                # probability clip before any logit transform
ANCHOR = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)  # release, house curve's origin
FLAT_SIGMA = 0.60         # spread when the page never refreshed in-window (stale page)
DAMP = 0.40               # fraction of the chosen law's logit move the median takes
MOVE_CAP = 1.0            # max |logit move| of the law before damping
SIGMA_FLOOR_A = 0.06      # horizon floor: sigma >= A + B * sqrt(horizon days)
SIGMA_FLOOR_B = 0.16
SIGMA_HARD_FLOOR_B = 0.40 # sigma >= B * sqrt(horizon days), whatever the fit says
SIGMA_CAP = 1.60
POOL_K = 4.0              # shrinkage weight toward pooled estimates
OVERRIDE_RATIO = 0.80     # per-trophy law override needs a clear backtest win
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


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def page_states(case):
    """Collapse history to distinct page states.

    A state is (x, players, rates) with x = ln(1 + hours since ANCHOR).
    Consecutive reads with identical players and rates are one state:
    the page only changes when Exophase refreshes it. Nulls inside a
    read are kept as None; states are compared on their raw contents."""
    states = []
    for read in case.get("history") or []:
        try:
            t = parse_ts(read["at"])
        except Exception:
            continue
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        key = (players, tuple(rates))
        if states and states[-1][3] == key:
            continue
        hours = max(0.0, (t - ANCHOR).total_seconds() / 3600.0)
        states.append((math.log1p(hours), players, list(rates), key, hours))
    return states


def last_rate(states, i):
    for st in reversed(states):
        rates = st[2]
        if i < len(rates) and rates[i] is not None:
            return float(rates[i])
    return None


def law_persistence(states, i, x_t):
    r = last_rate(states, i)
    return r


def law_logit_trend(states, i, x_t):
    """Logit-rate line on ln(1 + hours since release), undamped.

    Uses states from the last 24 hours of recorded time when at least 3
    usable points remain there, else all states (the house curve's
    window). Falls back to persistence when the line is not fittable."""
    pts = []
    for st in states:
        rates = st[2]
        if i < len(rates) and rates[i] is not None:
            pts.append((st[0], logit_prob(float(rates[i]) / 100.0), st[4]))
    if len(pts) < 3:
        return law_persistence(states, i, x_t)
    h_last = pts[-1][2]
    recent = [(x, y) for x, y, h in pts if h_last - h <= 24.0]
    use = recent if len(recent) >= 3 else [(x, y) for x, y, _ in pts]
    if len({y for _, y in use}) < 2:
        return law_persistence(states, i, x_t)
    fit = ols([x for x, _ in use], [y for _, y in use])
    if fit is None:
        return law_persistence(states, i, x_t)
    a, b = fit
    return inv_logit_pct(a + b * x_t)


def players_forecast(states, x_t):
    """Power-law players growth: OLS of ln players on ln(1 + hours)."""
    pts = [(st[0], math.log(st[1])) for st in states if st[1]]
    if len(pts) < 2:
        return float(states[-1][1]) if states and states[-1][1] else None
    fit = ols([x for x, _ in pts], [y for _, y in pts])
    if fit is None:
        return float(states[-1][1])
    a, b = fit
    return math.exp(a + b * x_t)


def law_counts(states, i, x_t, pooled_slope):
    """Forecast earners and players separately; rate = 100 * E / P.

    Earners are cumulative counts: e = rate * players / 100. Per-trophy
    OLS of ln(max(e, 0.25)) on ln(1 + hours), the slope shrunk toward
    the pooled cross-trophy slope (small-count trophies pool hardest)."""
    p_hat = players_forecast(states, x_t)
    r_last = last_rate(states, i)
    if p_hat is None or p_hat <= 0:
        return r_last
    pts = []
    for st in states:
        rates = st[2]
        if st[1] and i < len(rates) and rates[i] is not None:
            e = float(rates[i]) * float(st[1]) / 100.0
            pts.append((st[0], math.log(max(e, 0.25)), e))
    if len(pts) < 2 or r_last is None:
        return r_last
    fit = ols([x for x, _, _ in pts], [y for _, y, _ in pts])
    if fit is None:
        return r_last
    a, b = fit
    e_last = pts[-1][2]
    w = e_last / (e_last + 8.0)          # shrink small counts to the pool
    b_shrunk = w * b + (1.0 - w) * pooled_slope
    # Re-anchor the line at the last observed earners with the shrunk
    # slope, so the level is the data's, not the fit's intercept drift.
    x_last = pts[-1][0]
    ln_e = math.log(max(e_last, 0.25)) + b_shrunk * (x_t - x_last)
    e_hat = math.exp(ln_e)
    e_hat = min(e_hat, p_hat)            # earners cannot exceed players
    return 100.0 * e_hat / p_hat


def pooled_earner_slope(states, n_tr):
    slopes = []
    for i in range(n_tr):
        pts = []
        for st in states:
            rates = st[2]
            if st[1] and i < len(rates) and rates[i] is not None:
                e = float(rates[i]) * float(st[1]) / 100.0
                pts.append((st[0], math.log(max(e, 0.25))))
        if len(pts) >= 2:
            fit = ols([x for x, _ in pts], [y for _, y in pts])
            if fit is not None:
                slopes.append(fit[1])
    return median(slopes) if slopes else 0.0


LAWS = ("persist", "trend", "counts")


def law_predict(law, states, i, x_t, pooled_slope):
    if law == "trend":
        return law_logit_trend(states, i, x_t)
    if law == "counts":
        return law_counts(states, i, x_t, pooled_slope)
    return law_persistence(states, i, x_t)


def backtest(states, n_tr, pooled_slope):
    """Rolling-origin backtest over distinct states.

    Origins are states 2..S-2 (0-based, needs >= 2 fits points and a
    following state to score); each law is fit on states up to the
    origin and scored on the next state's rates, in logit space.
    Returns (pooled_err, per_trophy_err, n_points) with errors as mean
    |logit(y) - logit(pred)| per law; empty dicts when not runnable."""
    S = len(states)
    if S < 4:
        return {}, {}, 0
    pooled = {law: [] for law in LAWS}
    per = {law: [[] for _ in range(n_tr)] for law in LAWS}
    for origin in range(2, S - 1):
        sub = states[: origin + 1]
        nxt = states[origin + 1]
        x_n = nxt[0]
        sub_pool = pooled_earner_slope(sub, n_tr)
        for law in LAWS:
            for i in range(n_tr):
                rates = nxt[2]
                if i >= len(rates) or rates[i] is None:
                    continue
                pred = law_predict(law, sub, i, x_n, sub_pool)
                if pred is None:
                    continue
                err = abs(logit_prob(float(rates[i]) / 100.0) - logit_prob(pred / 100.0))
                pooled[law].append(err)
                per[law][i].append(err)
    pooled_err = {law: (sum(v) / len(v) if v else None) for law, v in pooled.items()}
    per_err = {
        law: [(sum(v) / len(v) if v else None) for v in per[law]] for law in LAWS
    }
    n_points = len(pooled[LAWS[0]])
    return pooled_err, per_err, n_points


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    levels = case.get("levels") or DEFAULT_LEVELS
    zs = [_NORM.inv_cdf(float(tau)) for tau in levels]
    states = page_states(case)

    try:
        target_ts = parse_ts(case["target"])
        target_h = max(0.0, (target_ts - ANCHOR).total_seconds() / 3600.0)
    except Exception:
        last_h = states[-1][4] if states else 0.0
        target_h = last_h + float(case.get("horizon_minutes") or 0) / 60.0
    x_t = math.log1p(target_h)
    horizon_days = max(0.0, float(case.get("horizon_minutes") or 0)) / 1440.0

    pooled_slope = pooled_earner_slope(states, n_tr)
    pooled_err, per_err, n_bt = backtest(states, n_tr, pooled_slope)

    # Pooled law choice: lowest backtest error; ties break toward the
    # counts model (the final's regime), then trend, then persistence.
    best_law = "persist"
    if n_bt > 0:
        order = {"counts": 0, "trend": 1, "persist": 2}
        cands = [(pooled_err[law], order[law], law) for law in LAWS if pooled_err.get(law) is not None]
        if cands:
            cands.sort()
            best_law = cands[0][2]

    last_rates = [last_rate(states, i) for i in range(n_tr)]
    known = [r for r in last_rates if r is not None]
    global_last = median(known) if known else 1.0
    flat = len(states) <= 1

    # Residual scale of the chosen law, for the spread: reuse the
    # backtest errors (they are mean absolute logit errors; a normal's
    # sigma is ~1.25x its mean absolute deviation), random-walk scaled
    # from the backtest gap to this case's gap where both are known.
    bt_sigma = None
    if n_bt > 0 and pooled_err.get(best_law) is not None:
        bt_sigma = 1.25 * pooled_err[best_law]

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i] if last_rates[i] is not None else global_last

        law = best_law
        if n_bt > 0 and per_err:
            base_err = per_err.get(best_law, [None] * n_tr)[i]
            if base_err is not None:
                for other in LAWS:
                    if other == best_law:
                        continue
                    e = per_err.get(other, [None] * n_tr)[i]
                    if e is not None and e < OVERRIDE_RATIO * base_err:
                        law = other
                        break

        pred = law_predict(law, states, i, x_t, pooled_slope)
        if pred is None:
            pred = r_last
        # Damp the law's move toward persistence (see header): take only
        # DAMP of its logit distance from the last rate, capped.
        mu_last = logit_prob(r_last / 100.0)
        move = logit_prob(pred / 100.0) - mu_last
        move = max(-MOVE_CAP, min(MOVE_CAP, move))
        mu = mu_last + DAMP * move

        if flat:
            sigma = max(FLAT_SIGMA, 0.5 * math.sqrt(horizon_days))
        else:
            sigma = SIGMA_FLOOR_A + SIGMA_FLOOR_B * math.sqrt(horizon_days)
            if bt_sigma is not None:
                sigma = max(sigma, bt_sigma)
            s_i = None
            if n_bt > 0 and per_err:
                e = per_err.get(law, [None] * n_tr)[i]
                if e is not None:
                    s_i = 1.25 * e
            if s_i is not None:
                sigma = (3.0 * s_i + POOL_K * sigma) / (3.0 + POOL_K)
            sigma = max(sigma, SIGMA_HARD_FLOOR_B * math.sqrt(horizon_days))
            sigma = min(max(sigma, SIGMA_FLOOR_A), SIGMA_CAP)

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
            qs.append(finish_pct(inv_logit_pct(mu + z * FLAT_SIGMA)))
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
