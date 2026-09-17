# Alertas Predictivas (Risk Score) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Risk Score (0-100) system to `financial_tracker.py` that detects upcoming events (earnings) and technical risk patterns, displayed as a new section in the daily report and deep dive.

**Architecture:** Two detector functions (`_detect_upcoming_events`, `_detect_technical_risks`) produce alert dicts. A scorer (`_calculate_risk_score`) sums points into a 0-100 score with semaphore level. Results integrate into `generate_report()` as a new section and into each portfolio/watchlist card. `fetch_technicals()` is extended to return the `hist` DataFrame and additional indicators (SMA200, MACD, Bollinger) needed by the technical detector.

**Tech Stack:** Python 3, yfinance, pandas, numpy (all already installed).

---

### Task 1: Extend `fetch_technicals()` to return additional data

**Files:**
- Modify: `financial_tracker.py` — function `fetch_technicals()` (lines 318-368)

The technical risk detector needs SMA200, MACD, Bollinger Bands, and the raw `hist` DataFrame. Currently these are only computed in `_compute_advanced_technicals()` for deep dive. We add them to `fetch_technicals()`.

- [ ] **Step 1: Add SMA200, MACD, Bollinger, and hist to fetch_technicals return dict**

In `financial_tracker.py`, replace the `fetch_technicals` function (lines 318-368) with this updated version. The changes are: (1) compute SMA200, (2) compute MACD line/signal/histogram, (3) compute Bollinger Bands, (4) include `hist` DataFrame in return dict.

Find this block inside `fetch_technicals` (after `ema26` computation, around line 346):

```python
        # Volume Profile — zonas de alto volumen (S/R)
        vol_zones = _volume_profile(df)
        vol_sr    = _classify_volume_zones(vol_zones, price)

        return {
            "ticker":       ticker,
            "price":        price,
            "chg_1d":       chg_1d,
            "chg_1w":       chg_1w,
            "chg_1m":       chg_1m,
            "rsi":          rsi,
            "sma_s":        sma_s,
            "sma_l":        sma_l,
            "ema12":        ema12,
            "ema26":        ema26,
            "signals":      signals,
            "vol_supports":     vol_sr["supports"],
            "vol_resistances":  vol_sr["resistances"],
        }
```

Replace with:

```python
        # SMA 200
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # MACD (12, 26, 9)
        macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist_s = macd_line - macd_signal
        macd_val  = float(macd_line.iloc[-1])
        macd_sig  = float(macd_signal.iloc[-1])
        macd_h    = float(macd_hist_s.iloc[-1])
        macd_h_prev = float(macd_hist_s.iloc[-2]) if len(macd_hist_s) >= 2 else 0.0

        # Bollinger Bands (20, 2)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = float((bb_mid + 2 * bb_std).iloc[-1])
        bb_lower = float((bb_mid - 2 * bb_std).iloc[-1])
        bb_pct   = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5

        # Volume Profile — zonas de alto volumen (S/R)
        vol_zones = _volume_profile(df)
        vol_sr    = _classify_volume_zones(vol_zones, price)

        return {
            "ticker":       ticker,
            "price":        price,
            "chg_1d":       chg_1d,
            "chg_1w":       chg_1w,
            "chg_1m":       chg_1m,
            "rsi":          rsi,
            "sma_s":        sma_s,
            "sma_l":        sma_l,
            "sma200":       sma200,
            "ema12":        ema12,
            "ema26":        ema26,
            "macd":         macd_val,
            "macd_signal":  macd_sig,
            "macd_hist":    macd_h,
            "macd_hist_prev": macd_h_prev,
            "bb_upper":     bb_upper,
            "bb_lower":     bb_lower,
            "bb_pct":       bb_pct,
            "signals":      signals,
            "vol_supports":     vol_sr["supports"],
            "vol_resistances":  vol_sr["resistances"],
            "hist":         df,
        }
```

- [ ] **Step 2: Verify the script still runs without errors**

Run: `python financial_tracker.py --ticker AAPL`

Expected: Deep dive generates successfully. The new fields don't break existing code because all consumers access dict keys they know about.

---

### Task 2: Implement `_detect_upcoming_events()`

**Files:**
- Modify: `financial_tracker.py` — add new function after `_classify_volume_zones()` (around line 316)

- [ ] **Step 1: Add the event detector function**

Insert this function after `_classify_volume_zones()` and before `fetch_technicals()`:

```python
# ─── Alertas predictivas ────────────────────────────────────────────────────

def _detect_upcoming_events(ticker: str) -> list[dict]:
    """Detecta eventos próximos (earnings, ex-dividend) y evalúa riesgo."""
    alerts = []
    try:
        tk = yf.Ticker(ticker)
    except Exception:
        return alerts

    today = datetime.now()

    # ── Earnings dates ──────────────────────────────────────
    try:
        ed = tk.earnings_dates
        if ed is not None and not ed.empty:
            # Filtrar fechas futuras y pasadas recientes
            for dt_idx in ed.index:
                dt = dt_idx.to_pydatetime()
                if dt.tzinfo:
                    dt = dt.replace(tzinfo=None)
                days_until = (dt.date() - today.date()).days

                # Earnings próximos (0 a 7 días)
                if 0 <= days_until <= 7:
                    if days_until <= 1:
                        pts = 30
                        msg = f"📅 Earnings MAÑANA ({dt.strftime('%Y-%m-%d')}) — máxima volatilidad esperada"
                    elif days_until <= 3:
                        pts = 25
                        msg = f"📅 Earnings en {days_until} días ({dt.strftime('%Y-%m-%d')}) — volatilidad alta"
                    else:
                        pts = 15
                        msg = f"📅 Earnings en {days_until} días ({dt.strftime('%Y-%m-%d')}) — precaución"
                    alerts.append({
                        "type": "event",
                        "signal": "earnings_soon",
                        "message": msg,
                        "points": pts,
                    })
                    break  # Solo la fecha más cercana

                # Earnings recién pasados (volatilidad post-earnings)
                elif -2 <= days_until < 0:
                    alerts.append({
                        "type": "event",
                        "signal": "earnings_recent",
                        "message": f"📅 Earnings fue hace {abs(days_until)} día(s) — volatilidad post-earnings activa",
                        "points": 10,
                    })
                    break
    except Exception:
        pass

    # ── Historial de sorpresas (earnings history) ──────────
    try:
        eh = tk.earnings_history
        if eh is not None and not eh.empty:
            # Contar cuántos de los últimos 4 quarters tuvieron miss
            recent_q = eh.tail(4)
            if "epsActual" in recent_q.columns and "epsEstimate" in recent_q.columns:
                misses = 0
                for _, row in recent_q.iterrows():
                    actual = row.get("epsActual")
                    estimate = row.get("epsEstimate")
                    if actual is not None and estimate is not None:
                        if actual < estimate:
                            misses += 1

                if misses >= 3:
                    alerts.append({
                        "type": "event",
                        "signal": "earnings_miss_history",
                        "message": f"📅 Historial: falló estimados {misses}/4 trimestres — patrón de misses",
                        "points": 15,
                    })
                elif misses >= 2:
                    alerts.append({
                        "type": "event",
                        "signal": "earnings_miss_history",
                        "message": f"📅 Historial: falló estimados {misses}/4 trimestres",
                        "points": 10,
                    })
                elif misses == 0:
                    alerts.append({
                        "type": "event",
                        "signal": "earnings_beat_history",
                        "message": f"📅 Historial: superó estimados 4/4 trimestres — track record sólido",
                        "points": -10,
                    })
    except Exception:
        pass

    # ── Ex-dividend ───────────────────────────────────────
    try:
        cal = tk.calendar
        if cal is not None:
            ex_div = None
            if isinstance(cal, pd.DataFrame) and "Ex-Dividend Date" in cal.columns:
                ex_div = cal["Ex-Dividend Date"].iloc[0]
            elif isinstance(cal, dict) and "Ex-Dividend Date" in cal:
                ex_div = cal["Ex-Dividend Date"]

            if ex_div is not None:
                if hasattr(ex_div, "date"):
                    ex_date = ex_div.date()
                else:
                    ex_date = pd.to_datetime(ex_div).date()
                days_until = (ex_date - today.date()).days
                if 0 <= days_until <= 5:
                    alerts.append({
                        "type": "event",
                        "signal": "ex_dividend",
                        "message": f"📅 Ex-dividend en {days_until} día(s) ({ex_date}) — ajuste de precio esperado",
                        "points": 5,
                    })
    except Exception:
        pass

    return alerts
```

- [ ] **Step 2: Quick smoke test**

Run in Python interactively:

```bash
python -c "from financial_tracker import _detect_upcoming_events; print(_detect_upcoming_events('AAPL'))"
```

Expected: Returns a list (possibly empty if no events are near). No crashes.

---

### Task 3: Implement `_detect_technical_risks()`

**Files:**
- Modify: `financial_tracker.py` — add new function right after `_detect_upcoming_events()`

- [ ] **Step 1: Add the technical risk detector function**

Insert right after `_detect_upcoming_events()`:

```python
def _detect_technical_risks(tech_data: dict) -> list[dict]:
    """Detecta patrones técnicos de riesgo en los datos de un ticker."""
    alerts = []
    if not tech_data or "error" in tech_data:
        return alerts

    price = tech_data["price"]
    rsi   = tech_data["rsi"]
    hist  = tech_data.get("hist")

    # ── 1. RSI extremo (>80) ────────────────────────────────
    if rsi > 80:
        alerts.append({
            "type": "technical",
            "signal": "rsi_extreme_high",
            "message": f"📉 RSI extremo ({rsi:.1f}) — sobrecompra severa, alto riesgo de corrección",
            "points": 10,
        })
    elif rsi < 20:
        alerts.append({
            "type": "technical",
            "signal": "rsi_extreme_low",
            "message": f"📉 RSI extremo bajo ({rsi:.1f}) — sobreventa severa, puede continuar cayendo",
            "points": 5,
        })

    # ── 2. Pérdida de SMA 200 ────────────────────────────────
    sma200 = tech_data.get("sma200")
    if sma200 is not None and price < sma200:
        pct_below = (price / sma200 - 1) * 100
        alerts.append({
            "type": "technical",
            "signal": "below_sma200",
            "message": f"📉 Precio {pct_below:.1f}% debajo de SMA 200 (${sma200:,.2f}) — tendencia largo plazo dañada",
            "points": 15,
        })

    # ── 3. Death Cross activo (SMA20 < SMA50) ────────────────
    sma_s = tech_data["sma_s"]
    sma_l = tech_data["sma_l"]
    if sma_s < sma_l:
        alerts.append({
            "type": "technical",
            "signal": "death_cross_active",
            "message": f"📉 Death Cross activo — SMA20 (${sma_s:,.2f}) por debajo de SMA50 (${sma_l:,.2f})",
            "points": 10,
        })

    # ── 4. MACD cruce bajista reciente ───────────────────────
    macd_h = tech_data.get("macd_hist", 0)
    macd_h_prev = tech_data.get("macd_hist_prev", 0)
    if macd_h < 0 and macd_h_prev > 0:
        alerts.append({
            "type": "technical",
            "signal": "macd_bearish_cross",
            "message": f"📉 MACD cruzó a bajista — histograma pasó de positivo a negativo",
            "points": 10,
        })

    # ── 5. Bollinger Squeeze ────────────────────────────────
    bb_upper = tech_data.get("bb_upper", 0)
    bb_lower = tech_data.get("bb_lower", 0)
    if price > 0 and bb_upper > 0 and bb_lower > 0:
        bb_width = (bb_upper - bb_lower) / price
        if bb_width < 0.05:
            alerts.append({
                "type": "technical",
                "signal": "bollinger_squeeze",
                "message": f"📉 Bollinger Squeeze detectado (ancho {bb_width:.1%}) — explosión de volatilidad inminente",
                "points": 10,
            })

    # ── 6. Divergencia bajista RSI ────────────────────────────
    if hist is not None and len(hist) >= 20:
        try:
            close_s = hist["Close"].squeeze()
            rsi_s = _rsi(close_s, RSI_PERIOD)

            recent = close_s.tail(20)
            rsi_recent = rsi_s.tail(20)
            first_half_close = recent.head(10)
            second_half_close = recent.tail(10)
            first_half_rsi = rsi_recent.head(10)
            second_half_rsi = rsi_recent.tail(10)

            price_high_1 = float(first_half_close.max())
            price_high_2 = float(second_half_close.max())
            rsi_at_high_1 = float(first_half_rsi.iloc[first_half_close.values.argmax()])
            rsi_at_high_2 = float(second_half_rsi.iloc[second_half_close.values.argmax()])

            if price_high_2 > price_high_1 and rsi_at_high_2 < rsi_at_high_1 - 2:
                alerts.append({
                    "type": "technical",
                    "signal": "rsi_bearish_divergence",
                    "message": f"📉 Divergencia bajista RSI — precio hizo nuevo máximo pero RSI no confirma",
                    "points": 20,
                })
        except Exception:
            pass

    # ── 7. Volumen decreciente en rally ────────────────────────
    if hist is not None and len(hist) >= 20:
        try:
            close_s = hist["Close"].squeeze()
            vol_s = hist["Volume"].squeeze()
            price_chg_20d = float((close_s.iloc[-1] / close_s.iloc[-20] - 1))

            vol_avg_10d = float(vol_s.tail(10).mean())
            vol_avg_20d = float(vol_s.tail(20).mean())

            if price_chg_20d > 0.10 and vol_avg_10d < vol_avg_20d * 0.8:
                alerts.append({
                    "type": "technical",
                    "signal": "volume_divergence",
                    "message": f"📉 Volumen decreciente en rally — precio subió {price_chg_20d:.0%} pero volumen cae",
                    "points": 10,
                })
        except Exception:
            pass

    # ── 8. Precio cerca de resistencia fuerte ─────────────────
    resistances = tech_data.get("vol_resistances", [])
    for r in resistances:
        if r.get("zone_type") != "current" and 0 < r["dist_pct"] < 2:
            alerts.append({
                "type": "technical",
                "signal": "near_resistance",
                "message": f"📉 Resistencia fuerte a {r['dist_pct']:.1f}% ({r['range']}) — riesgo de rechazo",
                "points": 5,
            })
            break  # Solo la más cercana

    # ── 9. Precio lejos de soporte ────────────────────────────
    supports = tech_data.get("vol_supports", [])
    nearest_support_dist = None
    for s in supports:
        if s.get("zone_type") != "current":
            nearest_support_dist = abs(s["dist_pct"])
            break
    if nearest_support_dist is not None and nearest_support_dist > 15:
        alerts.append({
            "type": "technical",
            "signal": "far_from_support",
            "message": f"📉 Sin soporte cercano — soporte más próximo a -{nearest_support_dist:.1f}%",
            "points": 5,
        })

    return alerts
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
from financial_tracker import fetch_technicals, _detect_technical_risks
tech = fetch_technicals('AAPL')
alerts = _detect_technical_risks(tech)
for a in alerts:
    print(f'{a[\"points\"]:+d} | {a[\"message\"]}')
"
```

Expected: Prints any detected alerts for AAPL. No crashes.

---

### Task 4: Implement `_calculate_risk_score()`

**Files:**
- Modify: `financial_tracker.py` — add function right after `_detect_technical_risks()`

- [ ] **Step 1: Add the risk score calculator**

```python
def _calculate_risk_score(alerts: list[dict]) -> tuple[int, str, list[dict]]:
    """Combina alertas en un Risk Score 0-100 con nivel semáforo."""
    raw_score = sum(a["points"] for a in alerts)
    score = max(0, min(raw_score, 100))  # Clamp 0-100

    if score <= 25:
        level = "🟢 BAJO"
    elif score <= 50:
        level = "🟡 MODERADO"
    elif score <= 75:
        level = "🔴 ALTO"
    else:
        level = "🔴🔴 CRÍTICO"

    sorted_alerts = sorted(alerts, key=lambda x: x["points"], reverse=True)
    return score, level, sorted_alerts


def _risk_action(score: int) -> str:
    """Retorna acción sugerida basada en el risk score."""
    if score <= 25:
        return ""
    elif score <= 50:
        return "Monitorear — precaución en nuevas entradas"
    elif score <= 75:
        return "Riesgo elevado — considerar stop loss ajustado o esperar"
    else:
        return "NO ENTRAR — múltiples alertas activas. Esperar estabilización"
```

- [ ] **Step 2: Integration smoke test — full pipeline for one ticker**

```bash
python -c "
from financial_tracker import fetch_technicals, _detect_upcoming_events, _detect_technical_risks, _calculate_risk_score
tech = fetch_technicals('TSLA')
events = _detect_upcoming_events('TSLA')
tech_risks = _detect_technical_risks(tech)
all_alerts = events + tech_risks
score, level, alerts = _calculate_risk_score(all_alerts)
print(f'TSLA: {score}/100 {level}')
for a in alerts:
    print(f'  {a[\"points\"]:+d} | {a[\"message\"]}')
"
```

Expected: Prints risk score and alert details for TSLA. No crashes.

---

### Task 5: Integrate alerts into `generate_report()`

**Files:**
- Modify: `financial_tracker.py` — function `generate_report()` (lines 638-990)

- [ ] **Step 1: Compute risk scores during portfolio analysis**

In `generate_report()`, right after the portfolio analysis loop (after line 666), add risk score computation. Find this block:

```python
    print("⏳ Analizando portafolio...")
    portfolio_data = []
    for t in MY_PORTFOLIO:
        print(f"   → {t}")
        portfolio_data.append(fetch_technicals(t))
        time.sleep(0.4)
```

Replace with:

```python
    print("⏳ Analizando portafolio...")
    portfolio_data = []
    risk_scores = {}  # ticker -> (score, level, alerts)
    for t in MY_PORTFOLIO:
        print(f"   → {t}")
        tech = fetch_technicals(t)
        portfolio_data.append(tech)
        # Risk score
        events = _detect_upcoming_events(t)
        tech_risks = _detect_technical_risks(tech) if tech and "error" not in tech else []
        score, level, alerts = _calculate_risk_score(events + tech_risks)
        risk_scores[t] = (score, level, alerts)
        time.sleep(0.4)
```

- [ ] **Step 2: Compute risk scores for CSV watchlist tickers**

Find the CSV analysis loop:

```python
    csv_data = []
    if csv_tickers:
        print(f"⏳ Analizando watchlist.csv ({len(csv_tickers)} tickers)...")
        for t in csv_tickers:
            print(f"   → {t}")
            csv_data.append(fetch_technicals(t))
            time.sleep(0.3)
```

Replace with:

```python
    csv_data = []
    if csv_tickers:
        print(f"⏳ Analizando watchlist.csv ({len(csv_tickers)} tickers)...")
        for t in csv_tickers:
            print(f"   → {t}")
            tech = fetch_technicals(t)
            csv_data.append(tech)
            events = _detect_upcoming_events(t)
            tech_risks = _detect_technical_risks(tech) if tech and "error" not in tech else []
            score, level, alerts = _calculate_risk_score(events + tech_risks)
            risk_scores[t] = (score, level, alerts)
            time.sleep(0.3)
```

- [ ] **Step 3: Add the alerts section to the Markdown output**

Find the line that starts the portfolio section (around line 741):

```python
    # ── Sección 1: Mis Inversiones ───────────────────────────
    L += [
        "---",
        "",
        "## 💼 Sección 1 — Mis Inversiones",
```

Insert the alerts section BEFORE it:

```python
    # ── Alertas Predictivas ──────────────────────────────────
    # Filtrar tickers con alertas y ordenar por score descendente
    scored_tickers = sorted(risk_scores.items(), key=lambda x: x[1][0], reverse=True)
    has_alerts = [(t, s, l, a) for t, (s, l, a) in scored_tickers if s > 25]

    if has_alerts:
        L += [
            "## ⚠️ Alertas Predictivas",
            "",
            "| Ticker | Risk Score | Nivel | Alertas principales |",
            "|---|---:|---|---|",
        ]
        for ticker, score, level, alerts in has_alerts:
            top_alerts = " · ".join(a["message"].split(" — ")[0] for a in alerts[:3] if a["points"] > 0)
            L.append(f"| **{ticker}** | {score}/100 | {level} | {top_alerts} |")
        L.append("")

        # Detalle solo para ALTO y CRÍTICO
        high_alerts = [(t, s, l, a) for t, s, l, a in has_alerts if s > 50]
        for ticker, score, level, alerts in high_alerts:
            action = _risk_action(score)
            L += [
                f"### {level.split()[0]} {ticker} — {score}/100",
                "",
                "| Tipo | Alerta | Pts |",
                "|---|---|---:|",
            ]
            for a in alerts:
                if a["points"] > 0:
                    tipo = "📅 Evento" if a["type"] == "event" else "📉 Técnico"
                    L.append(f"| {tipo} | {a['message']} | +{a['points']} |")
            if action:
                L += ["", f"**⚡ Acción sugerida:** {action}", ""]
            L.append("")

        L += ["---", ""]
    else:
        L += [
            "## ⚠️ Alertas Predictivas",
            "",
            "> 🟢 Sin alertas significativas en el portafolio y watchlist.",
            "",
            "---",
            "",
        ]

    # ── Sección 1: Mis Inversiones ───────────────────────────
    L += [
```

Note: This replaces the existing `---` and `## 💼 Sección 1` lines — make sure to keep the portfolio section start intact.

- [ ] **Step 4: Add risk score to each portfolio card**

Find the block that writes signals for portfolio cards (around line 781):

```python
            "**Señales técnicas:**",
        ]
        for sig in data["signals"]:
            L.append(f"- {sig}")
        L.append("")
```

After `L.append("")`, add:

```python
        # Risk Score en la card
        ticker = data.get("ticker", "?")
        if ticker in risk_scores:
            rs_score, rs_level, rs_alerts = risk_scores[ticker]
            if rs_score > 25:
                top_reasons = ", ".join(a["message"].split(" — ")[0] for a in rs_alerts[:2] if a["points"] > 0)
                L.append(f"**Risk Score:** {rs_score}/100 {rs_level} — {top_reasons}")
                L.append("")
```

- [ ] **Step 5: Add risk score to each watchlist CSV card**

Same pattern — find the signals block for watchlist cards (around line 872):

```python
            "**Señales:**",
            ]
            for sig in data["signals"]:
                L.append(f"- {sig}")
            L.append("")
```

After `L.append("")`, add:

```python
            # Risk Score en la card
            if ticker in risk_scores:
                rs_score, rs_level, rs_alerts = risk_scores[ticker]
                if rs_score > 25:
                    top_reasons = ", ".join(a["message"].split(" — ")[0] for a in rs_alerts[:2] if a["points"] > 0)
                    L.append(f"**Risk Score:** {rs_score}/100 {rs_level} — {top_reasons}")
                    L.append("")
```

- [ ] **Step 6: Update console output to show alert count**

Find the final print block (around line 986):

```python
    print(f"\n✅ Reporte guardado: {filename}")
    print(f"   💼 {len([d for d in portfolio_data if d])} activos en portafolio")
    print(f"   🟢 {buy_count} señal(es) de compra  |  🔴 {sell_count} señal(es) de venta")
    print(f"   📰 {news_count} noticias recopiladas")
```

Replace with:

```python
    alert_count = len([t for t, (s, _, _) in risk_scores.items() if s > 50])
    print(f"\n✅ Reporte guardado: {filename}")
    print(f"   💼 {len([d for d in portfolio_data if d])} activos en portafolio")
    print(f"   🟢 {buy_count} señal(es) de compra  |  🔴 {sell_count} señal(es) de venta")
    print(f"   ⚠️ {alert_count} alerta(s) de riesgo alto")
    print(f"   📰 {news_count} noticias recopiladas")
```

- [ ] **Step 7: Run full report and verify**

Run: `python financial_tracker.py`

Expected: Report generates with new `⚠️ Alertas Predictivas` section between macro indicators and portfolio. Console shows alert count. No crashes.

---

### Task 6: Integrate alerts into `generate_deep_dive()`

**Files:**
- Modify: `financial_tracker.py` — function `generate_deep_dive()` (around line 1141+)

- [ ] **Step 1: Add risk score section to deep dive**

In `generate_deep_dive()`, find the block after the "Resumen de Señales" section (around line 1343) that starts the analyst recommendations:

```python
        L.append("")

    # ── Recomendaciones de analistas ──────────────────────────
    recs = data["recs"]
```

Insert the risk score section before the analyst recommendations:

```python
        L.append("")

    # ── Risk Score ────────────────────────────────────────────
    print("⏳ Calculando Risk Score...")
    dd_events = _detect_upcoming_events(ticker)
    # Build a tech_data dict compatible with _detect_technical_risks
    dd_tech_data = None
    if tech:
        dd_tech_data = {
            "price": tech["price"],
            "rsi": tech["rsi"],
            "sma_s": tech["sma20"],
            "sma_l": tech["sma50"],
            "sma200": tech.get("sma200"),
            "ema12": tech["ema12"],
            "ema26": tech["ema26"],
            "macd_hist": tech.get("macd_hist", 0),
            "macd_hist_prev": 0,  # Approximate: compute from close
            "bb_upper": tech.get("bb_upper", 0),
            "bb_lower": tech.get("bb_lower", 0),
            "bb_pct": tech.get("bb_pct", 0.5),
            "vol_supports": [],
            "vol_resistances": [],
            "hist": None,  # Deep dive downloads its own hist
        }
        # Try to build hist from the close series for divergence detection
        if not close.empty:
            # Build a minimal DataFrame with Close and Volume
            dd_hist = yf.download(ticker, period="6mo", interval="1d",
                                  progress=False, auto_adjust=True)
            if not dd_hist.empty:
                dd_tech_data["hist"] = dd_hist
                # Compute volume profile zones for risk detection
                vz = _volume_profile(dd_hist)
                vsr = _classify_volume_zones(vz, tech["price"])
                dd_tech_data["vol_supports"] = vsr["supports"]
                dd_tech_data["vol_resistances"] = vsr["resistances"]
                # Get macd_hist_prev
                close_s = dd_hist["Close"].squeeze()
                macd_line_s = close_s.ewm(span=12, adjust=False).mean() - close_s.ewm(span=26, adjust=False).mean()
                macd_signal_s = macd_line_s.ewm(span=9, adjust=False).mean()
                macd_hist_s = macd_line_s - macd_signal_s
                if len(macd_hist_s) >= 2:
                    dd_tech_data["macd_hist_prev"] = float(macd_hist_s.iloc[-2])

    dd_tech_risks = _detect_technical_risks(dd_tech_data) if dd_tech_data else []
    dd_all_alerts = dd_events + dd_tech_risks
    dd_score, dd_level, dd_alerts = _calculate_risk_score(dd_all_alerts)

    L += ["---", "", "## ⚠️ Risk Score", ""]
    L += [
        f"**{dd_score}/100 {dd_level}**",
        "",
    ]
    action = _risk_action(dd_score)
    if action:
        L.append(f"> ⚡ {action}")
        L.append("")

    if dd_alerts:
        L += [
            "| Tipo | Alerta | Pts |",
            "|---|---|---:|",
        ]
        for a in dd_alerts:
            tipo = "📅 Evento" if a["type"] == "event" else "📉 Técnico"
            sign = "+" if a["points"] > 0 else ""
            L.append(f"| {tipo} | {a['message']} | {sign}{a['points']} |")
        L.append("")
    else:
        L.append("> Sin alertas activas.")
        L.append("")

    # ── Recomendaciones de analistas ──────────────────────────
    recs = data["recs"]
```

- [ ] **Step 2: Test the deep dive with risk score**

Run: `python financial_tracker.py --ticker AAPL`

Expected: Deep dive generates with new `⚠️ Risk Score` section showing score, level, and alert table. No crashes.

- [ ] **Step 3: Test with a ticker that should have high risk**

Run: `python financial_tracker.py --ticker S`

Expected: SentinelOne should show alerts for earnings proximity, potential MACD/SMA signals. Verify the risk score reflects the current technical state.

---

### Task 7: End-to-end verification

**Files:** None (testing only)

- [ ] **Step 1: Run daily report**

```bash
python financial_tracker.py
```

Expected:
- Console shows `⚠️ N alerta(s) de riesgo alto`
- Generated .md file has `## ⚠️ Alertas Predictivas` section
- Portfolio cards with score > 25 show `**Risk Score:**` line
- No Python errors

- [ ] **Step 2: Run deep dive**

```bash
python financial_tracker.py --ticker SHOP
```

Expected:
- Deep dive has `## ⚠️ Risk Score` section with score/level/table
- No Python errors

- [ ] **Step 3: Verify edge cases — ETF without earnings**

```bash
python financial_tracker.py --ticker VOO
```

Expected: Risk score works even without earnings data (event detector returns empty, only technical alerts apply). No crashes.
