#!/usr/bin/env python3
"""
Financial Tracker — Daily Investment Report Generator
Genera un reporte .md con estado del portafolio e ideas de inversión
Ejecutar cuando sea necesario
"""

import io
import os
import re
import sys
import time
import warnings
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from bs4 import BeautifulSoup
import feedparser
import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Configuración ────────────────────────────────────────────────────────────

# Hugging Face — token gratuito: https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = "facebook/bart-large-cnn"  # Modelo de summarization gratuito
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

MY_PORTFOLIO = ["META", "AAPL", "AMZN", "VOO", "NVDA", "MSTR", "TSLA", "GOOGL", "QQQ"]

# Watchlist base (tickers de referencia)
_WATCHLIST_BASE = [
    # Tech
    "MSFT", "AMD", "INTC", "NFLX", "CRM", "ORCL", "UBER", "SHOP",
    # Semis
    "TSM", "ASML", "AVGO", "QCOM", "MU",
    # ETFs sectoriales
    "SPY", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "ARKK", "SOXX",
    # Bonds / Commodities
    "TLT", "GLD", "SLV", "GDX",
    # Crypto
    "BTC-USD", "ETH-USD",
]

# Ruta al archivo CSV de watchlist (Name,Symbol)
WATCHLIST_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.csv")


def _load_watchlist() -> list[str]:
    """Carga tickers combinando la lista base + watchlist.csv (si existe)."""
    tickers = list(_WATCHLIST_BASE)
    if os.path.isfile(WATCHLIST_CSV):
        try:
            df = pd.read_csv(WATCHLIST_CSV)
            col = "Symbol" if "Symbol" in df.columns else df.columns[-1]
            csv_tickers = [str(t).strip().upper() for t in df[col] if str(t).strip()]
            # Agregar solo los que no estén ya en la lista base ni en el portafolio
            existing = set(t.upper() for t in tickers + MY_PORTFOLIO)
            added = [t for t in csv_tickers if t not in existing]
            tickers.extend(added)
            print(f"   📂 watchlist.csv cargado: {len(added)} tickers nuevos añadidos")
        except Exception as e:
            print(f"   ⚠️ Error leyendo watchlist.csv: {e}")
    return tickers


WATCHLIST = _load_watchlist()

# Parámetros técnicos
RSI_PERIOD    = 14
SMA_SHORT     = 20
SMA_LONG      = 50
RSI_OVERSOLD  = 35
RSI_OVERBOUGHT = 65

# Fibonacci Screener
FIB_LEVEL = 0.618
FIB_MIN_SWING = 0.15
FIB_PROXIMITY = 0.03
FIB_EXTREMA_ORDER = 5

# Feeds RSS de noticias
NEWS_FEEDS = [
    ("Reuters Business",  "https://feeds.reuters.com/reuters/businessNews"),
    ("Yahoo Finance",     "https://finance.yahoo.com/rss/topfinstories"),
    ("Seeking Alpha",     "https://seekingalpha.com/market_currents.xml"),
    ("Investing.com",     "https://www.investing.com/rss/news_25.rss"),
]

# Palabras clave por ticker para filtrar noticias
TICKER_KEYWORDS = {
    "META":  ["meta", "facebook", "instagram", "whatsapp", "zuckerberg"],
    "AAPL":  ["apple", "iphone", "ipad", "macbook", "ios", "tim cook", "aapl"],
    "AMZN":  ["amazon", "aws", "bezos", "prime"],
    "VOO":   ["vanguard", "s&p 500", "sp500", "index fund"],
    "NVDA":  ["nvidia", "gpu", "cuda", "jensen huang", "blackwell"],
    "MSTR":  ["microstrategy", "bitcoin", "saylor", "btc"],
    "TSLA":  ["tesla", "elon musk", "electric vehicle", "cybertruck", "tsla"],
    "GOOGL": ["google", "alphabet", "youtube", "gemini", "waymo"],
    "QQQ":   ["nasdaq", "qqq", "nasdaq-100"],
}

# Indicadores FRED (series_id, nombre para mostrar)
FRED_SERIES = [
    ("FEDFUNDS", "Fed Funds Rate (%)"),
    ("CPIAUCSL", "CPI — Inflación"),
    ("UNRATE",   "Desempleo (%)"),
    ("DGS10",    "Treasury 10Y (%)"),
    ("DGS2",     "Treasury 2Y (%)"),
    ("T10Y2Y",   "Spread 10Y-2Y (%)"),
]

# Descripciones y lógica de análisis por indicador
FRED_DESCRIPTIONS: dict[str, dict] = {
    "FEDFUNDS": {
        "desc": "Tasa de interés de referencia de la Fed. Determina el costo del crédito en toda la economía.",
        "analyze": lambda v, chg: (
            f"Tasa en {v:.2f}%. {'Política restrictiva — presión sobre acciones y deuda.' if v > 4.5 else 'Nivel moderado — entorno aceptable para activos de riesgo.' if v > 2.5 else 'Política expansiva — favorable para acciones.'}"
            + (f" Sin cambios vs período anterior." if abs(chg) < 0.01 else f" {'Subió' if chg > 0 else 'Bajó'} {abs(chg):.2f}pp — {'hawkish' if chg > 0 else 'dovish'}.")
        ),
    },
    "CPIAUCSL": {
        "desc": "Índice de precios al consumidor. Mide el nivel general de precios; su variación interanual es la inflación.",
        "analyze": lambda v, chg: (
            f"Índice en {v:.1f}."
            + (f" Subió {chg:.2f} pts vs lectura anterior — presión inflacionaria." if chg > 1.5
               else f" Cambio moderado ({chg:+.2f} pts) — inflación contenida." if chg > 0
               else f" Bajó — señal deflacionaria, raro.")
        ),
    },
    "UNRATE": {
        "desc": "Tasa de desempleo en EE.UU. Indicador clave del mercado laboral y la salud económica.",
        "analyze": lambda v, chg: (
            f"Desempleo en {v:.1f}%."
            + (" Mercado laboral muy ajustado — presión salarial." if v < 3.5
               else " Mercado laboral sólido." if v < 4.5
               else " Desempleo elevado — economía debilitándose." if v < 6.0
               else " Nivel de recesión.")
            + (f" {'Subió' if chg > 0 else 'Bajó'} {abs(chg):.1f}pp — {'negativo' if chg > 0 else 'positivo'} para el consumo." if abs(chg) >= 0.1 else " Sin cambio significativo.")
        ),
    },
    "DGS10": {
        "desc": "Rendimiento del bono del Tesoro a 10 años. Referencia global para hipotecas, crédito corporativo y valuación de acciones.",
        "analyze": lambda v, chg: (
            f"Yield 10Y en {v:.2f}%."
            + (" Rendimientos altos — compite con acciones por capital, presiona valuaciones." if v > 4.5
               else " Nivel moderado-alto — los bonos ofrecen alternativa razonable vs acciones." if v > 3.5
               else " Rendimientos bajos — favorece acciones y activos de riesgo.")
            + (f" {'Subió' if chg > 0 else 'Bajó'} {abs(chg):.2f}pp." if abs(chg) >= 0.01 else "")
        ),
    },
    "DGS2": {
        "desc": "Rendimiento del bono del Tesoro a 2 años. Refleja las expectativas del mercado sobre la política de tasas de la Fed a corto plazo.",
        "analyze": lambda v, chg: (
            f"Yield 2Y en {v:.2f}%."
            + (" El mercado espera tasas altas por más tiempo." if v > 4.5
               else " Expectativa de tasas moderadas." if v > 3.0
               else " El mercado anticipa recortes de tasas.")
            + (f" {'Subió' if chg > 0 else 'Bajó'} {abs(chg):.2f}pp." if abs(chg) >= 0.01 else "")
        ),
    },
    "T10Y2Y": {
        "desc": "Diferencia entre el Treasury 10Y y 2Y. Indicador adelantado de recesión cuando es negativo (curva invertida).",
        "analyze": lambda v, chg: (
            f"Spread en {v:.2f}pp."
            + (" **Curva invertida** — históricamente precede recesiones en 12-18 meses." if v < 0
               else " Curva plana — incertidumbre sobre el crecimiento." if v < 0.25
               else " Curva con pendiente normal — sin señal de recesión inminente.")
        ),
    },
}

# Índices para el panorama global
MARKET_INDICES = {
    "S&P 500":   "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "VIX":       "^VIX",
    "Bitcoin":   "BTC-USD",
    "Gold":      "GC=F",
    "Oil (WTI)": "CL=F",
    "USD Index": "DX-Y.NYB",
}


# ─── Análisis técnico ─────────────────────────────────────────────────────────

def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _determine_signal(rsi: float, price: float,
                      sma_s: float, sma_l: float,
                      sma_s_prev: float, sma_l_prev: float) -> list[str]:
    signals = []

    # RSI
    if rsi < RSI_OVERSOLD:
        signals.append(f"🟢 RSI Sobrevendido ({rsi:.1f})")
    elif rsi > RSI_OVERBOUGHT:
        signals.append(f"🔴 RSI Sobrecomprado ({rsi:.1f})")

    # Cruces de medias móviles
    if sma_s_prev <= sma_l_prev and sma_s > sma_l:
        signals.append(f"🟢 Golden Cross (SMA{SMA_SHORT} cruzó arriba SMA{SMA_LONG})")
    elif sma_s_prev >= sma_l_prev and sma_s < sma_l:
        signals.append(f"🔴 Death Cross (SMA{SMA_SHORT} cruzó abajo SMA{SMA_LONG})")

    # Tendencia general
    if price > sma_s > sma_l:
        signals.append(f"📈 Tendencia alcista (precio > SMA{SMA_SHORT} > SMA{SMA_LONG})")
    elif price < sma_s < sma_l:
        signals.append(f"📉 Tendencia bajista (precio < SMA{SMA_SHORT} < SMA{SMA_LONG})")

    return signals if signals else ["⚪ Sin señal clara"]


def _volume_profile(df: pd.DataFrame, num_bins: int = 50,
                    lookback_days: int = 120) -> list[dict]:
    """Calcula Volume Profile y retorna zonas de alto volumen (HVN).

    Divide el rango de precios en bins, suma el volumen en cada bin,
    y retorna los bins con volumen superior al percentil 75 como
    zonas de soporte/resistencia por volumen.
    """
    recent = df.tail(lookback_days).copy()
    if recent.empty or len(recent) < 20:
        return []

    close  = recent["Close"].squeeze()
    volume = recent["Volume"].squeeze()

    price_min = float(close.min())
    price_max = float(close.max())
    if price_max == price_min:
        return []

    bin_edges = np.linspace(price_min, price_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vol_by_bin = np.zeros(num_bins)

    for i in range(len(close)):
        p = float(close.iloc[i])
        v = float(volume.iloc[i])
        idx = int((p - price_min) / (price_max - price_min) * (num_bins - 1))
        idx = max(0, min(idx, num_bins - 1))
        vol_by_bin[idx] += v

    # Identificar High Volume Nodes (HVN): bins sobre el percentil 75
    threshold = np.percentile(vol_by_bin, 75)
    total_vol = vol_by_bin.sum()

    zones = []
    for i in range(num_bins):
        if vol_by_bin[i] >= threshold:
            zones.append({
                "price":    float(bin_centers[i]),
                "price_lo": float(bin_edges[i]),
                "price_hi": float(bin_edges[i + 1]),
                "volume":   float(vol_by_bin[i]),
                "vol_pct":  float(vol_by_bin[i] / total_vol * 100) if total_vol > 0 else 0,
            })

    # Agrupar zonas consecutivas para formar niveles consolidados
    merged: list[dict] = []
    for z in zones:
        if merged and z["price_lo"] <= merged[-1]["price_hi"] * 1.001:
            # Extender zona anterior
            merged[-1]["price_hi"] = z["price_hi"]
            merged[-1]["volume"]  += z["volume"]
            merged[-1]["vol_pct"] += z["vol_pct"]
            # Recalcular precio central ponderado por volumen
            total_v = merged[-1]["volume"]
            merged[-1]["price"] = (
                (merged[-1]["price"] * (total_v - z["volume"]) + z["price"] * z["volume"])
                / total_v
            ) if total_v > 0 else z["price"]
        else:
            merged.append(z.copy())

    return sorted(merged, key=lambda x: x["volume"], reverse=True)


def _classify_volume_zones(zones: list[dict], current_price: float) -> dict:
    """Clasifica zonas de volumen en soportes y resistencias relativas al precio actual."""
    supports    = []
    resistances = []

    for z in zones:
        dist_pct = (z["price"] - current_price) / current_price * 100
        entry = {
            "price":    z["price"],
            "range":    f"${z['price_lo']:,.2f} – ${z['price_hi']:,.2f}",
            "vol_pct":  z["vol_pct"],
            "dist_pct": dist_pct,
        }
        # Si el precio actual está dentro de la zona
        if z["price_lo"] <= current_price <= z["price_hi"]:
            entry["zone_type"] = "current"
            supports.append(entry)
            resistances.append(entry)
        elif z["price"] < current_price:
            entry["zone_type"] = "support"
            supports.append(entry)
        else:
            entry["zone_type"] = "resistance"
            resistances.append(entry)

    # Ordenar: soportes de más cercano a más lejano (desc por precio)
    supports    = sorted(supports,    key=lambda x: x["price"], reverse=True)[:3]
    resistances = sorted(resistances, key=lambda x: x["price"])[:3]

    return {"supports": supports, "resistances": resistances}


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
            for dt_idx in ed.index:
                dt = dt_idx.to_pydatetime()
                if dt.tzinfo:
                    dt = dt.replace(tzinfo=None)
                days_until = (dt.date() - today.date()).days

                if 0 <= days_until <= 7:
                    if days_until <= 1:
                        pts, msg = 30, f"📅 Earnings MAÑANA ({dt.strftime('%Y-%m-%d')}) — máxima volatilidad esperada"
                    elif days_until <= 3:
                        pts, msg = 25, f"📅 Earnings en {days_until} días ({dt.strftime('%Y-%m-%d')}) — volatilidad alta"
                    else:
                        pts, msg = 15, f"📅 Earnings en {days_until} días ({dt.strftime('%Y-%m-%d')}) — precaución"
                    alerts.append({"type": "event", "signal": "earnings_soon", "message": msg, "points": pts})
                    break
                elif -2 <= days_until < 0:
                    alerts.append({
                        "type": "event", "signal": "earnings_recent",
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
                    alerts.append({"type": "event", "signal": "earnings_miss_history",
                        "message": f"📅 Historial: falló estimados {misses}/4 trimestres — patrón de misses", "points": 15})
                elif misses >= 2:
                    alerts.append({"type": "event", "signal": "earnings_miss_history",
                        "message": f"📅 Historial: falló estimados {misses}/4 trimestres", "points": 10})
                elif misses == 0:
                    alerts.append({"type": "event", "signal": "earnings_beat_history",
                        "message": f"📅 Historial: superó estimados 4/4 trimestres — track record sólido", "points": -10})
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
                    alerts.append({"type": "event", "signal": "ex_dividend",
                        "message": f"📅 Ex-dividend en {days_until} día(s) ({ex_date}) — ajuste de precio esperado", "points": 5})
    except Exception:
        pass

    return alerts


def _detect_technical_risks(tech_data: dict) -> list[dict]:
    """Detecta patrones técnicos de riesgo en los datos de un ticker."""
    alerts = []
    if not tech_data or "error" in tech_data:
        return alerts

    price = tech_data["price"]
    rsi   = tech_data["rsi"]
    hist  = tech_data.get("hist")

    # 1. RSI extremo (>80)
    if rsi > 80:
        alerts.append({"type": "technical", "signal": "rsi_extreme_high",
            "message": f"📉 RSI extremo ({rsi:.1f}) — sobrecompra severa, alto riesgo de corrección", "points": 10})
    elif rsi < 20:
        alerts.append({"type": "technical", "signal": "rsi_extreme_low",
            "message": f"📉 RSI extremo bajo ({rsi:.1f}) — sobreventa severa, puede continuar cayendo", "points": 5})

    # 2. Pérdida de SMA 200
    sma200 = tech_data.get("sma200")
    if sma200 is not None and price < sma200:
        pct_below = (price / sma200 - 1) * 100
        alerts.append({"type": "technical", "signal": "below_sma200",
            "message": f"📉 Precio {pct_below:.1f}% debajo de SMA 200 (${sma200:,.2f}) — tendencia largo plazo dañada", "points": 15})

    # 3. Death Cross activo (SMA20 < SMA50)
    sma_s = tech_data["sma_s"]
    sma_l = tech_data["sma_l"]
    if sma_s < sma_l:
        alerts.append({"type": "technical", "signal": "death_cross_active",
            "message": f"📉 Death Cross activo — SMA20 (${sma_s:,.2f}) por debajo de SMA50 (${sma_l:,.2f})", "points": 10})

    # 4. MACD cruce bajista reciente
    macd_h = tech_data.get("macd_hist", 0)
    macd_h_prev = tech_data.get("macd_hist_prev", 0)
    if macd_h < 0 and macd_h_prev > 0:
        alerts.append({"type": "technical", "signal": "macd_bearish_cross",
            "message": "📉 MACD cruzó a bajista — histograma pasó de positivo a negativo", "points": 10})

    # 5. Bollinger Squeeze
    bb_upper = tech_data.get("bb_upper", 0)
    bb_lower = tech_data.get("bb_lower", 0)
    if price > 0 and bb_upper > 0 and bb_lower > 0:
        bb_width = (bb_upper - bb_lower) / price
        if bb_width < 0.05:
            alerts.append({"type": "technical", "signal": "bollinger_squeeze",
                "message": f"📉 Bollinger Squeeze detectado (ancho {bb_width:.1%}) — explosión de volatilidad inminente", "points": 10})

    # 6. Divergencia bajista RSI
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
                alerts.append({"type": "technical", "signal": "rsi_bearish_divergence",
                    "message": "📉 Divergencia bajista RSI — precio hizo nuevo máximo pero RSI no confirma", "points": 20})
        except Exception:
            pass

    # 7. Volumen decreciente en rally
    if hist is not None and len(hist) >= 20:
        try:
            close_s = hist["Close"].squeeze()
            vol_s = hist["Volume"].squeeze()
            price_chg_20d = float((close_s.iloc[-1] / close_s.iloc[-20] - 1))
            vol_avg_10d = float(vol_s.tail(10).mean())
            vol_avg_20d = float(vol_s.tail(20).mean())
            if price_chg_20d > 0.10 and vol_avg_10d < vol_avg_20d * 0.8:
                alerts.append({"type": "technical", "signal": "volume_divergence",
                    "message": f"📉 Volumen decreciente en rally — precio subió {price_chg_20d:.0%} pero volumen cae", "points": 10})
        except Exception:
            pass

    # 8. Precio cerca de resistencia fuerte
    resistances = tech_data.get("vol_resistances", [])
    for r in resistances:
        if r.get("zone_type") != "current" and 0 < r["dist_pct"] < 2:
            alerts.append({"type": "technical", "signal": "near_resistance",
                "message": f"📉 Resistencia fuerte a {r['dist_pct']:.1f}% ({r['range']}) — riesgo de rechazo", "points": 5})
            break

    # 9. Precio lejos de soporte
    supports = tech_data.get("vol_supports", [])
    nearest_support_dist = None
    for s in supports:
        if s.get("zone_type") != "current":
            nearest_support_dist = abs(s["dist_pct"])
            break
    if nearest_support_dist is not None and nearest_support_dist > 15:
        alerts.append({"type": "technical", "signal": "far_from_support",
            "message": f"📉 Sin soporte cercano — soporte más próximo a -{nearest_support_dist:.1f}%", "points": 5})

    return alerts


def _calculate_risk_score(alerts: list[dict]) -> tuple[int, str, list[dict]]:
    """Combina alertas en un Risk Score 0-100 con nivel semáforo."""
    raw_score = sum(a["points"] for a in alerts)
    score = max(0, min(raw_score, 100))

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


def fetch_technicals(ticker: str) -> dict | None:
    try:
        df = yf.download(ticker, period="2y", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None

        close = df["Close"].squeeze()

        price   = float(close.iloc[-1])
        chg_1d  = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
        chg_1w  = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6  else None
        chg_1m  = float((close.iloc[-1] / close.iloc[-22] - 1) * 100) if len(close) >= 22 else None

        rsi_s    = _rsi(close, RSI_PERIOD)
        sma_s_s  = close.rolling(SMA_SHORT).mean()
        sma_l_s  = close.rolling(SMA_LONG).mean()

        rsi      = float(rsi_s.iloc[-1])
        sma_s    = float(sma_s_s.iloc[-1])
        sma_l    = float(sma_l_s.iloc[-1])
        sma_s_p  = float(sma_s_s.iloc[-2])
        sma_l_p  = float(sma_l_s.iloc[-2])

        signals = _determine_signal(rsi, price, sma_s, sma_l, sma_s_p, sma_l_p)

        # EMA 12/26 para referencia
        ema12 = float(close.ewm(span=12, adjust=False).mean().iloc[-1])
        ema26 = float(close.ewm(span=26, adjust=False).mean().iloc[-1])

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
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ─── Panorama de mercado ──────────────────────────────────────────────────────

def fetch_market_overview() -> dict:
    results = {}
    for name, ticker in MARKET_INDICES.items():
        try:
            info = yf.Ticker(ticker).fast_info
            prev = info.previous_close
            last = info.last_price
            if prev and last:
                results[name] = {
                    "price":      last,
                    "change":     last - prev,
                    "change_pct": (last / prev - 1) * 100,
                }
        except Exception:
            pass
    return results


# ─── Indicadores macroeconómicos (FRED) ──────────────────────────────────────

def _fetch_fred(series_id: str, name: str) -> dict | None:
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        headers = {"User-Agent": "Mozilla/5.0 (financial-tracker/1.0)"}
        df = pd.read_csv(url, names=["date", "value"], skiprows=1,
                         parse_dates=["date"])
        df = df[df["value"] != "."].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        if len(df) < 2:
            return None
        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        return {
            "series_id": series_id,
            "name":   name,
            "value":  float(latest["value"]),
            "date":   latest["date"].strftime("%Y-%m-%d"),
            "prev":   float(prev["value"]),
            "change": float(latest["value"] - prev["value"]),
        }
    except Exception:
        return None


def fetch_macro_indicators() -> list[dict]:
    results = []
    for series_id, name in FRED_SERIES:
        data = _fetch_fred(series_id, name)
        if data:
            results.append(data)
        time.sleep(0.4)
    return results


# ─── Fear & Greed Index (CNN) ─────────────────────────────────────────────────

def fetch_fear_greed() -> dict | None:
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()["fear_and_greed"]
        return {
            "score":    round(float(data["score"])),
            "rating":   data["rating"].replace("_", " ").title(),
            "previous": round(float(data["previous_close"])),
        }
    except Exception:
        return None


# ─── Scraping de artículos y resumen con LLM ────────────────────────────────

_SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


def _scrape_article(url: str, max_chars: int = 3000) -> str:
    """Extrae el texto principal de una página web."""
    try:
        r = requests.get(url, headers=_SCRAPE_HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Eliminar elementos no relevantes
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "iframe", "noscript", "svg"]):
            tag.decompose()

        # Intentar extraer contenido del artículo con selectores comunes
        article = (
            soup.find("article")
            or soup.find("div", class_=re.compile(r"article|content|story|post|entry", re.I))
            or soup.find("main")
        )

        text = (article or soup.body or soup).get_text(separator=" ", strip=True)
        # Limpiar espacios múltiples
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars] if len(text) > 100 else ""
    except Exception:
        return ""


def _summarize_hf(text: str, max_length: int = 120, min_length: int = 30) -> str:
    """Envía texto al modelo de summarization de Hugging Face."""
    if not HF_TOKEN:
        return ""
    if not text or len(text) < 100:
        return ""

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": text[:2500],  # Límite del modelo BART
        "parameters": {
            "max_length": max_length,
            "min_length": min_length,
            "do_sample": False,
        },
    }
    try:
        r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        result = r.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("summary_text", "").strip()
        return ""
    except requests.exceptions.HTTPError as e:
        # Modelo en cold start — HF devuelve 503
        if e.response and e.response.status_code == 503:
            print(f"      ⏳ Modelo en cold start, reintentando en 20s...")
            time.sleep(20)
            try:
                r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
                r.raise_for_status()
                result = r.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("summary_text", "").strip()
            except Exception:
                pass
        return ""
    except Exception:
        return ""


def _enrich_news_with_summaries(headlines: list[dict], max_items: int = 0) -> None:
    """Enriquece las noticias con resúmenes generados por LLM (in-place).
    max_items=0 significa resumir todas las noticias."""
    if not HF_TOKEN:
        print("   ⚠️ HF_TOKEN no configurado — omitiendo resúmenes.")
        print("      Configura: set HF_TOKEN=hf_xxxxx  (Windows)")
        print("      O:         export HF_TOKEN=hf_xxxxx (Linux/Mac)")
        return

    items = headlines if max_items == 0 else headlines[:max_items]
    total = len(items)
    for i, h in enumerate(items):
        if h.get("summary"):
            continue
        print(f"      📝 Resumiendo [{i+1}/{total}]: {h['title'][:60]}...")

        # 1. Intentar scraping del artículo completo
        article_text = _scrape_article(h["link"])

        # 2. Fallback: usar descripción del RSS
        if not article_text:
            article_text = h.get("rss_desc", "")

        # 3. Resumir con LLM si hay texto suficiente
        if article_text and len(article_text) > 80:
            summary = _summarize_hf(article_text)
            h["summary"] = summary
        else:
            h["summary"] = ""
        time.sleep(1)  # Rate limiting


# ─── Noticias (RSS) ───────────────────────────────────────────────────────────

def fetch_news(max_per_feed: int = 8) -> list[dict]:
    headlines = []
    for source, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "").strip()
                if title:
                    # Muchos RSS traen un resumen en 'summary' o 'description'
                    rss_desc = entry.get("summary", entry.get("description", ""))
                    # Limpiar HTML del RSS description
                    if rss_desc:
                        rss_desc = BeautifulSoup(rss_desc, "html.parser").get_text(strip=True)
                    headlines.append({
                        "source":    source,
                        "title":     title,
                        "link":      entry.get("link", "#"),
                        "published": entry.get("published", ""),
                        "rss_desc":  rss_desc,  # Descripción del RSS como fallback
                        "summary":   "",         # Se llena con el LLM
                    })
        except Exception:
            continue
    return headlines


def _filter_news(headlines: list[dict], ticker: str) -> list[dict]:
    keywords = TICKER_KEYWORDS.get(ticker, [ticker.lower()])
    # Usar word boundaries (\b) para evitar falsos positivos
    # Ej: "mac" no debe matchear "pharmaceutical"
    patterns = [re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in keywords]
    return [h for h in headlines
            if any(p.search(h["title"]) for p in patterns)]


# ─── Scanner de señales (ideas de inversión) ─────────────────────────────────

def scan_signals(universe: list[str]) -> tuple[list[dict], list[dict]]:
    buy_signals  = []
    sell_signals = []
    for ticker in universe:
        data = fetch_technicals(ticker)
        if data and "error" not in data:
            buys  = [s for s in data["signals"] if "🟢" in s]
            sells = [s for s in data["signals"] if "🔴" in s]
            if buys:
                buy_signals.append({**data, "matched_signals": buys})
            if sells:
                sell_signals.append({**data, "matched_signals": sells})
        time.sleep(0.3)
    return buy_signals, sell_signals


# ─── Helpers de formato ───────────────────────────────────────────────────────

def _fmt_change(val: float | None, suffix: str = "%") -> str:
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}{suffix}"


def _arrow(val: float | None) -> str:
    if val is None:
        return ""
    return "▲" if val >= 0 else "▼"


def _fg_emoji(score: int) -> str:
    if score < 25:  return "😱"
    if score < 45:  return "😰"
    if score < 55:  return "😐"
    if score < 75:  return "😊"
    return "🤑"


# ─── Fibonacci Screener ──────────────────────────────────────────────────────


def _local_extrema(series: pd.Series, order: int = FIB_EXTREMA_ORDER):
    highs, lows = [], []
    for i in range(order, len(series) - order):
        val = float(series.iloc[i])
        if all(val >= float(series.iloc[i - j]) for j in range(1, order + 1)) and \
           all(val >= float(series.iloc[i + j]) for j in range(1, order + 1)):
            highs.append(i)
        if all(val <= float(series.iloc[i - j]) for j in range(1, order + 1)) and \
           all(val <= float(series.iloc[i + j]) for j in range(1, order + 1)):
            lows.append(i)
    return highs, lows


def _find_significant_swing(close: pd.Series, min_amplitude: float = FIB_MIN_SWING) -> dict | None:
    if len(close) < 30:
        return None
    high_idxs, low_idxs = _local_extrema(close)
    if not high_idxs or not low_idxs:
        return None

    points = []
    for i in high_idxs:
        points.append(("high", i, float(close.iloc[i]), close.index[i]))
    for i in low_idxs:
        points.append(("low", i, float(close.iloc[i]), close.index[i]))
    points.sort(key=lambda x: x[1], reverse=True)

    for idx_a in range(len(points)):
        for idx_b in range(idx_a + 1, len(points)):
            pa, pb = points[idx_a], points[idx_b]
            if pa[0] == pb[0]:
                continue
            if pa[0] == "high":
                swing_high, swing_low = pa[2], pb[2]
                high_date, low_date = pa[3], pb[3]
            else:
                swing_high, swing_low = pb[2], pa[2]
                high_date, low_date = pb[3], pa[3]

            if swing_low <= 0:
                continue
            amplitude = (swing_high - swing_low) / swing_low
            if amplitude < min_amplitude:
                continue

            if high_date > low_date:
                direction = "bullish"
            else:
                direction = "bearish"

            return {
                "swing_high": swing_high,
                "swing_low": swing_low,
                "high_date": high_date,
                "low_date": low_date,
                "direction": direction,
                "amplitude": amplitude,
            }
    return None


def _fibonacci_level(swing_high: float, swing_low: float,
                     level: float = FIB_LEVEL, direction: str = "bullish") -> float:
    if direction == "bullish":
        return swing_high - (swing_high - swing_low) * level
    return swing_low + (swing_high - swing_low) * level


def _fibonacci_screen(tickers: list[str]) -> list[dict]:
    results = []
    skipped = 0
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="6mo", interval="1d",
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < 30:
                skipped += 1
                continue
            close = df["Close"].squeeze()
            price = float(close.iloc[-1])

            swing = _find_significant_swing(close)
            if swing is None:
                skipped += 1
                continue

            fib = _fibonacci_level(swing["swing_high"], swing["swing_low"],
                                   FIB_LEVEL, swing["direction"])
            dist = (price - fib) / fib * 100
            if abs(dist) > FIB_PROXIMITY * 100:
                continue

            rsi_s = _rsi(close, RSI_PERIOD)
            rsi_val = float(rsi_s.iloc[-1]) if not rsi_s.empty else 0.0

            results.append({
                "ticker": ticker,
                "price": price,
                "swing_high": swing["swing_high"],
                "swing_low": swing["swing_low"],
                "high_date": swing["high_date"],
                "low_date": swing["low_date"],
                "direction": swing["direction"],
                "amplitude": swing["amplitude"],
                "fib_618": fib,
                "distance_pct": dist,
                "rsi": rsi_val,
            })
        except Exception:
            skipped += 1
            continue

    results.sort(key=lambda x: abs(x["distance_pct"]))
    return results


def generate_fibonacci_report() -> str:
    today = datetime.now()
    filename = f"{today.strftime('%Y_%m_%d')}_Fibonacci.md"

    print("\n📐 Fibonacci Screener — Golden Ratio (0.618)")
    print("=" * 55)

    all_tickers = list(dict.fromkeys(MY_PORTFOLIO + WATCHLIST))
    total = len(all_tickers)
    print(f"⏳ Analizando {total} tickers...")

    results = _fibonacci_screen(all_tickers)

    buys = [r for r in results if r["direction"] == "bullish"]
    resistances = [r for r in results if r["direction"] == "bearish"]
    no_match = total - len(results)

    L: list[str] = []
    L += [
        "# 📐 Fibonacci Screener — Golden Ratio (0.618)",
        f"> Generado el **{today.strftime('%Y-%m-%d %H:%M')}**  ",
        f"> Tickers analizados: {total} | Oportunidades encontradas: {len(results)}",
        "",
    ]

    # Oportunidades de compra
    L += [
        "## 🟢 Oportunidades de Compra (retroceso en swing alcista)",
        "",
        "Tickers cuyo precio retrocedió ~61.8% de un movimiento alcista significativo (>15%).",
        "Zona ideal para entrada en largo.",
        "",
    ]
    if buys:
        L += [
            "| Ticker | Precio | Swing (Low→High) | Amplitud | Nivel 0.618 | Distancia | RSI |",
            "|--------|--------|-------------------|----------|-------------|-----------|-----|",
        ]
        for r in buys:
            ld = r["low_date"].strftime("%m/%d") if hasattr(r["low_date"], "strftime") else str(r["low_date"])[:5]
            hd = r["high_date"].strftime("%m/%d") if hasattr(r["high_date"], "strftime") else str(r["high_date"])[:5]
            L.append(
                f"| {r['ticker']} | ${r['price']:,.2f} "
                f"| ${r['swing_low']:,.2f}→${r['swing_high']:,.2f} ({ld}→{hd}) "
                f"| +{r['amplitude']*100:.1f}% "
                f"| ${r['fib_618']:,.2f} "
                f"| {r['distance_pct']:+.1f}% "
                f"| {r['rsi']:.0f} |"
            )
        L.append("")
    else:
        L += ["> Sin oportunidades de compra detectadas.", ""]

    # Zonas de resistencia
    L += [
        "## ⚠️ Zonas de Resistencia (retroceso en swing bajista)",
        "",
        "Tickers cuyo precio rebotó ~61.8% de un movimiento bajista significativo (>15%).",
        "Zona de posible rechazo — precaución al comprar.",
        "",
    ]
    if resistances:
        L += [
            "| Ticker | Precio | Swing (High→Low) | Amplitud | Nivel 0.618 | Distancia | RSI |",
            "|--------|--------|-------------------|----------|-------------|-----------|-----|",
        ]
        for r in resistances:
            hd = r["high_date"].strftime("%m/%d") if hasattr(r["high_date"], "strftime") else str(r["high_date"])[:5]
            ld = r["low_date"].strftime("%m/%d") if hasattr(r["low_date"], "strftime") else str(r["low_date"])[:5]
            L.append(
                f"| {r['ticker']} | ${r['price']:,.2f} "
                f"| ${r['swing_high']:,.2f}→${r['swing_low']:,.2f} ({hd}→{ld}) "
                f"| -{r['amplitude']*100:.1f}% "
                f"| ${r['fib_618']:,.2f} "
                f"| {r['distance_pct']:+.1f}% "
                f"| {r['rsi']:.0f} |"
            )
        L.append("")
    else:
        L += ["> Sin zonas de resistencia detectadas.", ""]

    # Resumen
    L += [
        "## 📊 Resumen",
        "",
        f"- Total analizados: {total}",
        f"- Sin swing >15% o sin datos: {no_match}",
        f"- **Oportunidades de compra: {len(buys)}**",
        f"- **Zonas de resistencia: {len(resistances)}**",
        "",
        "---",
        "",
        f"*Fibonacci Screener generado automáticamente — {today.strftime('%Y-%m-%d %H:%M:%S')}*  ",
        "*Fuentes: Yahoo Finance (yfinance)*  ",
        "*⚠️ Solo informativo. No constituye asesoramiento financiero.*",
    ]

    report = "\n".join(L)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n🟢 Compra: {len(buys)} oportunidades")
    print(f"⚠️  Resistencia: {len(resistances)} zonas")
    print(f"✅ Reporte guardado: {filename}")

    return filename


# ─── Generador de reporte Markdown ───────────────────────────────────────────

def generate_report() -> str:
    today    = datetime.now()
    filename = today.strftime("%Y_%m_%d") + "_Investments.md"

    print(f"\n📊 Financial Tracker — {today.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    # ── Fetch data ──────────────────────────────────────────
    print("⏳ Panorama de mercado...")
    market = fetch_market_overview()

    print("⏳ Indicadores macro (FRED)...")
    macro = fetch_macro_indicators()

    print("⏳ Fear & Greed Index (CNN)...")
    fg = fetch_fear_greed()

    print("⏳ Noticias (RSS)...")
    all_news = fetch_news()

    print("⏳ Generando resúmenes de noticias (Hugging Face)...")
    _enrich_news_with_summaries(all_news)

    print("⏳ Analizando portafolio...")
    portfolio_data = []
    risk_scores = {}
    for t in MY_PORTFOLIO:
        print(f"   → {t}")
        tech = fetch_technicals(t)
        portfolio_data.append(tech)
        events = _detect_upcoming_events(t)
        tech_risks = _detect_technical_risks(tech) if tech and "error" not in tech else []
        score, level, alerts = _calculate_risk_score(events + tech_risks)
        risk_scores[t] = (score, level, alerts)
        time.sleep(0.4)

    # Analizar tickers del CSV (con detalle completo)
    csv_tickers = WATCHLIST[len(_WATCHLIST_BASE):]  # Solo los que vienen del CSV
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

    print("⏳ Escaneando watchlist para señales...")
    buys, sells = scan_signals(WATCHLIST)

    # ── Construir Markdown ──────────────────────────────────
    L: list[str] = []

    # Encabezado
    L += [
        f"# 📈 Financial Tracker — {today.strftime('%d %B %Y')}",
        f"> Reporte generado el **{today.strftime('%Y-%m-%d %H:%M')}**  ",
        f"> Fuentes: Yahoo Finance · FRED · CNN Fear & Greed · RSS Feeds",
        "",
        "---",
        "",
    ]

    # ── Panorama global ──────────────────────────────────────
    L += ["## 🌐 Panorama del Mercado", ""]
    L += ["| Índice | Precio | Cambio | % |",
          "|---|---:|---:|---:|"]
    for name, d in market.items():
        if d:
            color = "🟢" if d["change_pct"] >= 0 else "🔴"
            L.append(
                f"| {color} {name} | {d['price']:,.2f} "
                f"| {_fmt_change(d['change'], '')} "
                f"| {_arrow(d['change_pct'])} {abs(d['change_pct']):.2f}% |"
            )
    L.append("")

    # Fear & Greed
    if fg:
        emoji = _fg_emoji(fg["score"])
        direction = "▲" if fg["score"] >= fg["previous"] else "▼"
        L += [
            f"### {emoji} CNN Fear & Greed Index",
            f"| Score | Rating | Anterior | Tendencia |",
            f"|---:|---|---:|---:|",
            f"| **{fg['score']}** | {fg['rating']} | {fg['previous']} | {direction} |",
            "",
        ]

    # Macro FRED
    if macro:
        L += ["## 📉 Indicadores Macroeconómicos (FRED)", ""]
        L += ["| Indicador | Valor actual | Dato anterior | Cambio | Fecha | Análisis indicador |",
              "|---|---:|---:|---:|---|---|"]
        for m in macro:
            chg_str = _fmt_change(m["change"], "")
            color = "🟢" if m["change"] <= 0 else "🔴"  # baja en tasas/inflación = bueno
            # Generar análisis contextual
            sid = m.get("series_id", "")
            meta = FRED_DESCRIPTIONS.get(sid, {})
            desc = meta.get("desc", "")
            analyze_fn = meta.get("analyze")
            analysis = analyze_fn(m["value"], m["change"]) if analyze_fn else ""
            analysis_cell = f"**{desc}** {analysis}" if desc else ""
            L.append(
                f"| {m['name']} | **{m['value']:.2f}** | {m['prev']:.2f} "
                f"| {color} {chg_str} | {m['date']} | {analysis_cell} |"
            )
        L.append("")

    # ── Alertas Predictivas ──────────────────────────────────
    scored_tickers = sorted(risk_scores.items(), key=lambda x: x[1][0], reverse=True)
    has_alerts = [(t, s, l, a) for t, (s, l, a) in scored_tickers if s > 25]

    if has_alerts:
        L += [
            "## ⚠️ Alertas Predictivas",
            "",
            "| Ticker | Risk Score | Nivel | Alertas principales |",
            "|---|---:|---|---|",
        ]
        for ticker_a, score_a, level_a, alerts_a in has_alerts:
            top_alerts = " · ".join(a["message"].split(" — ")[0] for a in alerts_a[:3] if a["points"] > 0)
            L.append(f"| **{ticker_a}** | {score_a}/100 | {level_a} | {top_alerts} |")
        L.append("")

        high_alerts = [(t, s, l, a) for t, s, l, a in has_alerts if s > 50]
        for ticker_a, score_a, level_a, alerts_a in high_alerts:
            action = _risk_action(score_a)
            L += [
                f"### {level_a.split()[0]} {ticker_a} — {score_a}/100",
                "",
                "| Tipo | Alerta | Pts |",
                "|---|---|---:|",
            ]
            for a in alerts_a:
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
        "",
        "## 💼 Sección 1 — Mis Inversiones",
        "",
        f"Portafolio: `{'`, `'.join(MY_PORTFOLIO)}`",
        "",
    ]

    for data in portfolio_data:
        if data is None:
            continue
        ticker = data.get("ticker", "?")

        if "error" in data:
            L += [f"### {ticker}",
                  f"> ⚠️ Error al obtener datos: `{data['error']}`", ""]
            continue

        price   = data["price"]
        chg_1d  = data["chg_1d"]
        badge   = "🟢" if chg_1d >= 0 else "🔴"

        L += [
            f"### {badge} {ticker}",
            "",
            f"| Precio | 1D | 1W | 1M |",
            f"|---:|---:|---:|---:|",
            f"| **${price:,.2f}** | {_fmt_change(chg_1d)} | {_fmt_change(data['chg_1w'])} | {_fmt_change(data['chg_1m'])} |",
            "",
            f"| RSI ({RSI_PERIOD}) | SMA {SMA_SHORT} | SMA {SMA_LONG} | EMA 12 | EMA 26 | vs SMA{SMA_SHORT} | vs SMA{SMA_LONG} |",
            f"|---:|---:|---:|---:|---:|---:|---:|",
            f"| {data['rsi']:.1f} | ${data['sma_s']:,.2f} | ${data['sma_l']:,.2f} "
            f"| ${data['ema12']:,.2f} | ${data['ema26']:,.2f} "
            f"| {_fmt_change((price/data['sma_s']-1)*100)} "
            f"| {_fmt_change((price/data['sma_l']-1)*100)} |",
            "",
            "**Señales técnicas:**",
        ]
        for sig in data["signals"]:
            L.append(f"- {sig}")
        L.append("")

        # Risk Score en la card
        ticker = data.get("ticker", "?")
        if ticker in risk_scores:
            rs_score, rs_level, rs_alerts = risk_scores[ticker]
            if rs_score > 25:
                top_reasons = ", ".join(a["message"].split(" — ")[0] for a in rs_alerts[:2] if a["points"] > 0)
                L.append(f"**Risk Score:** {rs_score}/100 {rs_level} — {top_reasons}")
                L.append("")

        # Zonas de soporte/resistencia por volumen
        resistances = data.get("vol_resistances", [])
        supports    = data.get("vol_supports", [])
        if resistances or supports:
            L += [
                "**Zonas de alto volumen (Volume Profile 120D):**",
                "",
                "| Tipo | Zona de precio | Distancia | Vol % | Acción sugerida |",
                "|---|---|---:|---:|---|",
            ]
            for r in resistances:
                if r.get("zone_type") == "current":
                    label = "⚡ Zona actual"
                    action = "Precio en zona de alto volumen — posible congestión"
                else:
                    label = "🔴 Resistencia"
                    if abs(r["dist_pct"]) < 3:
                        action = "⚠️ Resistencia cercana — considerar take profit"
                    elif abs(r["dist_pct"]) < 7:
                        action = "Target de salida potencial"
                    else:
                        action = "Resistencia lejana"
                L.append(
                    f"| {label} | {r['range']} | {_fmt_change(r['dist_pct'])} "
                    f"| {r['vol_pct']:.1f}% | {action} |"
                )
            for s in supports:
                if s.get("zone_type") == "current":
                    continue  # Ya se mostró arriba
                label = "🟢 Soporte"
                if abs(s["dist_pct"]) < 3:
                    action = "⚠️ Soporte cercano — zona de entrada / stop loss"
                elif abs(s["dist_pct"]) < 7:
                    action = "Zona de entrada potencial en pullback"
                else:
                    action = "Soporte profundo"
                L.append(
                    f"| {label} | {s['range']} | {_fmt_change(s['dist_pct'])} "
                    f"| {s['vol_pct']:.1f}% | {action} |"
                )
            L.append("")

        # Noticias relacionadas
        news = _filter_news(all_news, ticker)
        if news:
            L.append("**Noticias relevantes:**")
            for n in news[:3]:
                summary = n.get("summary", "")
                L.append(f"- [{n['title']}]({n['link']}) — *{n['source']}*")
                if summary:
                    L.append(f"  > {summary}")
            L.append("")

        L += ["---", ""]

    # ── Sección 1B: Watchlist CSV (detalle completo) ─────────
    if csv_data and any(d for d in csv_data if d):
        L += [
            "## 📋 Watchlist (watchlist.csv)",
            "",
            f"> {len([d for d in csv_data if d and 'error' not in d])} tickers analizados desde `watchlist.csv`",
            "",
        ]
        for data in csv_data:
            if data is None:
                continue
            ticker = data.get("ticker", "?")
            if "error" in data:
                continue  # Saltar tickers sin datos

            price  = data["price"]
            chg_1d = data["chg_1d"]
            badge  = "🟢" if chg_1d >= 0 else "🔴"

            L += [
                f"### {badge} {ticker}",
                "",
                f"| Precio | 1D | 1W | 1M |",
                f"|---:|---:|---:|---:|",
                f"| **${price:,.2f}** | {_fmt_change(chg_1d)} | {_fmt_change(data['chg_1w'])} | {_fmt_change(data['chg_1m'])} |",
                "",
                f"| RSI ({RSI_PERIOD}) | SMA {SMA_SHORT} | SMA {SMA_LONG} | EMA 12 | EMA 26 |",
                f"|---:|---:|---:|---:|---:|",
                f"| {data['rsi']:.1f} | ${data['sma_s']:,.2f} | ${data['sma_l']:,.2f} "
                f"| ${data['ema12']:,.2f} | ${data['ema26']:,.2f} |",
                "",
                "**Señales:**",
            ]
            for sig in data["signals"]:
                L.append(f"- {sig}")
            L.append("")

            # Zonas de volumen
            resistances = data.get("vol_resistances", [])
            supports    = data.get("vol_supports", [])
            if resistances or supports:
                L += [
                    "**Zonas de alto volumen (Volume Profile 120D):**",
                    "",
                    "| Tipo | Zona de precio | Distancia | Vol % | Acción sugerida |",
                    "|---|---|---:|---:|---|",
                ]
                for r in resistances:
                    if r.get("zone_type") == "current":
                        label = "⚡ Zona actual"
                        action = "Precio en zona de alto volumen — posible congestión"
                    else:
                        label = "🔴 Resistencia"
                        action = ("⚠️ Resistencia cercana — considerar take profit" if abs(r["dist_pct"]) < 3
                                  else "Target de salida potencial" if abs(r["dist_pct"]) < 7
                                  else "Resistencia lejana")
                    L.append(
                        f"| {label} | {r['range']} | {_fmt_change(r['dist_pct'])} "
                        f"| {r['vol_pct']:.1f}% | {action} |"
                    )
                for s in supports:
                    if s.get("zone_type") == "current":
                        continue
                    label = "🟢 Soporte"
                    action = ("⚠️ Soporte cercano — zona de entrada / stop loss" if abs(s["dist_pct"]) < 3
                              else "Zona de entrada potencial en pullback" if abs(s["dist_pct"]) < 7
                              else "Soporte profundo")
                    L.append(
                        f"| {label} | {s['range']} | {_fmt_change(s['dist_pct'])} "
                        f"| {s['vol_pct']:.1f}% | {action} |"
                    )
                L.append("")

            L += ["---", ""]

    # ── Sección 2: Ideas de Inversión ────────────────────────
    L += [
        "## 💡 Sección 2 — Ideas de Inversión",
        "",
        f"> Watchlist escaneado: {len(WATCHLIST)} activos  ",
        f"> Criterios: RSI < {RSI_OVERSOLD} (sobrevendido) · RSI > {RSI_OVERBOUGHT} (sobrecomprado) · Golden/Death Cross · Tendencia",
        "",
    ]

    # Señales de compra
    L += ["### 🟢 Señales de Compra", ""]
    if buys:
        L += [f"| Ticker | Precio | 1D | RSI | SMA{SMA_SHORT} | SMA{SMA_LONG} | Señal |",
              f"|---|---:|---:|---:|---:|---:|---|"]
        for i in sorted(buys, key=lambda x: x["rsi"]):
            sigs = " · ".join(i["matched_signals"])
            L.append(
                f"| **{i['ticker']}** | ${i['price']:,.2f} | {_fmt_change(i['chg_1d'])} "
                f"| {i['rsi']:.1f} | ${i['sma_s']:,.2f} | ${i['sma_l']:,.2f} | {sigs} |"
            )
        L.append("")
    else:
        L += ["> Sin señales de compra en el watchlist actual.", ""]

    # Señales de venta / precaución
    L += ["### 🔴 Señales de Venta / Precaución", ""]
    if sells:
        L += [f"| Ticker | Precio | 1D | RSI | SMA{SMA_SHORT} | SMA{SMA_LONG} | Señal |",
              f"|---|---:|---:|---:|---:|---:|---|"]
        for i in sorted(sells, key=lambda x: x["rsi"], reverse=True):
            sigs = " · ".join(i["matched_signals"])
            L.append(
                f"| **{i['ticker']}** | ${i['price']:,.2f} | {_fmt_change(i['chg_1d'])} "
                f"| {i['rsi']:.1f} | ${i['sma_s']:,.2f} | ${i['sma_l']:,.2f} | {sigs} |"
            )
        L.append("")
    else:
        L += ["> Sin señales de venta/precaución en el watchlist actual.", ""]

    # ── Noticias generales ───────────────────────────────────
    L += ["---", "", "## 📰 Noticias del Mercado", ""]
    by_source: dict[str, list] = {}
    for h in all_news:
        by_source.setdefault(h["source"], []).append(h)

    for source, items in by_source.items():
        L += [f"**{source}**", ""]
        for item in items[:6]:
            summary = item.get("summary", "")
            L.append(f"- [{item['title']}]({item['link']})")
            if summary:
                L.append(f"  > {summary}")
        L.append("")

    # Footer
    L += [
        "---",
        "",
        f"*Reporte generado automáticamente — {today.strftime('%Y-%m-%d %H:%M:%S')}*  ",
        "*Fuentes: Yahoo Finance (yfinance) · FRED St. Louis · CNN Markets · RSS Feeds*  ",
        "*⚠️ Solo informativo. No constituye asesoramiento financiero.*",
    ]

    # ── Escribir archivo ─────────────────────────────────────
    content = "\n".join(L)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    buy_count  = len(buys)
    sell_count = len(sells)
    news_count = len(all_news)
    alert_count = len([t for t, (s, _, _) in risk_scores.items() if s > 50])
    print(f"\n✅ Reporte guardado: {filename}")
    print(f"   💼 {len([d for d in portfolio_data if d])} activos en portafolio")
    print(f"   🟢 {buy_count} señal(es) de compra  |  🔴 {sell_count} señal(es) de venta")
    print(f"   ⚠️ {alert_count} alerta(s) de riesgo alto")
    print(f"   📰 {news_count} noticias recopiladas")
    return filename


# ─── Deep Dive: análisis profundo de un ticker ──────────────────────────────

def _safe(val, fmt=".2f", prefix="", suffix="", fallback="N/A"):
    """Formatea un valor de forma segura; devuelve fallback si es None/NaN."""
    if val is None:
        return fallback
    try:
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return fallback
        return f"{prefix}{val:{fmt}}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def _fetch_deep_data(ticker: str) -> dict:
    """Descarga datos extendidos de un ticker vía yfinance."""
    tk = yf.Ticker(ticker)

    # Precio histórico 1 año (para técnicos avanzados)
    hist = yf.download(ticker, period="1y", interval="1d",
                       progress=False, auto_adjust=True)
    close = hist["Close"].squeeze() if not hist.empty else pd.Series(dtype=float)

    # Info fundamental
    try:
        info = tk.info
    except Exception:
        info = {}

    # Recomendaciones de analistas
    try:
        recs = tk.recommendations
        if recs is not None and not recs.empty:
            recs = recs.tail(10)
        else:
            recs = None
    except Exception:
        recs = None

    # Earnings próximos y pasados
    try:
        cal = tk.calendar
    except Exception:
        cal = None

    # Financials trimestrales
    try:
        qf = tk.quarterly_financials
    except Exception:
        qf = None

    # Institutional holders
    try:
        holders = tk.institutional_holders
        if holders is not None and not holders.empty:
            holders = holders.head(10)
        else:
            holders = None
    except Exception:
        holders = None

    return {
        "info": info,
        "close": close,
        "recs": recs,
        "calendar": cal,
        "quarterly_financials": qf,
        "holders": holders,
    }


def _compute_advanced_technicals(close: pd.Series) -> dict:
    """Calcula indicadores técnicos avanzados sobre una serie de precios."""
    if close.empty or len(close) < 60:
        return {}

    price = float(close.iloc[-1])

    # RSI
    rsi = float(_rsi(close, RSI_PERIOD).iloc[-1])

    # SMAs
    sma20  = float(close.rolling(20).mean().iloc[-1])
    sma50  = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    # EMAs
    ema12 = float(close.ewm(span=12, adjust=False).mean().iloc[-1])
    ema26 = float(close.ewm(span=26, adjust=False).mean().iloc[-1])

    # MACD
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    macd_val  = float(macd_line.iloc[-1])
    macd_sig  = float(macd_signal.iloc[-1])
    macd_h    = float(macd_hist.iloc[-1])

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = float((bb_mid + 2 * bb_std).iloc[-1])
    bb_lower = float((bb_mid - 2 * bb_std).iloc[-1])
    bb_mid_v = float(bb_mid.iloc[-1])
    bb_pct   = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5

    # ATR (14)
    high = close  # Aproximación: usamos close ya que descargamos solo close
    low  = close
    tr = close.diff().abs()
    atr = float(tr.rolling(14).mean().iloc[-1])

    # Volumen promedio (si tenemos datos)
    # No tenemos volumen aquí, se maneja aparte

    # Performance periods
    def _perf(days):
        if len(close) > days:
            return float((close.iloc[-1] / close.iloc[-days] - 1) * 100)
        return None

    # 52-week high/low
    hi_52 = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
    lo_52 = float(close.tail(252).min()) if len(close) >= 252 else float(close.min())

    # Soporte / Resistencia aproximados (pivotes)
    recent = close.tail(20)
    support    = float(recent.min())
    resistance = float(recent.max())

    return {
        "price": price,
        "rsi": rsi,
        "sma20": sma20, "sma50": sma50, "sma200": sma200,
        "ema12": ema12, "ema26": ema26,
        "macd": macd_val, "macd_signal": macd_sig, "macd_hist": macd_h,
        "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid_v, "bb_pct": bb_pct,
        "atr14": atr,
        "perf_1w":  _perf(5),
        "perf_1m":  _perf(22),
        "perf_3m":  _perf(63),
        "perf_6m":  _perf(126),
        "perf_ytd": _perf(len(close) - 1),  # aprox
        "hi_52": hi_52, "lo_52": lo_52,
        "support": support, "resistance": resistance,
    }


def generate_deep_dive(ticker: str) -> str:
    """Genera un reporte Markdown detallado para un solo ticker."""
    ticker = ticker.upper().strip()
    today  = datetime.now()
    filename = f"{today.strftime('%Y_%m_%d')}_{ticker}_DeepDive.md"

    print(f"\n🔍 Deep Dive — {ticker}")
    print("=" * 55)

    print("⏳ Descargando datos...")
    data = _fetch_deep_data(ticker)
    info  = data["info"]
    close = data["close"]

    print("⏳ Calculando indicadores técnicos...")
    tech = _compute_advanced_technicals(close)

    print("⏳ Obteniendo noticias...")
    all_news = fetch_news()
    ticker_news = _filter_news(all_news, ticker)
    # Si no hay noticias filtradas, buscar por nombre de empresa
    if not ticker_news and info.get("shortName"):
        company_name = info["shortName"].split()[0].lower()
        ticker_news = [h for h in all_news if company_name in h["title"].lower()]

    print("⏳ Generando resúmenes de noticias (Hugging Face)...")
    _enrich_news_with_summaries(ticker_news)

    L: list[str] = []

    # ── Header ────────────────────────────────────────────────
    name    = info.get("shortName", ticker)
    sector  = info.get("sector", "—")
    industry = info.get("industry", "—")
    exchange = info.get("exchange", "—")
    currency = info.get("currency", "USD")

    L += [
        f"# 🔍 Deep Dive — {ticker} ({name})",
        f"> Generado el **{today.strftime('%Y-%m-%d %H:%M')}**  ",
        f"> Sector: {sector} · Industria: {industry} · Exchange: {exchange}",
        "",
    ]

    # Descripción de la empresa
    summary = info.get("longBusinessSummary", "")
    if summary:
        # Limitar a ~300 caracteres
        short_summary = summary[:300] + ("..." if len(summary) > 300 else "")
        L += [
            "## 🏢 Descripción",
            "",
            short_summary,
            "",
        ]

    # ── Fundamentales ─────────────────────────────────────────
    L += ["---", "", "## 📊 Datos Fundamentales", ""]

    mcap = info.get("marketCap")
    mcap_str = f"${mcap/1e9:,.2f}B" if mcap else "N/A"
    ev = info.get("enterpriseValue")
    ev_str = f"${ev/1e9:,.2f}B" if ev else "N/A"

    L += [
        "| Métrica | Valor |",
        "|---|---:|",
        f"| Market Cap | {mcap_str} |",
        f"| Enterprise Value | {ev_str} |",
        f"| Precio | {_safe(tech.get('price'), prefix='$')} {currency} |",
        f"| 52W High | {_safe(tech.get('hi_52'), prefix='$')} |",
        f"| 52W Low | {_safe(tech.get('lo_52'), prefix='$')} |",
        f"| P/E (TTM) | {_safe(info.get('trailingPE'))} |",
        f"| P/E Forward | {_safe(info.get('forwardPE'))} |",
        f"| PEG Ratio | {_safe(info.get('pegRatio'))} |",
        f"| P/S (TTM) | {_safe(info.get('priceToSalesTrailing12Months'))} |",
        f"| P/B | {_safe(info.get('priceToBook'))} |",
        f"| EV/EBITDA | {_safe(info.get('enterpriseToEbitda'))} |",
        f"| EPS (TTM) | {_safe(info.get('trailingEps'), prefix='$')} |",
        f"| EPS Forward | {_safe(info.get('forwardEps'), prefix='$')} |",
        f"| Dividend Yield | {_safe(info.get('dividendYield'), '.2%') if info.get('dividendYield') else 'N/A'} |",
        f"| Beta | {_safe(info.get('beta'))} |",
        f"| Short % of Float | {_safe(info.get('shortPercentOfFloat'), '.2%') if info.get('shortPercentOfFloat') else 'N/A'} |",
        "",
    ]

    # Márgenes y rentabilidad
    L += [
        "### 💰 Márgenes y Rentabilidad",
        "",
        "| Métrica | Valor |",
        "|---|---:|",
        f"| Gross Margin | {_safe(info.get('grossMargins'), '.1%') if info.get('grossMargins') else 'N/A'} |",
        f"| Operating Margin | {_safe(info.get('operatingMargins'), '.1%') if info.get('operatingMargins') else 'N/A'} |",
        f"| Profit Margin | {_safe(info.get('profitMargins'), '.1%') if info.get('profitMargins') else 'N/A'} |",
        f"| ROE | {_safe(info.get('returnOnEquity'), '.1%') if info.get('returnOnEquity') else 'N/A'} |",
        f"| ROA | {_safe(info.get('returnOnAssets'), '.1%') if info.get('returnOnAssets') else 'N/A'} |",
        f"| Revenue (TTM) | {_safe(info.get('totalRevenue'), ',.0f', prefix='$') if info.get('totalRevenue') else 'N/A'} |",
        f"| Free Cash Flow | {_safe(info.get('freeCashflow'), ',.0f', prefix='$') if info.get('freeCashflow') else 'N/A'} |",
        f"| Debt/Equity | {_safe(info.get('debtToEquity'))} |",
        f"| Current Ratio | {_safe(info.get('currentRatio'))} |",
        "",
    ]

    # ── Análisis técnico ──────────────────────────────────────
    if tech:
        L += ["---", "", "## 📈 Análisis Técnico", ""]

        # Performance
        L += [
            "### 🏃 Performance",
            "",
            "| Período | Retorno |",
            "|---|---:|",
            f"| 1 Semana | {_fmt_change(tech.get('perf_1w'))} |",
            f"| 1 Mes | {_fmt_change(tech.get('perf_1m'))} |",
            f"| 3 Meses | {_fmt_change(tech.get('perf_3m'))} |",
            f"| 6 Meses | {_fmt_change(tech.get('perf_6m'))} |",
            "",
        ]

        # Medias móviles
        price = tech["price"]
        L += [
            "### 📐 Medias Móviles",
            "",
            "| Indicador | Valor | Precio vs MA | Señal |",
            "|---|---:|---:|---|",
        ]
        for name_ma, key in [("SMA 20", "sma20"), ("SMA 50", "sma50"),
                              ("SMA 200", "sma200"), ("EMA 12", "ema12"), ("EMA 26", "ema26")]:
            val = tech.get(key)
            if val is not None:
                diff_pct = (price / val - 1) * 100
                signal = "🟢 Por encima" if price > val else "🔴 Por debajo"
                L.append(f"| {name_ma} | ${val:,.2f} | {_fmt_change(diff_pct)} | {signal} |")
        L.append("")

        # RSI + MACD + Bollinger
        rsi_val = tech["rsi"]
        rsi_zone = ("🟢 Sobrevendido" if rsi_val < RSI_OVERSOLD
                    else "🔴 Sobrecomprado" if rsi_val > RSI_OVERBOUGHT
                    else "⚪ Neutral")

        macd_signal_str = ("🟢 Alcista" if tech["macd"] > tech["macd_signal"]
                           else "🔴 Bajista")

        bb_zone = ("🔴 Cerca de banda superior" if tech["bb_pct"] > 0.8
                   else "🟢 Cerca de banda inferior" if tech["bb_pct"] < 0.2
                   else "⚪ Zona media")

        L += [
            "### 🔬 Osciladores e Indicadores",
            "",
            "| Indicador | Valor | Interpretación |",
            "|---|---:|---|",
            f"| **RSI ({RSI_PERIOD})** | {rsi_val:.1f} | {rsi_zone} |",
            f"| **MACD** | {tech['macd']:.4f} | {macd_signal_str} (señal: {tech['macd_signal']:.4f}) |",
            f"| **MACD Histograma** | {tech['macd_hist']:.4f} | {'Momentum positivo' if tech['macd_hist'] > 0 else 'Momentum negativo'} |",
            f"| **Bollinger %B** | {tech['bb_pct']:.2f} | {bb_zone} |",
            f"| **Bollinger Superior** | ${tech['bb_upper']:,.2f} | — |",
            f"| **Bollinger Inferior** | ${tech['bb_lower']:,.2f} | — |",
            f"| **ATR (14)** | ${tech['atr14']:,.2f} | Volatilidad diaria promedio |",
            "",
        ]

        # Soporte / Resistencia
        L += [
            "### 🧱 Soporte y Resistencia (20 días)",
            "",
            "| Nivel | Precio | Distancia |",
            "|---|---:|---:|",
            f"| Resistencia | ${tech['resistance']:,.2f} | {_fmt_change((tech['resistance']/price - 1)*100)} |",
            f"| Precio actual | **${price:,.2f}** | — |",
            f"| Soporte | ${tech['support']:,.2f} | {_fmt_change((tech['support']/price - 1)*100)} |",
            f"| 52W High | ${tech['hi_52']:,.2f} | {_fmt_change((tech['hi_52']/price - 1)*100)} |",
            f"| 52W Low | ${tech['lo_52']:,.2f} | {_fmt_change((tech['lo_52']/price - 1)*100)} |",
            "",
        ]

        # Resumen de señales
        signals = _determine_signal(
            tech["rsi"], price,
            tech["sma20"], tech["sma50"],
            float(close.rolling(SMA_SHORT).mean().iloc[-2]),
            float(close.rolling(SMA_LONG).mean().iloc[-2]),
        )
        L += [
            "### 🚦 Resumen de Señales",
            "",
        ]
        for s in signals:
            L.append(f"- {s}")
        if tech["macd"] > tech["macd_signal"]:
            L.append(f"- 🟢 MACD por encima de señal — momentum alcista")
        else:
            L.append(f"- 🔴 MACD por debajo de señal — momentum bajista")
        if tech.get("sma200"):
            if price > tech["sma200"]:
                L.append(f"- 🟢 Precio por encima de SMA 200 — tendencia largo plazo alcista")
            else:
                L.append(f"- 🔴 Precio por debajo de SMA 200 — tendencia largo plazo bajista")
        L.append("")

    # ── Risk Score ────────────────────────────────────────────
    print("⏳ Calculando Risk Score...")
    dd_events = _detect_upcoming_events(ticker)
    dd_tech_data = None
    if tech:
        dd_tech_data = {
            "price": tech["price"], "rsi": tech["rsi"],
            "sma_s": tech["sma20"], "sma_l": tech["sma50"],
            "sma200": tech.get("sma200"),
            "ema12": tech["ema12"], "ema26": tech["ema26"],
            "macd_hist": tech.get("macd_hist", 0),
            "macd_hist_prev": 0,
            "bb_upper": tech.get("bb_upper", 0),
            "bb_lower": tech.get("bb_lower", 0),
            "bb_pct": tech.get("bb_pct", 0.5),
            "vol_supports": [], "vol_resistances": [],
            "hist": None,
        }
        if not close.empty:
            dd_hist = yf.download(ticker, period="6mo", interval="1d",
                                  progress=False, auto_adjust=True)
            if not dd_hist.empty:
                dd_tech_data["hist"] = dd_hist
                vz = _volume_profile(dd_hist)
                vsr = _classify_volume_zones(vz, tech["price"])
                dd_tech_data["vol_supports"] = vsr["supports"]
                dd_tech_data["vol_resistances"] = vsr["resistances"]
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
    L.append(f"**{dd_score}/100 {dd_level}**")
    L.append("")
    action = _risk_action(dd_score)
    if action:
        L.append(f"> ⚡ {action}")
        L.append("")

    if dd_alerts:
        L += ["| Tipo | Alerta | Pts |", "|---|---|---:|"]
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
    if recs is not None and not recs.empty:
        L += ["---", "", "## 🎯 Recomendaciones de Analistas", ""]
        # yfinance devuelve resumen mensual: period, strongBuy, buy, hold, sell, strongSell
        if "strongBuy" in recs.columns:
            L += ["| Período | Strong Buy | Buy | Hold | Sell | Strong Sell | Total |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
            for _, row in recs.iterrows():
                sb = int(row.get("strongBuy", 0))
                b  = int(row.get("buy", 0))
                h  = int(row.get("hold", 0))
                s  = int(row.get("sell", 0))
                ss = int(row.get("strongSell", 0))
                total = sb + b + h + s + ss
                period = row.get("period", "—")
                L.append(f"| {period} | {sb} | {b} | {h} | {s} | {ss} | {total} |")
            L.append("")
        else:
            # Formato legacy por firma
            L += ["| Fecha | Firma | Acción | De | A |",
                  "|---|---|---|---|---|"]
            for _, row in recs.iterrows():
                date_str = row.name.strftime("%Y-%m-%d") if hasattr(row.name, "strftime") else str(row.name)
                firm     = row.get("Firm", "—")
                action   = row.get("To Grade", "—")
                fr       = row.get("From Grade", "—")
                act_type = row.get("Action", "—")
                L.append(f"| {date_str} | {firm} | {act_type} | {fr} | {action} |")
            L.append("")

    # Target price consensus
    target_mean = info.get("targetMeanPrice")
    target_low  = info.get("targetLowPrice")
    target_high = info.get("targetHighPrice")
    num_analysts = info.get("numberOfAnalystOpinions")
    rec_key = info.get("recommendationKey", "").replace("_", " ").title()

    if target_mean:
        price_now = tech.get("price", 0)
        upside = ((target_mean / price_now) - 1) * 100 if price_now else 0
        L += [
            "### 🎯 Price Target Consensus",
            "",
            "| Métrica | Valor |",
            "|---|---:|",
            f"| Recomendación | **{rec_key}** |",
            f"| # Analistas | {num_analysts or 'N/A'} |",
            f"| Target Bajo | ${target_low:,.2f} |" if target_low else "",
            f"| Target Medio | **${target_mean:,.2f}** |",
            f"| Target Alto | ${target_high:,.2f} |" if target_high else "",
            f"| Upside/Downside | **{_fmt_change(upside)}** |",
            "",
        ]
        L = [x for x in L if x != ""]  # limpiar líneas vacías de targets faltantes
        L.append("")

    # ── Top holders institucionales ───────────────────────────
    holders = data["holders"]
    if holders is not None and not holders.empty:
        L += ["---", "", "## 🏦 Top Holders Institucionales", ""]
        L += ["| Holder | Acciones | % Out | Valor |",
              "|---|---:|---:|---:|"]
        for _, row in holders.iterrows():
            holder_name = row.get("Holder", "—")
            shares = row.get("Shares", 0)
            pct    = row.get("pctHeld", row.get("% Out", 0))
            value  = row.get("Value", 0)
            pct_val = float(pct) * 100 if pct and float(pct) < 1 else (float(pct) if pct else 0)
            L.append(
                f"| {holder_name} | {int(shares):,} | {pct_val:.2f}% "
                f"| ${int(value):,} |" if shares and value else f"| {holder_name} | — | — | — |"
            )
        L.append("")

    # ── Noticias ──────────────────────────────────────────────
    if ticker_news:
        L += ["---", "", f"## 📰 Noticias recientes — {ticker}", ""]
        for n in ticker_news[:8]:
            summary = n.get("summary", "")
            L.append(f"- [{n['title']}]({n['link']}) — *{n['source']}*")
            if summary:
                L.append(f"  > {summary}")
        L.append("")

    # Footer
    L += [
        "---",
        "",
        f"*Deep Dive generado automáticamente — {today.strftime('%Y-%m-%d %H:%M:%S')}*  ",
        "*Fuentes: Yahoo Finance (yfinance) · RSS Feeds*  ",
        "*⚠️ Solo informativo. No constituye asesoramiento financiero.*",
    ]

    content = "\n".join(L)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n✅ Deep Dive guardado: {filename}")
    return filename


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Financial Tracker — Reporte diario o deep dive por ticker",
    )
    parser.add_argument(
        "--ticker", "-t",
        type=str,
        default=None,
        help="Ticker para generar un Deep Dive detallado (ej: --ticker NFLX)",
    )
    parser.add_argument(
        "--fibonacci", "-f",
        action="store_true",
        help="Ejecutar Fibonacci Screener — busca tickers cerca del nivel 0.618",
    )
    args = parser.parse_args()

    if args.fibonacci:
        generate_fibonacci_report()
    elif args.ticker:
        generate_deep_dive(args.ticker)
    else:
        generate_report()
