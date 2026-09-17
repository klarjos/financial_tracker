# Spec: Sistema de Alertas Predictivas — Financial Tracker

**Fecha:** 2026-06-01  
**Módulo:** `financial_tracker.py` (extensión in-place)  
**Fase:** 1 (Eventos + Técnico) — Sentimiento de noticias queda para Fase 2

---

## Objetivo

Agregar un sistema de Risk Score (0-100) que combine detección de eventos próximos y patrones técnicos de riesgo para cada ticker del portafolio y watchlist. El score se muestra con semáforo (🟢🟡🔴) en una nueva sección del reporte `.md` y en el deep dive.

---

## Arquitectura

### Módulos nuevos (funciones dentro de `financial_tracker.py`)

```
_detect_upcoming_events(ticker) -> list[dict]
_detect_technical_risks(ticker, tech_data) -> list[dict]
_calculate_risk_score(alerts) -> tuple[int, str, list[dict]]
```

Cada detector retorna una lista de alertas con formato:

```python
{
    "type": "event" | "technical",
    "signal": "earnings_soon",
    "message": "Earnings en 3 días — históricamente cae -8.2% post-report",
    "points": 25,
}
```

### Score final

`_calculate_risk_score()` suma los puntos, aplica cap a 100, y retorna:
- `score` (int 0-100)
- `level` (str: "🟢 BAJO", "🟡 MODERADO", "🔴 ALTO", "🔴🔴 CRÍTICO")
- `alerts` (list de alertas ordenadas por puntos desc)

---

## Módulo 1: Detector de Eventos

### Función: `_detect_upcoming_events(ticker) -> list[dict]`

Usa `yfinance.Ticker` para obtener calendario de eventos.

| Evento | Fuente yfinance | Lógica | Puntos |
|---|---|---|---|
| **Earnings próximos** | `tk.earnings_dates` | Si earnings < 7 días → alerta. < 3 días → mayor peso | 15 (7d), 25 (3d), 30 (mañana) |
| **Earnings recién pasados** | `tk.earnings_dates` | Si earnings fue hace < 2 días → post-earnings volatility | 10 |
| **Reacción histórica post-earnings** | `tk.earnings_history` | Promedio de `epsSurprise` últimos 4 quarters. Si históricamente falla → más peso | +5 a +15 según historial |
| **Ex-dividend** | `tk.calendar` | Si ex-dividend < 5 días → informativo | 5 (informativo, no riesgo) |

### Manejo de errores

yfinance puede no tener datos de calendar para todos los tickers (especialmente ETFs apalancados). Si falla, retorna lista vacía sin error.

---

## Módulo 2: Detector Técnico

### Función: `_detect_technical_risks(ticker, tech_data) -> list[dict]`

Recibe el dict de `fetch_technicals()` que ya existe. No descarga datos nuevos.

| Patrón | Detección | Puntos |
|---|---|---|
| **Divergencia bajista RSI** | Precio 20D high > precio 10D high anterior, pero RSI 20D high < RSI 10D high anterior | 20 |
| **RSI extremo (>80)** | RSI actual > 80 | 10 |
| **RSI extremo (<20)** | RSI actual < 20 (oversold extremo, riesgo de continuar) | 5 |
| **Pérdida de SMA 200** | Precio < SMA 200 y SMA 200 disponible | 15 |
| **Death Cross activo** | SMA 20 < SMA 50 | 10 |
| **MACD cruce bajista** | MACD histograma < 0 y era > 0 en período anterior | 10 |
| **Bollinger Squeeze** | (BB superior - BB inferior) / precio < 5% → volatilidad comprimida, explosión inminente | 10 |
| **Volumen decreciente en rally** | Precio subió >10% en 20D pero volumen promedio 10D < volumen promedio 20D | 10 |
| **Precio en resistencia fuerte** | Distancia a resistencia de Volume Profile < 2% | 5 |
| **Precio lejos de soporte** | Distancia a soporte más cercano > 15% | 5 |

### Datos requeridos del `tech_data` existente

- `rsi`, `sma20`, `sma50`, `ema12`, `ema26` — ya calculados
- `price`, `signals` — ya disponibles
- `volume_zones` — ya calculadas por `_volume_profile()`
- Nuevo: necesita `hist` (DataFrame de precios históricos) para divergencias y volumen. Se pasa como parámetro adicional.

### Cálculo de divergencia bajista RSI

```python
# Últimos 20 días del DataFrame hist
recent = hist.tail(20)
first_half = recent.head(10)
second_half = recent.tail(10)

price_high_1 = first_half["Close"].max()
price_high_2 = second_half["Close"].max()
rsi_at_high_1 = # RSI en la fecha del max de first_half
rsi_at_high_2 = # RSI en la fecha del max de second_half

if price_high_2 > price_high_1 and rsi_at_high_2 < rsi_at_high_1:
    # Divergencia bajista
```

### Cálculo de Bollinger Squeeze

```python
bb_width = (bb_upper - bb_lower) / price
if bb_width < 0.05:  # < 5%
    # Squeeze detectado
```

### Cálculo de volumen decreciente en rally

```python
price_change_20d = (price - hist["Close"].iloc[-20]) / hist["Close"].iloc[-20]
vol_avg_10d = hist["Volume"].tail(10).mean()
vol_avg_20d = hist["Volume"].tail(20).mean()

if price_change_20d > 0.10 and vol_avg_10d < vol_avg_20d * 0.8:
    # Volumen decreciente en rally
```

---

## Risk Score

### Función: `_calculate_risk_score(alerts) -> tuple[int, str, list[dict]]`

```python
raw_score = sum(a["points"] for a in alerts)
score = min(raw_score, 100)

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
```

---

## Integración en el reporte .md

### Ubicación

Nueva sección **`## ⚠️ Alertas Predictivas`** entre los indicadores macroeconómicos y la sección del portafolio.

### Formato — Tabla resumen

```markdown
## ⚠️ Alertas Predictivas

| Ticker | Risk Score | Nivel | Alertas principales |
|---|---:|---|---|
| S | 78/100 | 🔴🔴 CRÍTICO | Earnings miss + Layoffs + Downgrade |
| AAPL | 40/100 | 🟡 MODERADO | RSI >80 + Earnings en 5 días |
| META | 12/100 | 🟢 BAJO | Sin alertas significativas |
```

### Formato — Detalle (solo para 🔴 y 🔴🔴)

```markdown
### 🔴 S (SentinelOne) — 78/100

| Tipo | Alerta | Pts |
|---|---|---:|
| 📅 Evento | Earnings reportados hace 1 día — volatilidad post-earnings | 10 |
| 📅 Evento | Historial: falla estimados 2 de 4 trimestres | 15 |
| 📉 Técnico | Divergencia bajista RSI detectada | 20 |
| 📉 Técnico | MACD cruce bajista | 10 |
| 📉 Técnico | Precio por debajo de SMA 20 | — |

**Acción sugerida:** Alta volatilidad post-earnings. Esperar estabilización 2-3 sesiones.
```

### Solo se detallan tickers con score > 50

Para mantener el reporte limpio. Los tickers 🟢 y 🟡 solo aparecen en la tabla resumen.

---

## Integración en deep dive (`--ticker`)

Agregar sección **`## ⚠️ Risk Score`** después del análisis técnico y antes de recomendaciones de analistas.

Muestra el desglose completo de alertas independientemente del score.

---

## Integración en las cards del portafolio

Agregar una línea al final de cada card del portafolio y watchlist:

```markdown
**Risk Score:** 78/100 🔴🔴 CRÍTICO — Earnings miss + MACD bajista
```

Solo se muestra si score > 25 (no ensuciar cards 🟢).

---

## Cambios en `fetch_technicals()`

Actualmente retorna un dict. Se agrega:
- Pasar el DataFrame `hist` como parte del retorno (campo `"hist"`) para que los detectores técnicos puedan calcular divergencias y volumen.
- Calcular SMA 200 y Bollinger Bands en `fetch_technicals()` (actualmente solo se calculan en deep dive). Mover a `fetch_technicals()` para que estén disponibles para el risk score.

---

## Dependencias nuevas

**Ninguna.** Todo se implementa con yfinance + pandas + numpy (ya instalados).

---

## Acción sugerida automática

Basada en el score:

| Score | Acción sugerida |
|---|---|
| 0-25 | *(no se muestra)* |
| 26-50 | "Monitorear — precaución en nuevas entradas" |
| 51-75 | "Riesgo elevado — considerar stop loss ajustado o esperar" |
| 76-100 | "NO ENTRAR — múltiples alertas activas. Esperar estabilización" |

---

## Flujo de ejecución

```
generate_report():
    ...indicadores macro...

    # NUEVO: Calcular risk scores para todos los tickers
    all_risk_scores = {}
    for ticker in MY_PORTFOLIO + WATCHLIST:
        events = _detect_upcoming_events(ticker)
        tech_risks = _detect_technical_risks(ticker, tech_data, hist)
        all_alerts = events + tech_risks
        score, level, alerts = _calculate_risk_score(all_alerts)
        all_risk_scores[ticker] = (score, level, alerts)

    # NUEVO: Escribir sección de alertas
    _write_alerts_section(all_risk_scores)

    ...portafolio cards (con risk score en cada card)...
    ...watchlist cards...
    ...señales...
```

---

## Fase 2 (futura): Sentimiento de noticias

Se agregará un tercer detector `_detect_news_risks()` que:
- Busca keywords en noticias filtradas (layoffs, downgrade, miss, etc.)
- Retorna alertas con el mismo formato `{"type": "sentiment", ...}`
- Se integra al risk score redistribuyendo pesos (Eventos 30, Técnico 40, Sentimiento 30)

No se implementa en esta fase.

---

*Spec creada: 2026-06-01*
