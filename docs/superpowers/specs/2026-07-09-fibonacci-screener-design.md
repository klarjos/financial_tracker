# Spec: Fibonacci Screener — Golden Ratio (0.618)

**Fecha:** 2026-07-09  
**Módulo:** `financial_tracker.py` (extensión in-place)  
**Comando:** `--fibonacci`

---

## Objetivo

Nuevo modo de ejecución que escanea portafolio + watchlist CSV, detecta automáticamente el último swing significativo (>15%) de cada ticker, calcula el nivel de retroceso 0.618, y genera un reporte `.md` con los tickers cuyo precio actual está dentro de ±3% de ese nivel.

---

## Arquitectura

### Funciones nuevas

```
_find_significant_swing(close: pd.Series, min_amplitude: float = 0.15) -> dict | None
_fibonacci_level(swing_high: float, swing_low: float, level: float = 0.618, direction: str = "bullish") -> float
_fibonacci_screen(tickers: list[str]) -> list[dict]
generate_fibonacci_report() -> str
```

---

## Módulo 1: Detección de Swing Significativo

### Función: `_find_significant_swing(close, min_amplitude=0.15) -> dict | None`

Recibe una Serie de precios de cierre (6 meses mínimo). Retorna el último swing significativo o `None`.

**Algoritmo — Zigzag simplificado:**

1. Calcular máximos locales: puntos donde `close[i] > close[i-1]` y `close[i] > close[i+1]` (ventana de 5 días para suavizar ruido)
2. Calcular mínimos locales: puntos donde `close[i] < close[i-1]` y `close[i] < close[i+1]` (ventana de 5 días)
3. Combinar y ordenar cronológicamente
4. Recorrer de más reciente a más antiguo buscando el primer par (máximo, mínimo) o (mínimo, máximo) donde `abs(high - low) / low > min_amplitude`
5. Retornar:

```python
{
    "swing_high": float,       # precio del máximo del swing
    "swing_low": float,        # precio del mínimo del swing
    "high_date": datetime,     # fecha del máximo
    "low_date": datetime,      # fecha del mínimo
    "direction": "bullish" | "bearish",  # bullish = low→high, bearish = high→low
    "amplitude": float,        # amplitud porcentual del swing
}
```

- `"bullish"`: el mínimo ocurrió ANTES que el máximo (precio subió). El retroceso desde el máximo es oportunidad de compra.
- `"bearish"`: el máximo ocurrió ANTES que el mínimo (precio bajó). El retroceso desde el mínimo es zona de resistencia.

### Ventana de detección de extremos locales

Usar `scipy.signal.argrelextrema` si disponible, sino implementar manualmente con rolling window de orden 5 (5 días a cada lado).

**Decisión: NO agregar scipy como dependencia.** Implementar manualmente:

```python
def _local_extrema(series, order=5):
    highs, lows = [], []
    for i in range(order, len(series) - order):
        if all(series.iloc[i] >= series.iloc[i-j] for j in range(1, order+1)) and \
           all(series.iloc[i] >= series.iloc[i+j] for j in range(1, order+1)):
            highs.append(i)
        if all(series.iloc[i] <= series.iloc[i-j] for j in range(1, order+1)) and \
           all(series.iloc[i] <= series.iloc[i+j] for j in range(1, order+1)):
            lows.append(i)
    return highs, lows
```

---

## Módulo 2: Cálculo del nivel Fibonacci

### Función: `_fibonacci_level(swing_high, swing_low, level=0.618, direction="bullish") -> float`

- **Swing alcista (bullish):** El retroceso va desde el high hacia abajo.
  `fib_level = swing_high - (swing_high - swing_low) * level`
  (Precio donde el retroceso alcanza 61.8% del movimiento alcista)

- **Swing bajista (bearish):** El retroceso va desde el low hacia arriba.
  `fib_level = swing_low + (swing_high - swing_low) * level`
  (Precio donde el rebote alcanza 61.8% del movimiento bajista)

---

## Módulo 3: Screener

### Función: `_fibonacci_screen(tickers) -> list[dict]`

Para cada ticker:

1. Descargar 6 meses de datos con `yf.download(ticker, period="6mo")`
2. Llamar `_find_significant_swing(close)`
3. Si no hay swing >15%, descartar ticker
4. Calcular `_fibonacci_level(...)` con nivel 0.618
5. Calcular distancia: `(price - fib_level) / fib_level * 100`
6. Si `abs(distancia) <= 3.0%`, incluir en resultados
7. Obtener RSI actual para contexto

Retorna lista de dicts:

```python
{
    "ticker": str,
    "price": float,
    "swing_high": float,
    "swing_low": float,
    "high_date": str,
    "low_date": str,
    "direction": "bullish" | "bearish",
    "amplitude": float,
    "fib_618": float,
    "distance_pct": float,
    "rsi": float,
}
```

Ordenados por `abs(distance_pct)` ascendente (más cerca del nivel primero).

---

## Módulo 4: Generación del reporte

### Función: `generate_fibonacci_report() -> str`

Genera archivo `YYYY_MM_DD_Fibonacci.md`.

### Formato del reporte

```markdown
# 📐 Fibonacci Screener — Golden Ratio (0.618)
> Generado el **YYYY-MM-DD HH:MM**
> Tickers analizados: N | Oportunidades encontradas: M

## 🟢 Oportunidades de Compra (retroceso en swing alcista)

Tickers cuyo precio retrocedió ~61.8% de un movimiento alcista significativo (>15%).
Zona ideal para entrada en largo.

| Ticker | Precio | Swing (Low→High) | Amplitud | Nivel 0.618 | Distancia | RSI |
|--------|--------|-------------------|----------|-------------|-----------|-----|
| META   | $585.00 | $525.72→$788.15 | +49.9% | $625.91 | -6.5% | 48.2 |

## ⚠️ Zonas de Resistencia (retroceso en swing bajista)

Tickers cuyo precio rebotó ~61.8% de un movimiento bajista significativo (>15%).
Zona de posible rechazo — precaución al comprar.

| Ticker | Precio | Swing (High→Low) | Amplitud | Nivel 0.618 | Distancia | RSI |
|--------|--------|-------------------|----------|-------------|-----------|-----|

## 📊 Resumen

- Total analizados: N
- Sin swing >15%: X (descartados)
- Fuera de rango ±3%: Y
- **Oportunidades de compra: A**
- **Zonas de resistencia: B**
```

Si no hay oportunidades en alguna categoría, mostrar "> Sin oportunidades detectadas."

---

## Integración CLI

En el bloque `if __name__ == "__main__"`:

```python
if "--fibonacci" in sys.argv:
    report = generate_fibonacci_report()
    print(f"\n✅ Fibonacci Screener guardado: {report}")
    sys.exit(0)
```

---

## Datos de entrada

- **Portafolio:** variable `MY_PORTFOLIO` existente en el script
- **Watchlist:** archivo `watchlist.csv` existente (55 ETFs apalancados)
- **Período:** 6 meses (`period="6mo"`)

---

## Parámetros configurables (constantes)

```python
FIB_LEVEL = 0.618
FIB_MIN_SWING = 0.15      # amplitud mínima del swing (15%)
FIB_PROXIMITY = 0.03      # ±3% de distancia al nivel para incluir
FIB_EXTREMA_ORDER = 5     # ventana de días para extremos locales
```

---

## Dependencias nuevas

**Ninguna.** Solo usa yfinance + pandas + numpy (ya instalados).

---

## Manejo de errores

- Ticker sin datos: skip silencioso, incrementar contador "descartados"
- Ticker sin swing >15%: skip, incrementar contador
- yfinance 404 (ETFs apalancados): try/except, skip silencioso

---

*Spec creada: 2026-07-09*
