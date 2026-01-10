# Trading Signal API

A production-ready REST API that generates cryptocurrency trading signals based on technical analysis. Built with FastAPI, this project demonstrates clean architecture, comprehensive testing, and real-world API design patterns.

## Features

- **Real-time Trading Signals**: Get BUY/SELL/HOLD recommendations with confidence scores
- **Technical Indicators**: RSI, MACD, EMA crossovers calculated from live market data
- **Multi-Indicator Analysis**: Weighted composite scoring system for more reliable signals
- **Smart Caching**: In-memory caching to respect API rate limits and improve response times
- **40+ Cryptocurrencies**: Pre-mapped support for major cryptocurrencies
- **Interactive Docs**: Auto-generated Swagger UI and ReDoc documentation
- **Type Safety**: Full type hints with Pydantic validation
- **Docker Ready**: Multi-stage Dockerfile for production deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   /signal   │    │ /indicators │    │       /health           │ │
│  │   Endpoint  │    │   Endpoint  │    │       Endpoint          │ │
│  └──────┬──────┘    └──────┬──────┘    └─────────────────────────┘ │
│         │                  │                                        │
│         └────────┬─────────┘                                        │
│                  ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Signal Generator                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────────────┐  │   │
│  │  │  RSI    │  │  MACD   │  │   EMA   │  │   Composite    │  │   │
│  │  │  40%    │  │  35%    │  │  25%    │  │   Scoring      │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘  │
│                  │                                                  │
│                  ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Indicator Calculator                        │   │
│  │        (pandas + numpy technical analysis)                   │   │
│  └──────────────────────────────────────────────────────────────┘  │
│                  │                                                  │
│                  ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     Cache Layer                              │   │
│  │           (cachetools TTLCache - 5min OHLCV)                 │   │
│  └──────────────────────────────────────────────────────────────┘  │
│                  │                                                  │
│                  ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   CoinGecko Client                           │   │
│  │              (async httpx, rate limiting)                    │   │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    CoinGecko API        │
                    │   (External Service)    │
                    └─────────────────────────┘
```

## API Endpoints

### GET `/signal/{ticker}`

Generate a trading signal for a cryptocurrency.

**Parameters:**
- `ticker` (path): Cryptocurrency symbol (e.g., BTC, ETH, SOL)

**Response:**
```json
{
  "ticker": "BTC",
  "signal": "BUY",
  "confidence": 0.72,
  "price": 97234.50,
  "indicators": {
    "rsi": 42.3,
    "macd": {
      "value": 234.5,
      "signal": 180.2,
      "histogram": 54.3
    },
    "ema_cross": "BULLISH",
    "ema_short": 96500.0,
    "ema_long": 95200.0
  },
  "timestamp": "2025-01-10T14:30:00Z"
}
```

### GET `/indicators/{ticker}`

Get raw technical indicator values.

**Parameters:**
- `ticker` (path): Cryptocurrency symbol
- `days` (query, optional): Historical data days (7-365, default: 90)

**Response:**
```json
{
  "ticker": "ETH",
  "price": 3456.78,
  "indicators": {
    "rsi": 55.2,
    "macd": {
      "value": 45.6,
      "signal": 42.1,
      "histogram": 3.5
    },
    "ema_cross": "NEUTRAL",
    "ema_short": 3450.0,
    "ema_long": 3445.0
  },
  "ohlcv_data_points": 90,
  "timestamp": "2025-01-10T14:30:00Z"
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-10T14:30:00Z"
}
```

### GET `/tickers`

List all supported cryptocurrency tickers.

### GET `/cache/stats`

Get cache performance statistics.

## Signal Generation Logic

The API uses a **weighted multi-indicator approach** to generate signals:

### Indicators Used

| Indicator | Weight | Purpose |
|-----------|--------|---------|
| RSI (14) | 40% | Identify overbought/oversold conditions |
| MACD (12,26,9) | 35% | Measure trend direction and momentum |
| EMA Cross (12,26) | 25% | Confirm trend direction |

### Scoring System

Each indicator produces a score from -1.0 (strong sell) to +1.0 (strong buy):

**RSI Scoring:**
- RSI < 30 (oversold): +0.7 to +1.0 (buy signal)
- RSI 30-40: +0.0 to +0.7 (weak buy)
- RSI 40-60: 0.0 (neutral)
- RSI 60-70: -0.0 to -0.7 (weak sell)
- RSI > 70 (overbought): -0.7 to -1.0 (sell signal)

**MACD Scoring:**
- Based on histogram value relative to MACD magnitude
- Positive histogram → positive score
- Negative histogram → negative score

**EMA Cross Scoring:**
- Short EMA > Long EMA → positive score (bullish)
- Short EMA < Long EMA → negative score (bearish)
- Strength based on percentage gap

### Final Signal Determination

```
Composite Score = (RSI_score × 0.40) + (MACD_score × 0.35) + (EMA_score × 0.25)

If composite > 0.2  → BUY
If composite < -0.2 → SELL
Otherwise           → HOLD
```

### Confidence Score

Confidence (0.0 to 1.0) is based on:
1. **Signal strength**: Stronger composite score → higher confidence
2. **Indicator agreement**: All indicators pointing same direction → bonus confidence

## Installation

### Prerequisites

- Python 3.10+
- pip

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-signal-api.git
cd trading-signal-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Run the server
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`

### Docker Setup

```bash
# Build and run with Docker
docker build -t trading-signal-api .
docker run -p 8000:8000 trading-signal-api

# Or use Docker Compose
docker-compose up --build
```

## Example Usage

### cURL

```bash
# Get trading signal for Bitcoin
curl http://localhost:8000/signal/BTC

# Get trading signal for Ethereum
curl http://localhost:8000/signal/ETH

# Get raw indicators with custom timeframe
curl "http://localhost:8000/indicators/SOL?days=30"

# Health check
curl http://localhost:8000/health

# List supported tickers
curl http://localhost:8000/tickers
```

### Python

```python
import httpx

async def get_signal(ticker: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://localhost:8000/signal/{ticker}")
        return response.json()

# Usage
signal = await get_signal("BTC")
print(f"Signal: {signal['signal']} (Confidence: {signal['confidence']})")
```

### JavaScript

```javascript
// Using fetch
const response = await fetch('http://localhost:8000/signal/BTC');
const signal = await response.json();
console.log(`Signal: ${signal.signal} (Confidence: ${signal.confidence})`);
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_indicators.py -v

# Run with output
pytest -v -s
```

## Configuration

All settings can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode |
| `COINGECKO_API_KEY` | - | Optional API key for higher rate limits |
| `CACHE_TTL_OHLCV` | `300` | OHLCV cache duration (seconds) |
| `CACHE_TTL_PRICE` | `60` | Price cache duration (seconds) |
| `RSI_PERIOD` | `14` | RSI calculation period |
| `OHLCV_DAYS` | `90` | Days of historical data to fetch |

See `.env.example` for all options.

## Project Structure

```
trading-signal-api/
├── src/
│   ├── __init__.py       # Package initialization
│   ├── main.py           # FastAPI app and routes
│   ├── config.py         # Configuration management
│   ├── models.py         # Pydantic models
│   ├── data.py           # CoinGecko API client
│   ├── cache.py          # Caching layer
│   ├── indicators.py     # Technical indicator calculations
│   └── signals.py        # Signal generation logic
├── tests/
│   ├── __init__.py
│   ├── test_indicators.py
│   └── test_signals.py
├── .env.example          # Environment template
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Future Improvements

- [ ] **Additional Indicators**: Bollinger Bands, Stochastic RSI, Volume analysis
- [ ] **Backtesting**: Historical performance analysis of signals
- [ ] **WebSocket Support**: Real-time signal streaming
- [ ] **Redis Caching**: Distributed caching for horizontal scaling
- [ ] **Rate Limiting**: Per-client rate limiting with sliding window
- [ ] **Authentication**: API key authentication for production use
- [ ] **Alerts**: Webhook notifications when signals change
- [ ] **Multiple Timeframes**: Support for different candlestick intervals
- [ ] **Portfolio Analysis**: Multi-asset signal aggregation
- [ ] **Machine Learning**: ML-based signal enhancement

## Disclaimer

This API is for **educational and informational purposes only**. It does not constitute financial advice. Cryptocurrency trading involves substantial risk of loss. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
