FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV DATA_TICKERS=BTC-USD,ETH-USD,SPY
# delta × period_years = jours d’historique initial (365 × 1 ≈ 1 an)
ENV DATA_PERIOD_YEARS=1
ENV DATA_DELTA_DAYS=365

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY setup.py .

COPY models models

RUN python -m app.ml_logic.data

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.api.fast:app --host 0.0.0.0 --port ${PORT}"]
