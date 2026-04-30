# AI for Finance

Minimal FastAPI service for prediction endpoint.

## Deployed Service

- Base URL: [https://ai-for-finance-688958849481.europe-west1.run.app](https://ai-for-finance-688958849481.europe-west1.run.app)
- Swagger UI: [https://ai-for-finance-688958849481.europe-west1.run.app/docs](https://ai-for-finance-688958849481.europe-west1.run.app/docs)

## Endpoints

- `GET /` greeting
- `GET /predict` mock prediction endpoint

Example:

```bash
curl "https://ai-for-finance-688958849481.europe-west1.run.app/predict?date=2026-03-31&close=10&high=12&low=9&open=9.5&volume=1000"
```

## Environment Variables

Copy `.env.sample` to `.env` and set your values:

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `LOCAL_DATA_PATH`
