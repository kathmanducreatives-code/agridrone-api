# AgriDrone API

Mission-based batch backend for AgriDrone Guardian, with backward-compatible raw-image inference endpoints.

## Docs

- Operator and developer workflow: [AGENTS.md](/Users/prasidha/screeningpilot/screeningpilot/agridrone-api/AGENTS.md)

## Backward-Compatible Endpoints

- `POST /predict`: raw JPEG request body
- `POST /predict_form`: multipart upload for browser/manual testing
- `POST /predict_upload`: alias for Swagger/manual multipart testing
- `GET /health`: model and runtime health
- `GET /`: root probe endpoint, also supports `HEAD`

## Mission Endpoints

- `POST /missions`
- `GET /missions/{missionId}`
- `POST /missions/{missionId}/analyze`

Use [AGENTS.md](/Users/prasidha/screeningpilot/screeningpilot/agridrone-api/AGENTS.md) for local run commands, Render deployment, required environment variables, and smoke tests.
