# AgriDrone API

## Raw ESP32 Upload

`POST /predict` accepts raw JPEG bytes in the request body.

```bash
curl -X POST "http://localhost:8000/predict?crop=rice&save_to_firebase=true" \
  -H "Content-Type: image/jpeg" \
  --data-binary @leaf.jpg
```

## Browser/UI Upload

`POST /predict_form` accepts `multipart/form-data` with a required `image` field for manual testing from the browser, frontend, or Swagger UI.

## Debug Image Inspection

Every successful prediction saves lightweight debug copies locally:
- `/debug/latest.jpg` serves the latest raw JPEG received by the API
- `/debug/latest_decoded.jpg` serves the latest decoded JPEG saved after parsing

Notes:
- `/predict` remains raw-body only and does not require `multipart/form-data`.
- Invalid `crop` values fall back to `rice`.
- Empty bodies and invalid JPEG payloads return `400`.
- `save_to_firebase` is kept for compatibility; if Firebase is not configured in this repo the response reports `firebase_saved: false`.
