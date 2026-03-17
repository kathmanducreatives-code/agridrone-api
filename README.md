# AgriDrone API

`POST /predict` accepts raw JPEG bytes in the request body.

Example ESP32-compatible request:

```bash
curl -X POST "http://localhost:8000/predict?crop=rice&save_to_firebase=true" \
  -H "Content-Type: image/jpeg" \
  --data-binary @leaf.jpg
```

Notes:
- The request body must be the raw JPEG payload, not `multipart/form-data`.
- Invalid `crop` values fall back to `rice`.
- Empty bodies and invalid JPEG payloads return `400`.
