import logging
import os

import requests

from inference import decode_image_bytes

logger = logging.getLogger("agridrone.storage")

DOWNLOAD_TIMEOUT_SEC = float(os.getenv("DOWNLOAD_TIMEOUT_SEC", "20"))
MAX_IMAGE_DOWNLOAD_BYTES = int(os.getenv("MAX_IMAGE_DOWNLOAD_BYTES", str(8 * 1024 * 1024)))


class StorageDownloadError(ValueError):
    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


def download_and_validate_image(url: str) -> tuple[bytes, str]:
    if not url:
        raise StorageDownloadError("missing_storage_url", "Mission image is missing storage_url")

    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").lower()
            if not content_type.startswith("image/"):
                raise StorageDownloadError(
                    "invalid_content_type",
                    f"Unexpected content type '{content_type or 'unknown'}'",
                )

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_IMAGE_DOWNLOAD_BYTES:
                    raise StorageDownloadError(
                        "image_too_large",
                        f"Image exceeds MAX_IMAGE_DOWNLOAD_BYTES ({MAX_IMAGE_DOWNLOAD_BYTES} bytes)",
                    )
                chunks.append(chunk)
            raw_bytes = b"".join(chunks)
    except StorageDownloadError:
        raise
    except requests.Timeout as exc:
        raise StorageDownloadError("download_timeout", "Timed out while downloading mission image") from exc
    except requests.RequestException as exc:
        raise StorageDownloadError("download_failed", "Image download failed") from exc

    try:
        decode_image_bytes(raw_bytes)
    except Exception as exc:
        raise StorageDownloadError("decode_failed", "Failed to decode downloaded image bytes") from exc

    return raw_bytes, content_type
