import os
import time
from typing import Any, Optional
from urllib.parse import quote

import requests


class SupabaseConfigError(RuntimeError):
    pass


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SupabaseConfigError(f"Missing required Supabase environment variable: {name}")
    return value


def _optional_env(name: str) -> str:
    return os.getenv(name, "").strip()


def _project_url() -> str:
    raw = _required_env("SUPABASE_URL").strip()
    for suffix in ("/rest/v1/", "/rest/v1"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            break
    return raw.rstrip("/")


def _rest_url() -> str:
    return f"{_project_url()}/rest/v1"


def _storage_url() -> str:
    return f"{_project_url()}/storage/v1"


def _service_key() -> str:
    return (
        _optional_env("SUPABASE_SECRET_KEY")
        or _optional_env("SUPABASE_SERVICE_ROLE_KEY")
        or _optional_env("SUPABASE_PUBLISHABLE_KEY")
        or _required_env("SUPABASE_ANON_KEY")
    )


def _anon_key() -> str:
    return _optional_env("SUPABASE_PUBLISHABLE_KEY") or _optional_env("SUPABASE_ANON_KEY")


def _bucket_name() -> str:
    return _required_env("SUPABASE_STORAGE_BUCKET")


def supabase_is_configured() -> bool:
    return bool(
        _optional_env("SUPABASE_URL")
        and (
            _optional_env("SUPABASE_SECRET_KEY")
            or _optional_env("SUPABASE_SERVICE_ROLE_KEY")
            or _optional_env("SUPABASE_PUBLISHABLE_KEY")
            or _optional_env("SUPABASE_ANON_KEY")
        )
        and _optional_env("SUPABASE_STORAGE_BUCKET")
    )


def supabase_service_role_is_configured() -> bool:
    return bool(
        _optional_env("SUPABASE_URL")
        and (
            _optional_env("SUPABASE_SECRET_KEY")
            or _optional_env("SUPABASE_SERVICE_ROLE_KEY")
        )
        and _optional_env("SUPABASE_STORAGE_BUCKET")
    )


_STATUS_CACHE: dict[str, Any] = {
    "checked_at": 0.0,
    "value": {
        "supabase_connection_ok": False,
        "supabase_connection_error": None,
    },
}


def supabase_connection_status(ttl_seconds: int = 30) -> dict[str, Any]:
    if not supabase_is_configured():
        return {
            "supabase_connection_ok": False,
            "supabase_connection_error": "Supabase URL, key, or storage bucket is not configured.",
        }

    now = time.monotonic()
    cached_at = float(_STATUS_CACHE.get("checked_at") or 0.0)
    if now - cached_at < ttl_seconds:
        return dict(_STATUS_CACHE["value"])

    try:
        response = requests.get(
            f"{_storage_url()}/bucket/{quote(_bucket_name(), safe='')}",
            headers={
                "apikey": _anon_key() or _service_key(),
                "Authorization": f"Bearer {_service_key()}",
            },
            timeout=8,
        )
        if response.ok:
            value = {
                "supabase_connection_ok": True,
                "supabase_connection_error": None,
            }
        else:
            value = {
                "supabase_connection_ok": False,
                "supabase_connection_error": response.text.strip()
                or f"Supabase HTTP {response.status_code}",
            }
    except Exception as exc:
        value = {
            "supabase_connection_ok": False,
            "supabase_connection_error": str(exc),
        }

    _STATUS_CACHE["checked_at"] = now
    _STATUS_CACHE["value"] = value
    return dict(value)


def _headers(*, prefer: Optional[str] = None) -> dict[str, str]:
    key = _service_key()
    headers = {
        "apikey": _anon_key() or key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _raise_for_supabase(response: requests.Response) -> None:
    if response.ok:
        return
    detail = response.text.strip()
    raise RuntimeError(f"Supabase HTTP {response.status_code}: {detail}")


def _table_url(table: str) -> str:
    return f"{_rest_url()}/{quote(table, safe='')}"


def insert_row(table: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        _table_url(table),
        headers=_headers(prefer="return=representation"),
        json=payload,
        timeout=20,
    )
    _raise_for_supabase(response)
    decoded = response.json()
    return decoded[0] if isinstance(decoded, list) and decoded else payload


def upsert_row(table: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        _table_url(table),
        headers=_headers(prefer="resolution=merge-duplicates,return=representation"),
        json=payload,
        timeout=20,
    )
    _raise_for_supabase(response)
    decoded = response.json()
    return decoded[0] if isinstance(decoded, list) and decoded else payload


def patch_row(table: str, key: str, value: str, payload: dict[str, Any]) -> None:
    response = requests.patch(
        f"{_table_url(table)}?{quote(key, safe='')}=eq.{quote(value, safe='')}",
        headers=_headers(),
        json=payload,
        timeout=20,
    )
    _raise_for_supabase(response)


def fetch_row(table: str, key: str, value: str) -> Optional[dict[str, Any]]:
    response = requests.get(
        f"{_table_url(table)}?{quote(key, safe='')}=eq.{quote(value, safe='')}&limit=1",
        headers=_headers(),
        timeout=20,
    )
    _raise_for_supabase(response)
    decoded = response.json()
    if not isinstance(decoded, list) or not decoded:
        return None
    return decoded[0]


def list_rows(table: str, *, limit: int = 20, order: str = "created_at.desc") -> list[dict[str, Any]]:
    response = requests.get(
        f"{_table_url(table)}?order={quote(order, safe='.')}&limit={limit}",
        headers=_headers(),
        timeout=20,
    )
    _raise_for_supabase(response)
    decoded = response.json()
    return decoded if isinstance(decoded, list) else []


def list_rows_eq(
    table: str,
    *,
    key: str,
    value: str,
    limit: int = 100,
    order: str = "created_at.desc",
) -> list[dict[str, Any]]:
    response = requests.get(
        f"{_table_url(table)}?{quote(key, safe='')}=eq.{quote(value, safe='')}"
        f"&order={quote(order, safe='.')}&limit={limit}",
        headers=_headers(),
        timeout=20,
    )
    _raise_for_supabase(response)
    decoded = response.json()
    return decoded if isinstance(decoded, list) else []


def upload_object(
    *,
    path: str,
    raw_bytes: bytes,
    content_type: str,
    metadata: Optional[dict[str, str]] = None,
) -> str:
    encoded_path = "/".join(quote(part, safe="") for part in path.strip("/").split("/"))
    url = f"{_storage_url()}/object/{quote(_bucket_name(), safe='')}/{encoded_path}"
    headers = {
        "apikey": _anon_key() or _service_key(),
        "Authorization": f"Bearer {_service_key()}",
        "Content-Type": content_type or "application/octet-stream",
        "x-upsert": "true",
    }
    response = requests.post(url, headers=headers, data=raw_bytes, timeout=30)
    _raise_for_supabase(response)
    return f"supabase://{_bucket_name()}/{path.strip('/')}"


def download_object(storage_path: str) -> tuple[bytes, str]:
    prefix = f"supabase://{_bucket_name()}/"
    key = storage_path[len(prefix) :] if storage_path.startswith(prefix) else storage_path.lstrip("/")
    encoded_path = "/".join(quote(part, safe="") for part in key.split("/"))
    url = f"{_storage_url()}/object/{quote(_bucket_name(), safe='')}/{encoded_path}"
    response = requests.get(
        url,
        headers={
            "apikey": _anon_key() or _service_key(),
            "Authorization": f"Bearer {_service_key()}",
        },
        timeout=30,
    )
    _raise_for_supabase(response)
    return response.content, response.headers.get("Content-Type", "application/octet-stream")
