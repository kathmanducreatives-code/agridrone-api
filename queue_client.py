import os
from functools import lru_cache
from types import SimpleNamespace

from redis import Redis
from rq import Queue, Retry

try:
    import boto3
except ImportError:  # pragma: no cover - optional in some local envs
    boto3 = None


QUEUE_NAME = os.getenv("FLIGHT_QUEUE_NAME", "agridrone-flight-jobs")
JOB_TIMEOUT_SEC = int(os.getenv("FLIGHT_JOB_TIMEOUT_SEC", "1800"))
SQS_QUEUE_URL = os.getenv("AWS_SQS_QUEUE_URL", "").strip()
AWS_REGION = os.getenv("AWS_REGION", "").strip() or os.getenv("AWS_DEFAULT_REGION", "").strip()


def _redis_url() -> str:
    url = os.getenv("REDIS_URL", "").strip() or os.getenv("REDIS_INTERNAL_URL", "").strip()
    if not url:
        raise RuntimeError("REDIS_URL or REDIS_INTERNAL_URL must be configured for flight processing")
    return url


def redis_is_configured() -> bool:
    return bool(os.getenv("REDIS_URL", "").strip() or os.getenv("REDIS_INTERNAL_URL", "").strip())


def sqs_is_configured() -> bool:
    return bool(SQS_QUEUE_URL)


def queue_provider_name() -> str:
    if sqs_is_configured():
        return "sqs"
    if redis_is_configured():
        return "redis"
    return "none"


@lru_cache(maxsize=1)
def get_redis_connection() -> Redis:
    return Redis.from_url(_redis_url())


@lru_cache(maxsize=1)
def get_flight_queue() -> Queue:
    return Queue(
        QUEUE_NAME,
        connection=get_redis_connection(),
        default_timeout=JOB_TIMEOUT_SEC,
    )


@lru_cache(maxsize=1)
def get_sqs_client():
    if boto3 is None:
        raise RuntimeError("boto3 is required when AWS_SQS_QUEUE_URL is configured")
    kwargs = {"region_name": AWS_REGION} if AWS_REGION else {}
    return boto3.client("sqs", **kwargs)


def enqueue_flight_processing(flight_id: str):
    if sqs_is_configured():
        response = get_sqs_client().send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=flight_id,
            MessageAttributes={
                "flight_id": {
                    "StringValue": flight_id,
                    "DataType": "String",
                },
            },
        )
        return SimpleNamespace(id=response.get("MessageId", flight_id))

    return get_flight_queue().enqueue(
        "flight_batch.process_flight_job",
        flight_id,
        job_timeout=JOB_TIMEOUT_SEC,
        retry=Retry(max=3, interval=[10, 30, 60]),
    )
