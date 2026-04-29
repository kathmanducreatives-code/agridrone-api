import json
import time

from rq import Worker

import flight_batch
from queue_client import (
    JOB_TIMEOUT_SEC,
    QUEUE_NAME,
    SQS_QUEUE_URL,
    get_redis_connection,
    get_sqs_client,
    queue_provider_name,
)


def _run_sqs_worker() -> None:
    client = get_sqs_client()
    print(f"[worker] polling SQS queue {SQS_QUEUE_URL}")
    while True:
        response = client.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=min(JOB_TIMEOUT_SEC, 600),
            MessageAttributeNames=["All"],
        )
        for message in response.get("Messages", []):
            receipt_handle = message["ReceiptHandle"]
            flight_id = message.get("Body", "").strip()
            try:
                if not flight_id:
                    raise RuntimeError("SQS message body did not contain a flight id")
                flight_batch.process_flight_job(flight_id)
                client.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
            except Exception as exc:  # pragma: no cover - worker path
                print(f"[worker] failed processing message {json.dumps(message)}: {exc}")
        time.sleep(1)


def main() -> None:
    if queue_provider_name() == "sqs":
        _run_sqs_worker()
        return

    worker = Worker([QUEUE_NAME], connection=get_redis_connection())
    worker.work()


if __name__ == "__main__":
    main()
