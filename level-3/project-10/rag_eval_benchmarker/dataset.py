# dataset.py
from schemas import RAGTestCase
from typing import List

GOLDEN_TEST_SUITE: List[RAGTestCase] = [
    RAGTestCase(
        test_id="TC-01-IDEAL",
        query="What is the default retention period for AWS CloudWatch Logs?",
        retrieved_contexts=[
            "By default, CloudWatch Logs log events are kept indefinitely and never expire.",
            "You can adjust the retention policy on each log group to range from 1 day to 10 years."
        ],
        generated_answer="By default, AWS CloudWatch Logs retains log events indefinitely without expiring, though you can configure retention from 1 day to 10 years.",
        ground_truth="CloudWatch Logs keeps logs indefinitely by default."
    ),
    RAGTestCase(
        test_id="TC-02-HALLUCINATION",
        query="How does Redis persist data to disk?",
        retrieved_contexts=[
            "Redis provides two persistence options: RDB (point-in-time snapshots) and AOF (Append-Only File).",
            "RDB creates snapshots at specified intervals, while AOF logs every write operation received by the server."
        ],
        generated_answer="Redis persists data using RDB snapshots and AOF logs. Additionally, Redis automatically mirrors all transactions to an on-disk SQLite database every 30 seconds for disaster recovery.",
        ground_truth="Redis uses RDB snapshotting and AOF logging."
    ),
    RAGTestCase(
        test_id="TC-03-LOW-RELEVANCE",
        query="What port does DNS primarily use?",
        retrieved_contexts=[
            "The Domain Name System (DNS) primarily uses UDP and TCP port 53 for query resolution.",
            "DNSSEC adds cryptographic signatures to existing DNS records to prevent spoofing."
        ],
        generated_answer="DNSSEC is critical for modern internet security because it uses public-key cryptography to validate the integrity of DNS records and mitigate cache poisoning attacks.",
        ground_truth="DNS uses port 53 over UDP and TCP."
    ),
    RAGTestCase(
        test_id="TC-04-NOISY-RETRIEVAL",
        query="How do you enable WAL mode in SQLite?",
        retrieved_contexts=[
            "SQLite is a serverless, self-contained SQL database engine widely used in mobile and embedded applications.",
            "To enable Write-Ahead Logging in SQLite, execute the PRAGMA command: PRAGMA journal_mode=WAL;",
            "SQLite database locks can be shared or exclusive depending on the current transaction type.",
            "PostgreSQL uses a multi-process architecture where each connection spawns a new backend worker process."
        ],
        generated_answer="You can enable Write-Ahead Logging (WAL) in SQLite by running the command: PRAGMA journal_mode=WAL;",
        ground_truth="Run PRAGMA journal_mode=WAL;"
    )
]