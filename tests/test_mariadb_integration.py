"""Real MariaDB integration tests. Run only when TINY_LLM_TEST_DB_HOST is set."""
import json
import os
import socket
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pymysql
import benchmark_upload


@unittest.skipUnless(os.environ.get("TINY_LLM_TEST_DB_HOST"), "MariaDB integration service not configured")
class MariaDBUploadIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = os.environ["TINY_LLM_TEST_DB_HOST"]
        cls.port = int(os.environ.get("TINY_LLM_TEST_DB_PORT", "3306"))
        cls.user = os.environ.get("TINY_LLM_TEST_DB_USER", "root")
        cls.password = os.environ.get("TINY_LLM_TEST_DB_PASSWORD", "ci-root-password")
        cls.database = os.environ.get("TINY_LLM_TEST_DB_NAME", "tiny_llm_benchmarks")

    def setUp(self):
        self.hostname = socket.gethostname().split(".")[0].lower()
        self.config_path = Path(f"/tmp/tiny-llm-upload-{uuid.uuid4()}.json")
        self.config_path.write_text(json.dumps({
            "enabled": True, "host": self.host, "port": self.port,
            "database": self.database, "db_user": self.user,
            "allowed_hosts": [self.hostname],
        }), encoding="utf-8")
        self.upload_id = str(uuid.uuid4())
        self.dataset = "CI dataset " + self.upload_id
        self.model = "CI model " + self.upload_id
        self.record = {
            "upload_id": self.upload_id,
            "notes": "CI integration test " + self.upload_id,
            "hostname": self.hostname,
            "cpu_model": "CI test CPU",
            "cpu_threads_available": 2,
            "operating_system": "GitHub Actions",
            "dataset_name": self.dataset,
            "dataset_notes": "fixture=synthetic; source=CI",
            "vocabulary_size": 32,
            "model_name": self.model,
            "parameter_count": 1000,
            "n_embd": 16, "n_head": 4, "n_layer": 2,
            "block_size": 64, "dropout": 0.1, "tokenizer": "character",
            "start_step": 0, "training_steps": 10, "steps_this_run": 10,
            "batch_size": 2, "learning_rate": 0.0003,
            "pytorch_version": "CI", "pytorch_threads": 2, "interop_threads": 1,
            "train_loss": 1.0, "validation_loss": 1.1, "real_seconds": 2.0,
        }

    def tearDown(self):
        try:
            with self.connect() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM benchmark_runs WHERE upload_id=%s", (self.upload_id,))
                cur.execute("DELETE FROM datasets WHERE name=%s", (self.dataset,))
                cur.execute("DELETE FROM models WHERE name=%s", (self.model,))
                conn.commit()
        finally:
            self.config_path.unlink(missing_ok=True)

    def connect(self):
        return pymysql.connect(host=self.host, port=self.port, user=self.user,
                               password=self.password, database=self.database,
                               autocommit=False)

    def upload(self):
        return benchmark_upload.upload_benchmark(self.record, self.config_path)

    def test_registers_metadata_and_benchmark_in_real_database(self):
        with patch.object(benchmark_upload.socket, "gethostname", return_value=self.hostname):
            status, run_id = self.upload()
        self.assertEqual("inserted", status)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM datasets WHERE name=%s", (self.dataset,))
            self.assertEqual(1, cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM models WHERE name=%s", (self.model,))
            self.assertEqual(1, cur.fetchone()[0])
            cur.execute("SELECT upload_id,dataset_id,model_id FROM benchmark_runs WHERE run_id=%s", (run_id,))
            row = cur.fetchone()
            self.assertEqual(self.upload_id, row[0])
            self.assertIsNotNone(row[1])
            self.assertIsNotNone(row[2])


    def test_failed_benchmark_insert_rolls_back_new_dataset_and_model(self):
        self.record["training_steps"] = 0  # rejected by the MariaDB CHECK constraint
        with patch.object(benchmark_upload.socket, "gethostname", return_value=self.hostname):
            with self.assertRaises(Exception):
                self.upload()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM datasets WHERE name=%s", (self.dataset,))
            self.assertEqual(0, cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM models WHERE name=%s", (self.model,))
            self.assertEqual(0, cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM benchmark_runs WHERE upload_id=%s", (self.upload_id,))
            self.assertEqual(0, cur.fetchone()[0])

    def test_retry_with_same_upload_id_does_not_duplicate_run(self):
        with patch.object(benchmark_upload.socket, "gethostname", return_value=self.hostname):
            first = self.upload()
            second = self.upload()
        self.assertEqual("inserted", first[0])
        self.assertEqual("already_present", second[0])
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM benchmark_runs WHERE upload_id=%s", (self.upload_id,))
            self.assertEqual(1, cur.fetchone()[0])


if __name__ == "__main__":
    unittest.main()
