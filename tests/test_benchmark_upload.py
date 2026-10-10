import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_upload


class FakeCursor:
    def __init__(self, fail_on=None):
        self.statements = []
        self.lastrowid = 44
        self._next = None
        self.fail_on = fail_on

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError("simulated database failure")
        if "SELECT run_id FROM benchmark_runs" in sql:
            self._next = None
        elif "SELECT machine_id FROM machines" in sql:
            self._next = (1,)
        elif "SELECT dataset_id FROM datasets" in sql:
            self._next = (2,)
        elif "SELECT model_id FROM models" in sql:
            self._next = (3,)
        else:
            self._next = None

    def fetchone(self):
        return self._next


class FakeConnection:
    def __init__(self, cursor):
        self.fake_cursor = cursor
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.fake_cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class BenchmarkUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / "config.json"
        self.config_path.write_text(json.dumps({
            "enabled": True,
            "host": "127.0.0.1",
            "port": 3306,
            "database": "tiny_llm_benchmarks",
            "db_user": "writer",
            "allowed_hosts": ["dell"],
        }), encoding="utf-8")
        self.record = {
            "upload_id": "12345678-1234-4234-8234-123456789abc",
            "notes": "upload_id=12345678-1234-4234-8234-123456789abc",
            "hostname": "dell",
            "cpu_model": "Test CPU",
            "cpu_threads_available": 4,
            "operating_system": "Test OS",
            "dataset_name": "New Book",
            "dataset_notes": "source=test; train_tokens=90; val_tokens=10",
            "vocabulary_size": 40,
            "model_name": "TinyGPT-Test",
            "parameter_count": 1000,
            "n_embd": 16,
            "n_head": 4,
            "n_layer": 2,
            "block_size": 128,
            "dropout": 0.1,
            "tokenizer": "character",
            "start_step": 0,
            "training_steps": 10,
            "steps_this_run": 10,
            "batch_size": 4,
            "learning_rate": 0.0003,
            "pytorch_version": "test",
            "pytorch_threads": 4,
            "interop_threads": 1,
            "train_loss": 1.0,
            "validation_loss": 1.1,
            "real_seconds": 3.0,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def fake_pymysql(self, connection):
        return types.SimpleNamespace(connect=lambda **kwargs: connection)

    def test_registers_dataset_model_and_run_in_one_transaction(self):
        cursor = FakeCursor()
        connection = FakeConnection(cursor)
        with patch.object(benchmark_upload.socket, "gethostname", return_value="dell"), \
             patch.dict(os.environ, {"TINY_LLM_DB_PASSWORD": "test-password"}), \
             patch.dict(sys.modules, {"pymysql": self.fake_pymysql(connection)}):
            result = benchmark_upload.upload_benchmark(self.record, self.config_path)

        sql = [statement for statement, _ in cursor.statements]
        self.assertEqual(("inserted", 44), result)
        self.assertTrue(any("INSERT INTO datasets" in statement for statement in sql))
        self.assertTrue(any("INSERT INTO models" in statement for statement in sql))
        self.assertTrue(any("INSERT INTO benchmark_runs" in statement and "upload_id" in statement for statement in sql))
        self.assertTrue(connection.committed)
        self.assertFalse(connection.rolled_back)
        self.assertTrue(connection.closed)

    def test_rolls_back_if_a_later_insert_fails(self):
        cursor = FakeCursor(fail_on="INSERT INTO benchmark_runs")
        connection = FakeConnection(cursor)
        with patch.object(benchmark_upload.socket, "gethostname", return_value="dell"), \
             patch.dict(os.environ, {"TINY_LLM_DB_PASSWORD": "test-password"}), \
             patch.dict(sys.modules, {"pymysql": self.fake_pymysql(connection)}):
            with self.assertRaisesRegex(RuntimeError, "simulated database failure"):
                benchmark_upload.upload_benchmark(self.record, self.config_path)

        self.assertFalse(connection.committed)
        self.assertTrue(connection.rolled_back)
        self.assertTrue(connection.closed)

    def test_rejects_unauthorized_host_before_connecting(self):
        with patch.object(benchmark_upload.socket, "gethostname", return_value="euclid"):
            with self.assertRaises(PermissionError):
                benchmark_upload.upload_benchmark(self.record, self.config_path)


if __name__ == "__main__":
    unittest.main()
