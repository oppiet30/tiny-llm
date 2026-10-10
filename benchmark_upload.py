"""Optional MariaDB benchmark upload. Database credentials are never stored here."""
import json
import os
import socket
from pathlib import Path

CONFIG_PATH = Path("benchmark-upload.local.json")

def load_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Upload configuration must be a JSON object")
    return config

def upload_benchmark(record, config_path=CONFIG_PATH, db_host_override=None):
    config = load_config(config_path)
    hostname = socket.gethostname().split(".")[0].lower()
    # Per-host opt-in; Euclid has no direct MariaDB connection.
    if hostname == "euclid" or config.get("enabled") is not True:
        raise PermissionError("Direct database upload is disabled on this machine")
    allowed = config.get("allowed_hosts")
    if not isinstance(allowed, list) or hostname not in [str(name).lower() for name in allowed]:
        raise PermissionError(f"Machine {hostname!r} is not authorized for direct upload")
    if record["hostname"].lower() != hostname:
        raise ValueError("Benchmark hostname does not match this machine")

    db_host = db_host_override or config.get("host")
    db_user = os.environ.get("TINY_LLM_DB_USER") or config.get("db_user")
    db_password = os.environ.get("TINY_LLM_DB_PASSWORD")
    if not db_host or not db_user or not db_password:
        raise ValueError("Configure database host and TINY_LLM_DB_USER/TINY_LLM_DB_PASSWORD")

    try:
        import pymysql
    except ImportError as exc:
        raise RuntimeError("Direct uploads require PyMySQL: pip install PyMySQL") from exc

    connection = pymysql.connect(
        host=db_host, port=int(config.get("port", 3306)),
        user=db_user, password=db_password,
        database=config.get("database", "tiny_llm_benchmarks"),
        connect_timeout=5, read_timeout=15, write_timeout=15, autocommit=False,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT run_id FROM benchmark_runs WHERE upload_id = %s LIMIT 1", (record["upload_id"],))
            previous = cursor.fetchone()
            if previous:
                return ("already_present", previous[0])

            cursor.execute(
                "INSERT INTO machines (hostname,cpu_model,cpu_threads,operating_system) "
                "VALUES (%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                "cpu_model=VALUES(cpu_model), cpu_threads=VALUES(cpu_threads), "
                "operating_system=VALUES(operating_system)",
                (record["hostname"],record["cpu_model"],
                 record["cpu_threads_available"],record["operating_system"]),
            )
            cursor.execute("SELECT machine_id FROM machines WHERE hostname=%s", (record["hostname"],))
            machine = cursor.fetchone()
            cursor.execute("SELECT dataset_id FROM datasets WHERE name=%s", (record["dataset_name"],))
            dataset = cursor.fetchone()
            cursor.execute("SELECT model_id FROM models WHERE name=%s", (record["model_name"],))
            model = cursor.fetchone()
            if not (machine and dataset and model):
                raise ValueError("Dataset or model not found; register them in MariaDB first")
            cursor.execute(
                "INSERT INTO benchmark_runs "
                "(machine_id,dataset_id,model_id,start_step,training_steps,steps_this_run,"
                "batch_size,learning_rate,pytorch_version,pytorch_threads,interop_threads,"
                "train_loss,validation_loss,real_seconds,notes) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (machine[0],dataset[0],model[0],record["start_step"],
                 record["training_steps"],record["steps_this_run"],
                 record["batch_size"],record["learning_rate"],
                 record["pytorch_version"],record["pytorch_threads"],
                 record["interop_threads"],record["train_loss"],
                 record["validation_loss"],record["real_seconds"],record["notes"]),
            )
            run_id = cursor.lastrowid
        connection.commit()
        return ("inserted", run_id)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
