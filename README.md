# tiny-llm
A tiny LLM for testing and playing with.

```
# Installation
git clone git@github.com:YOUR_USERNAME/tiny-llm.git
cd tiny-llm

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```


## Optional MariaDB benchmark upload

Training always writes a SQL fallback file under `benchmarks/`. Direct database upload is **off unless you pass `--upload`**.

```bash
# Save the SQL file only; no database connection is attempted.
python train.py --steps 10000

# Save the SQL file and then attempt a direct MariaDB upload.
python train.py --steps 10000 --upload

# Override the configured MariaDB server address for this run.
python train.py --steps 10000 --upload --db-host 192.168.1.100
```

Create a local configuration from `benchmark-upload.example.json`:

```bash
cp benchmark-upload.example.json benchmark-upload.local.json
chmod 600 benchmark-upload.local.json
```

Set `enabled` to `true`, set `host` to the MariaDB server IP, and ensure `allowed_hosts` contains the exact short hostname of this computer. The configuration file is ignored by Git. Euclid should not be enabled because it has no direct database access.

Install the optional driver with `pip install -r requirements-upload.txt`. Set `TINY_LLM_DB_USER` and `TINY_LLM_DB_PASSWORD` in the environment (or set the non-secret username in the local configuration); do not commit passwords. Use a dedicated MariaDB account with only the permissions required for benchmark inserts/updates.

The uploader requires the machine, dataset, and model to exist in the database and uses parameterized SQL within a transaction. Upload failure does not discard the SQL fallback. Uploads should only be enabled on trusted machines and trusted networks.


### Register new datasets/models and protect duplicate uploads

Before enabling direct uploads, apply the migration once to the benchmark database:

```bash
mariadb -h 192.168.1.100 -u YOUR_ADMIN_USER -p tiny_llm_benchmarks < database/migrations/001_add_upload_id.sql
```

Replace the sample address with the MariaDB server IP. This migration adds a nullable `upload_id` column and a unique index; existing benchmark rows remain valid.

With `--upload`, the uploader now creates a missing dataset row (including its vocabulary size and source/token-count notes) and a missing model row (including parameter count and architecture metadata) inside the same transaction as the benchmark run. Existing dataset/model names are reused rather than duplicated. Register the correct dataset name in `data/meta.json`; the trainer uses that name as the dataset identity.

The generated SQL fallback also registers the dataset/model, wraps the writes in a transaction, and uses the same unique `upload_id` for idempotent imports. Thus importing the fallback after a successful direct upload will not create a second benchmark run.

Run the uploader unit tests with:

```bash
python -m unittest discover -s tests -v
```

These tests use a fake database connection; they verify query flow and rollback behavior but do not replace a real MariaDB integration test.
