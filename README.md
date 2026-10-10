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
