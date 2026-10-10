-- Apply once to an existing tiny_llm_benchmarks database before using --upload.
-- NULL values allow legacy benchmark rows to remain unchanged. New uploads supply a UUID.
ALTER TABLE benchmark_runs
    ADD COLUMN upload_id CHAR(36) NULL AFTER run_id,
    ADD UNIQUE KEY uq_benchmark_upload_id (upload_id);
