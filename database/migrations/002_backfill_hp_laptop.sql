-- Optional one-time backfill for an existing HP-Laptop machine record.
-- Does not delete or modify any benchmark_runs rows.
-- Review the existing machine before applying:
-- SELECT machine_id, hostname, cpu_model, cpu_cores, cpu_threads, ram_mb FROM machines WHERE hostname = 'HP-Laptop';
USE tiny_llm_benchmarks;
UPDATE machines
SET cpu_model = 'AMD Ryzen 3 3250U with Radeon Graphics',
    cpu_cores = 2,
    cpu_threads = 4,
    ram_mb = 3502
WHERE hostname = 'HP-Laptop';
-- RAM value is approximately 3.42 GiB converted to MiB.
-- SELECT machine_id, hostname, cpu_model, cpu_cores, cpu_threads, ram_mb FROM machines WHERE hostname = 'HP-Laptop';
