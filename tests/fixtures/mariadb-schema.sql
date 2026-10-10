CREATE TABLE machines (
 machine_id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
 hostname VARCHAR(255) NOT NULL UNIQUE,
 cpu_model VARCHAR(255) NOT NULL,
 cpu_threads SMALLINT UNSIGNED NULL,
 operating_system VARCHAR(255) NULL
) ENGINE=InnoDB;
CREATE TABLE datasets (
 dataset_id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
 name VARCHAR(255) NOT NULL UNIQUE,
 vocabulary_size INT UNSIGNED NULL,
 notes TEXT NULL
) ENGINE=InnoDB;
CREATE TABLE models (
 model_id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
 name VARCHAR(255) NOT NULL UNIQUE,
 parameter_count BIGINT UNSIGNED NULL,
 n_embd INT UNSIGNED NULL,
 n_head INT UNSIGNED NULL,
 n_layer INT UNSIGNED NULL,
 block_size INT UNSIGNED NULL,
 dropout DECIMAL(6,5) NULL,
 tokenizer VARCHAR(100) NULL
) ENGINE=InnoDB;
CREATE TABLE benchmark_runs (
 run_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
 machine_id INT UNSIGNED NOT NULL,
 dataset_id INT UNSIGNED NOT NULL,
 model_id INT UNSIGNED NOT NULL,
 start_step INT UNSIGNED NULL,
 training_steps INT UNSIGNED NOT NULL CHECK (training_steps > 0),
 steps_this_run INT UNSIGNED NULL,
 batch_size INT UNSIGNED NULL,
 learning_rate DECIMAL(12,10) NULL,
 pytorch_version VARCHAR(100) NULL,
 pytorch_threads SMALLINT UNSIGNED NULL,
 interop_threads SMALLINT UNSIGNED NULL,
 train_loss DECIMAL(12,8) NULL,
 validation_loss DECIMAL(12,8) NULL,
 real_seconds DECIMAL(12,3) NULL,
 notes TEXT NULL,
 CONSTRAINT fk_ci_machine FOREIGN KEY (machine_id) REFERENCES machines(machine_id),
 CONSTRAINT fk_ci_dataset FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
 CONSTRAINT fk_ci_model FOREIGN KEY (model_id) REFERENCES models(model_id)
) ENGINE=InnoDB;


-- Representative metadata based on the existing Tiny LLM benchmark catalog.
INSERT INTO datasets (name, vocabulary_size, notes)
VALUES ('Adventures of Huckleberry Finn', 65, 'fixture=representative existing dataset');
INSERT INTO models (name, parameter_count, n_embd, n_head, n_layer, block_size, dropout, tokenizer)
VALUES ('TinyGPT-821K', 821000, 128, 4, 4, 256, 0.1, 'character');
