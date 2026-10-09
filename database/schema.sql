/*M!999999\- enable the sandbox mode */ 
CREATE DATABASE IF NOT EXISTS tiny_llm_benchmarks
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE tiny_llm_benchmarks;
-- MariaDB dump 10.19-11.8.8-MariaDB, for Linux (x86_64)
--
-- Host: localhost    Database: tiny_llm_benchmarks
-- ------------------------------------------------------
-- Server version	11.8.8-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*M!100616 SET @OLD_NOTE_VERBOSITY=@@NOTE_VERBOSITY, NOTE_VERBOSITY=0 */;

--
-- Temporary table structure for view `benchmark_leaderboard`
--

SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8mb4;
/*!50001 CREATE VIEW `benchmark_leaderboard` AS SELECT
 NULL AS `run_id`,
 NULL AS `hostname`,
 NULL AS `cpu_model`,
 NULL AS `model`,
 NULL AS `dataset`,
 NULL AS `training_steps`,
 NULL AS `runtime`,
 NULL AS `steps_per_second`,
 NULL AS `train_loss`,
 NULL AS `validation_loss`,
 NULL AS `run_date` */;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `benchmark_runs`
--

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `benchmark_runs` (
  `run_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `machine_id` int(10) unsigned NOT NULL,
  `dataset_id` int(10) unsigned NOT NULL,
  `model_id` int(10) unsigned NOT NULL,
  `start_step` int(10) unsigned DEFAULT NULL,
  `training_steps` int(10) unsigned NOT NULL,
  `steps_this_run` int(10) unsigned DEFAULT NULL,
  `batch_size` int(10) unsigned DEFAULT NULL,
  `learning_rate` decimal(12,10) DEFAULT NULL,
  `pytorch_version` varchar(100) DEFAULT NULL,
  `pytorch_threads` smallint(5) unsigned DEFAULT NULL,
  `interop_threads` smallint(5) unsigned DEFAULT NULL,
  `train_loss` decimal(12,8) DEFAULT NULL,
  `validation_loss` decimal(12,8) DEFAULT NULL,
  `real_seconds` decimal(12,3) DEFAULT NULL,
  `user_seconds` decimal(12,3) DEFAULT NULL,
  `system_seconds` decimal(12,3) DEFAULT NULL,
  `run_date` timestamp NOT NULL DEFAULT current_timestamp(),
  `notes` text DEFAULT NULL,
  PRIMARY KEY (`run_id`),
  KEY `idx_machine` (`machine_id`),
  KEY `idx_dataset` (`dataset_id`),
  KEY `idx_model` (`model_id`),
  KEY `idx_run_date` (`run_date`),
  CONSTRAINT `fk_run_dataset` FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`dataset_id`),
  CONSTRAINT `fk_run_machine` FOREIGN KEY (`machine_id`) REFERENCES `machines` (`machine_id`),
  CONSTRAINT `fk_run_model` FOREIGN KEY (`model_id`) REFERENCES `models` (`model_id`),
  CONSTRAINT `chk_training_steps` CHECK (`training_steps` > 0),
  CONSTRAINT `chk_steps_this_run` CHECK (`steps_this_run` is null or `steps_this_run` > 0),
  CONSTRAINT `chk_start_step` CHECK (`start_step` is null or `start_step` <= `training_steps`),
  CONSTRAINT `chk_step_range` CHECK (`start_step` is null or `steps_this_run` is null or `start_step` + `steps_this_run` <= `training_steps`),
  CONSTRAINT `chk_batch_size` CHECK (`batch_size` is null or `batch_size` > 0),
  CONSTRAINT `chk_learning_rate` CHECK (`learning_rate` is null or `learning_rate` > 0),
  CONSTRAINT `chk_real_seconds` CHECK (`real_seconds` is null or `real_seconds` > 0),
  CONSTRAINT `chk_train_loss` CHECK (`train_loss` is null or `train_loss` >= 0),
  CONSTRAINT `chk_validation_loss` CHECK (`validation_loss` is null or `validation_loss` >= 0)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `datasets`
--

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `datasets` (
  `dataset_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `total_characters` bigint(20) unsigned DEFAULT NULL,
  `training_characters` bigint(20) unsigned DEFAULT NULL,
  `validation_characters` bigint(20) unsigned DEFAULT NULL,
  `vocabulary_size` int(10) unsigned DEFAULT NULL,
  `notes` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`dataset_id`),
  UNIQUE KEY `uq_dataset_name` (`name`),
  CONSTRAINT `chk_dataset_total_characters` CHECK (`total_characters` is null or `total_characters` > 0),
  CONSTRAINT `chk_dataset_training_characters` CHECK (`training_characters` is null or `training_characters` > 0),
  CONSTRAINT `chk_dataset_validation_characters` CHECK (`validation_characters` is null or `validation_characters` > 0),
  CONSTRAINT `chk_dataset_vocabulary_size` CHECK (`vocabulary_size` is null or `vocabulary_size` > 0),
  CONSTRAINT `chk_dataset_character_counts` CHECK (`total_characters` is null or `training_characters` is null or `validation_characters` is null or `training_characters` <= `total_characters` and `validation_characters` <= `total_characters` - `training_characters`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `machines`
--

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `machines` (
  `machine_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `hostname` varchar(255) NOT NULL,
  `cpu_model` varchar(255) NOT NULL,
  `cpu_cores` smallint(5) unsigned DEFAULT NULL,
  `cpu_threads` smallint(5) unsigned DEFAULT NULL,
  `ram_mb` int(10) unsigned DEFAULT NULL,
  `operating_system` varchar(255) DEFAULT NULL,
  `notes` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`machine_id`),
  UNIQUE KEY `uq_machine_hostname` (`hostname`),
  CONSTRAINT `chk_machine_cpu_cores` CHECK (`cpu_cores` is null or `cpu_cores` > 0),
  CONSTRAINT `chk_machine_cpu_threads` CHECK (`cpu_threads` is null or `cpu_threads` > 0),
  CONSTRAINT `chk_machine_ram` CHECK (`ram_mb` is null or `ram_mb` > 0),
  CONSTRAINT `chk_machine_cpu_thread_count` CHECK (`cpu_cores` is null or `cpu_threads` is null or `cpu_threads` >= `cpu_cores`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `models`
--

/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `models` (
  `model_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `parameter_count` bigint(20) unsigned DEFAULT NULL,
  `n_embd` int(10) unsigned DEFAULT NULL,
  `n_head` int(10) unsigned DEFAULT NULL,
  `n_layer` int(10) unsigned DEFAULT NULL,
  `block_size` int(10) unsigned DEFAULT NULL,
  `dropout` decimal(6,5) DEFAULT NULL,
  `tokenizer` varchar(100) DEFAULT NULL,
  `notes` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`model_id`),
  UNIQUE KEY `uq_model_name` (`name`),
  CONSTRAINT `chk_model_parameter_count` CHECK (`parameter_count` is null or `parameter_count` > 0),
  CONSTRAINT `chk_model_n_embd` CHECK (`n_embd` is null or `n_embd` > 0),
  CONSTRAINT `chk_model_n_head` CHECK (`n_head` is null or `n_head` > 0),
  CONSTRAINT `chk_model_n_layer` CHECK (`n_layer` is null or `n_layer` > 0),
  CONSTRAINT `chk_model_block_size` CHECK (`block_size` is null or `block_size` > 0),
  CONSTRAINT `chk_model_dropout` CHECK (`dropout` is null or `dropout` >= 0 and `dropout` <= 1),
  CONSTRAINT `chk_model_attention_heads` CHECK (`n_embd` is null or `n_head` is null or `n_head` > 0 and `n_embd` MOD `n_head` = 0)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Final view structure for view `benchmark_leaderboard`
--

/*!50001 DROP VIEW IF EXISTS `benchmark_leaderboard`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8mb4 */;
/*!50001 SET character_set_results     = utf8mb4 */;
/*!50001 SET collation_connection      = utf8mb4_uca1400_ai_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`root`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `benchmark_leaderboard` AS select `b`.`run_id` AS `run_id`,`m`.`hostname` AS `hostname`,`m`.`cpu_model` AS `cpu_model`,`mo`.`name` AS `model`,`d`.`name` AS `dataset`,`b`.`training_steps` AS `training_steps`,sec_to_time(`b`.`real_seconds`) AS `runtime`,round(coalesce(`b`.`steps_this_run`,`b`.`training_steps`) / nullif(`b`.`real_seconds`,0),3) AS `steps_per_second`,`b`.`train_loss` AS `train_loss`,`b`.`validation_loss` AS `validation_loss`,`b`.`run_date` AS `run_date` from (((`benchmark_runs` `b` join `machines` `m` on(`m`.`machine_id` = `b`.`machine_id`)) join `models` `mo` on(`mo`.`model_id` = `b`.`model_id`)) join `datasets` `d` on(`d`.`dataset_id` = `b`.`dataset_id`)) */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*M!100616 SET NOTE_VERBOSITY=@OLD_NOTE_VERBOSITY */;

-- Dump completed on 2026-10-09 15:35:17
