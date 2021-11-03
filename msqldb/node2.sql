-- MariaDB dump 10.17  Distrib 10.4.8-MariaDB, for osx10.10 (x86_64)
--
-- Host: localhost    Database: node2
-- ------------------------------------------------------
-- Server version	10.4.8-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `node2info`
--

DROP TABLE IF EXISTS `node2info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `node2info` (
  `node2id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `nodename` varchar(255) NOT NULL,
  `xpredvalues` text DEFAULT NULL,
  `ypredvalues` text DEFAULT NULL,
  `predictiontype` varchar(100) DEFAULT NULL,
  `dataloss` varchar(700) DEFAULT NULL,
  PRIMARY KEY (`node2id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `node2info`
--

LOCK TABLES `node2info` WRITE;
/*!40000 ALTER TABLE `node2info` DISABLE KEYS */;
INSERT INTO `node2info` VALUES (1,'node2institution','node2123','node2',NULL,NULL,'beds',NULL);
/*!40000 ALTER TABLE `node2info` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `node2predictions`
--

DROP TABLE IF EXISTS `node2predictions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `node2predictions` (
  `jobid` int(11) NOT NULL AUTO_INCREMENT,
  `xpredvalues` text DEFAULT NULL,
  `ypredvalues` text DEFAULT NULL,
  `predictiontype` varchar(100) DEFAULT NULL,
  `dataloss` varchar(700) DEFAULT NULL,
  PRIMARY KEY (`jobid`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;


-- Dump completed on 2021-11-02 19:21:40
