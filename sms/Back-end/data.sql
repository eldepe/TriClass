CREATE DATABASE IF NOT EXISTS smstriclass;
USE smstriclass;

CREATE TABLE contributions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) DEFAULT NULL,
    email VARCHAR(100) DEFAULT NULL,
    sms_text TEXT NOT NULL,
    suggested_category VARCHAR(50) NOT NULL,
    status ENUM('pending', 'approved', 'rejected') NOT NULL DEFAULT 'pending',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE DATABASE IF NOT EXISTS smstriclass;
USE smstriclass;

CREATE TABLE dataset_entries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL,
    label VARCHAR(50) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS admin_tbl (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  password VARCHAR(100) NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO admin_tbl (username, password)
VALUES ('admin', 'admin123'); 
