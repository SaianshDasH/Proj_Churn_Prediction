
-- Day 1: Database Schema Design
-- E-Commerce Customer Churn Prediction

-- This script creates a normalized relational schema for
-- storing e-commerce customer behavioral data.
-- 
-- Tables:
--   1. customers      → Demographics & churn label
--   2. orders          → Purchase behavior metrics
--   3. engagement      → App usage & satisfaction data


-- Drop existing tables if re-running
DROP TABLE IF EXISTS engagement;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;

-- Core customer demographics and the target variable (churn)
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    gender TEXT,
    marital_status TEXT,
    city_tier INTEGER,
    tenure REAL,
    preferred_login_device TEXT,
    preferred_payment_mode TEXT,
    preferred_order_cat TEXT,
    churn INTEGER  -- Target: 1 = churned, 0 = retained
);



-- Purchase behavior and monetary metrics
CREATE TABLE IF NOT EXISTS orders (
    customer_id INTEGER PRIMARY KEY,
    order_count INTEGER,
    order_amount_hike_from_last_year REAL,
    coupon_used INTEGER,
    day_since_last_order REAL,
    cashback_amount REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);


-- App usage, satisfaction, and complaint data
CREATE TABLE IF NOT EXISTS engagement (
    customer_id INTEGER PRIMARY KEY,
    hours_on_app REAL,
    number_of_address INTEGER,
    complain INTEGER,
    satisfaction_score INTEGER,
    number_of_device_registered INTEGER,
    warehouse_to_home REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);


SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;
