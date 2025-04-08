import mysql.connector

from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(".ENV")

conn = mysql.connector.connect(
    host=os.getenv("HOSTDB"),
    user=os.getenv("USERBASE"),
    password=os.getenv("PASSWORD"),
    database=os.getenv("DATABASE"),
)
