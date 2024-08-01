import sqlite3
import csv
import pandas as pd

# Importing the data from the dataset
# ------------------------------->
db_path = 'datasets\wikibooks.sqlite'

# Connect to the SQLite database
conn = sqlite3.connect(db_path)

# SQL query to select the 'Text body' column from the 'he' table
query = "SELECT `body_text` FROM he LIMIT 100"
# Use pandas to execute the query and read the data into a DataFrame
df = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# Save the extracted data to a CSV file
output_path = 'datasets/hebrew_text.txt'
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Data extracted and saved to {output_path}")