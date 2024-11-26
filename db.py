import psycopg2
from psycopg2 import OperationalError
from config import db_url, db_url_sql

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
)
from sqlalchemy.orm import sessionmaker


# Function to set up the database engine and session
def setup_database():
    # Create the database engine
    DATABASE_URL = db_url_sql
    engine = create_engine(DATABASE_URL)

    # Use MetaData to reflect the 'assets_to_forecast' table from the 'forecast' schema
    metadata = MetaData()
    Asset = Table(
        "assets_to_forecast", metadata, autoload_with=engine, schema="forecast"
    )

    # Create a new session for database interactions
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return SessionLocal, Asset


def create_schema_and_table():
    try:
        # Establish connection to the PostgreSQL database
        connection = psycopg2.connect(db_url)
        cursor = connection.cursor()

        # Step 1: Create the 'forecast' schema if it doesn't exist
        create_schema_query = "CREATE SCHEMA IF NOT EXISTS forecast;"
        cursor.execute(create_schema_query)
        print("Schema 'forecast' created (or already exists).")

        # Step 2: Create the 'assets' table in the 'forecast' schema
        create_table_query = """
        CREATE TABLE IF NOT EXISTS forecast.assets_to_forecast (
            id SERIAL PRIMARY KEY,  
            gai VARCHAR(255) NOT NULL,
            target_attribute VARCHAR(255) NOT NULL,
            feature_attributes JSONB,
            forecast_length INT NOT NULL,
            start_date VARCHAR(255),
            parameters JSONB,
            datalength INT,
            hyperparameters JSONB,
            trainingparameters JSONB,
            latest_timestamp VARCHAR(255),
            context_length INT,
            processing_status VARCHAR(255),
            scaler BYTEA,
            state BYTEA,
            UNIQUE(gai, target_attribute, forecast_length)  -- Enforce uniqueness of the combination
        );
        """
        cursor.execute(create_table_query)
        print("Table 'assets' created (or already exists).")

        # Commit the changes to the database
        connection.commit()

        # Close the cursor and connection
        cursor.close()
        connection.close()

    except OperationalError as e:
        print(f"Connection failed: {e}")
