import psycopg2
from psycopg2 import OperationalError
from config import db_url
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_schema_and_table():
    try:
        connection = psycopg2.connect(db_url)
        cursor = connection.cursor()
        
        create_schema_query = "CREATE SCHEMA IF NOT EXISTS forecast;"
        cursor.execute(create_schema_query)

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
            forecast_status VARCHAR(255),
            train_status VARCHAR(255),
            scaler BYTEA,
            state BYTEA,
            train BOOLEAN,
            forecast BOOLEAN,
            UNIQUE(gai, target_attribute, forecast_length)  -- Enforce uniqueness of the combination
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        connection.close()

    except OperationalError as e:
        logger.info(f"Connection failed: {e}")
