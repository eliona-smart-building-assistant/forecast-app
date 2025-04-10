import logging
from sqlalchemy import (
    Table,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
)
import os

from api.models import AssetModel

logging.basicConfig(level=logging.INFO)


def database_setup():
    db_url = os.getenv("CONNECTION_STRING")
    db_url_sql = db_url.replace("postgres", "postgresql")
    DATABASE_URL = db_url_sql
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()
    Asset = Table(
        "assets_to_forecast", metadata, autoload_with=engine, schema="forecast"
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return SessionLocal, Asset


def get_asset_by_id(id: int) -> AssetModel:
    SessionLocal, Asset = database_setup()
    logging.info(f"Fetching asset with ID {id}")

    with SessionLocal() as session:
        query = Asset.select().where(Asset.c.id == id)
        result = session.execute(query).first()
        if result is None:
            logging.warning(f"Asset with ID {id} not found.")
            return None
        return AssetModel.from_orm(result)



def update_asset(id: int, **kwargs):
    logging.info(f"Updating asset with ID {id}")
    SessionLocal, Asset = database_setup()
    
    # Filter update_values to only use keys from the table columns.
    update_values = {
        key: value for key, value in kwargs.items()
        if value is not None and key in Asset.c
    }
    if not update_values:
        logging.warning(f"No valid values provided to update for asset ID {id}")
        return

    with SessionLocal() as session:
        db_asset = session.execute(Asset.select().where(Asset.c.id == id)).first()
        if db_asset is None:
            raise ValueError(f"Asset with ID {id} not found.")

        update_query = (
            Asset.update()
            .where(Asset.c.id == id)
            .values(**update_values)
        )
        session.execute(update_query)
        session.commit()
        logging.info(f"Asset with ID {id} updated successfully.")


