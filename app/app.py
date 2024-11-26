import time
import logging
from multiprocessing import Process
import sys
import os
from api.api_calls import update_asset, create_asset
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.data_to_eliona.add_forecast_attributes import (
    add_forecast_attributes_to_all_assets,
)
from app.forecast.forecast import forecast
from app.train_and_retrain.train_and_retrain import train_and_retrain
from app.data_to_eliona.create_asset_to_save_models import create_asset_to_save_models


logger = logging.getLogger(__name__)


def app_background_worker(SessionLocal, Asset):
    logger.info("app_background_worker started")
    while True:
        with SessionLocal() as session:
            # Fetch all assets marked as 'new'
            new_assets = session.execute(
                Asset.select().where(Asset.c.processing_status == "new")
            ).fetchall()

            # Convert each row to a dictionary
            new_assets_dict = [dict(row._mapping) for row in new_assets]

            if not new_assets_dict:
                logging.info("No new assets to process.")
                time.sleep(60)
                continue

            all_assets_with_asset_id = add_forecast_attributes_to_all_assets(
                new_assets_dict
            )
            create_asset_to_save_models()

            for asset_id, asset_details in all_assets_with_asset_id:
                logging.info(f"Asset ID: {asset_id}")
                logging.info(f"Asset details: {asset_details}")

                id = asset_details["id"]

                # Update the asset's status to 'processing'
                session.execute(
                    Asset.update()
                    .where(Asset.c.id == id)
                    .values(processing_status="processing")
                )
                session.commit()

                # Start forecast and training processes
                forecast_process = Process(
                    target=forecast,
                    args=(
                        asset_details,
                        asset_id,
                    ),
                )

                train_process = Process(
                    target=train_and_retrain,
                    args=(
                        asset_details,
                        asset_id,
                    ),
                )

                # Start both processes
                forecast_process.start()
                # Uncomment if you also want to start the training process simultaneously
                train_process.start()

                logging.info(
                    f"Started forecast and train_and_retrain for asset ID {asset_id}"
                )

        # Sleep before checking again for new assets
        time.sleep(60)
