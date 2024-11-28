import time
import logging
from multiprocessing import Process
import sys
import os
from api.api_calls import update_asset, create_asset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.data_to_eliona.add_forecast_attributes import add_forecast_attributes

from app.forecast.forecast import forecast
from app.train_and_retrain.train_and_retrain import train_and_retrain

logger = logging.getLogger(__name__)


def app_background_worker(SessionLocal, Asset):
    logger.info("app_background_worker started")
    running_processes = []
    while True:
        with SessionLocal() as session:
            # Fetch all assets
            assets = session.execute(Asset.select()).fetchall()

            # Convert each row to a dictionary
            assets_dict = [dict(row._mapping) for row in assets]

            # Get list of current asset IDs
            current_ids = [asset["id"] for asset in assets_dict]

            # Terminate processes for assets that no longer exist
            for process_info in running_processes[:]:
                if process_info["id"] not in current_ids:
                    # Terminate forecast process if running
                    if process_info.get("forecast_process"):
                        process_info["forecast_process"].terminate()
                        process_info["forecast_process"] = None
                        logger.info(
                            f"Terminated forecast process for deleted asset ID {process_info['id']}"
                        )
                    # Terminate train process if running
                    if process_info.get("train_process"):
                        process_info["train_process"].terminate()
                        process_info["train_process"] = None
                        logger.info(
                            f"Terminated training process for deleted asset ID {process_info['id']}"
                        )
                    # Remove from running_processes
                    running_processes.remove(process_info)

            if not assets_dict:
                logger.info("No assets to process.")
            else:

                for asset in assets_dict:

                    asset_details = asset
                    id = asset["id"]
                    logger.info(f"id: {id}")

                    # Find existing processes for this asset
                    existing_process_info = next(
                        (p for p in running_processes if p["id"] == id),
                        None,
                    )

                    # Check and start/stop forecast process
                    if asset_details.get("forecast"):
                        if not existing_process_info or not existing_process_info.get(
                            "forecast_process"
                        ):
                            forecast_name_suffix = (
                                f"_forecast_{asset['forecast_length']}"
                            )

                            # Assuming add_forecast_attributes returns asset_id
                            asset_id = add_forecast_attributes(
                                asset["gai"],
                                asset["target_attribute"],
                                forecast_name_suffix,
                            )
                            forecast_process = Process(
                                target=forecast, args=(asset_details, asset_id)
                            )
                            forecast_process.start()
                            logger.info(f"Started forecast process for ID {id}")

                            if existing_process_info:
                                existing_process_info["forecast_process"] = (
                                    forecast_process
                                )
                            else:
                                running_processes.append(
                                    {
                                        "id": id,
                                        "forecast_process": forecast_process,
                                        "train_process": (
                                            existing_process_info.get("train_process")
                                            if existing_process_info
                                            else None
                                        ),
                                    }
                                )
                    else:
                        if existing_process_info and existing_process_info.get(
                            "forecast_process"
                        ):
                            existing_process_info["forecast_process"].terminate()
                            existing_process_info["forecast_process"] = None
                            logger.info(f"Terminated forecast process for ID {id}")

                    # Check and start/stop training process
                    if asset_details.get("train"):
                        if not existing_process_info or not existing_process_info.get(
                            "train_process"
                        ):
                            forecast_name_suffix = (
                                f"_forecast_{asset['forecast_length']}"
                            )

                            # Assuming add_forecast_attributes returns asset_id
                            asset_id = add_forecast_attributes(
                                asset["gai"],
                                asset["target_attribute"],
                                forecast_name_suffix,
                            )
                            train_process = Process(
                                target=train_and_retrain, args=(asset_details, asset_id)
                            )
                            train_process.start()
                            logger.info(f"Started training process for ID {id}")

                            if existing_process_info:
                                existing_process_info["train_process"] = train_process
                            else:
                                running_processes.append(
                                    {
                                        "id": id,
                                        "forecast_process": (
                                            existing_process_info.get(
                                                "forecast_process"
                                            )
                                            if existing_process_info
                                            else None
                                        ),
                                        "train_process": train_process,
                                    }
                                )
                    else:
                        if existing_process_info and existing_process_info.get(
                            "train_process"
                        ):
                            existing_process_info["train_process"].terminate()
                            existing_process_info["train_process"] = None
                            logger.info(f"Terminated training process for ID {id}")

            # Clean up processes that have both processes terminated
            running_processes = [
                p
                for p in running_processes
                if p.get("forecast_process") or p.get("train_process")
            ]
        logger.info(f"running processes: {running_processes}")
        # Sleep before checking again
        time.sleep(60)
