from datetime import  timedelta
import os
import time
from api.models import AssetModel, ForecastStatus
from app.forecast.perform_forecast import perform_forecast
from app.get_data.api_calls import get_forecast_bool, set_forecast_status
import websocket
import ssl
import threading
import sys
from api.api_calls import get_asset_by_id
import logging


logger = logging.getLogger(__name__)
last_processed_time = 0 



def forecast(asset: AssetModel, asset_id):
    set_forecast_status(asset, ForecastStatus.STARTING)
    process_lock = threading.Lock()

    ELIONA_API_KEY = os.getenv("API_TOKEN")
    ELIONA_HOST = os.getenv("API_ENDPOINT")

    if not ELIONA_API_KEY or not ELIONA_HOST:
        logger.info("Error: API_TOKEN or API_ENDPOINT environment variables not set.")
        return

    if ELIONA_HOST.startswith("https://"):
        base_websocket_url = (
            ELIONA_HOST.replace("https://", "wss://").rstrip("/") + "/data-listener"
        )
    elif ELIONA_HOST.startswith("http://"):
        base_websocket_url = (
            ELIONA_HOST.replace("http://", "ws://").rstrip("/") + "/data-listener"
        )
    else:
        base_websocket_url = "ws://" + ELIONA_HOST.rstrip("/") + "/data-listener"

    query_params = []
    if asset_id is not None:
        query_params.append(f"assetId={asset_id}")
        query_params.append('data_subtype="input"')

    if query_params:
        base_websocket_url += "?" + "&".join(query_params)

    headers = [f"X-API-Key: {ELIONA_API_KEY}"]
    reconnect_delay = 1 

    while True:
        
        if not get_forecast_bool(asset):
            set_forecast_status(asset, ForecastStatus.INACTIVE)
            logger.info(f"Forecast bool for {asset.id} is false. Stopping forecast thread.")
            break
        websocket_url = base_websocket_url 
        
        def on_message(ws, message):
            set_forecast_status(asset, ForecastStatus.SIGNAL_RECIEVED)
            global last_processed_time
            current_time = time.time()
            with process_lock:
                if not get_forecast_bool(asset):
                    set_forecast_status(asset, ForecastStatus.INACTIVE)
                    logger.info(f"Forecast bool for {asset.id} is false. Stopping forecast thread.")
                    ws.close()  
                    return
                if (current_time - last_processed_time) >= 5:
                    last_processed_time = current_time
                    logger.info(f"Recieved message:{message}")
                    try:
                        if get_asset_by_id(id=asset.id) is None:
                            set_forecast_status(asset, ForecastStatus.ASSET_NOT_FOUND)
                            logger.warning(f"Asset for {asset.id} does not exist")
                            sys.exit()
                        perform_forecast(asset, asset_id)
                    except Exception as e:
                        logger.exception("Error processing message {e}")
                set_forecast_status(asset, ForecastStatus.WAITING_FOR_DATA)

        def on_error(ws, error):
            logger.warning(f"WebSocket {asset.id} error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.warning(
                f"WebSocket connection for {asset.id} closed. Code: {close_status_code}, Message: {close_msg}"
            )

        def on_open(ws):
            set_forecast_status(asset, ForecastStatus.WAITING_FOR_DATA)
            logger.info(f"WebSocket for {asset.id} connection opened")
            nonlocal reconnect_delay
            reconnect_delay = 3 

        ws = websocket.WebSocketApp(
            websocket_url,
            header=headers,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        sslopt = {"cert_reqs": ssl.CERT_NONE}

        try:
            ws.run_forever(sslopt=sslopt, ping_interval=10, ping_timeout=8)
        except KeyboardInterrupt:
            logger.info("WebSocket connection closed by user.")
            break
        except Exception as e:
            logger.info(f"Exception occurred in {asset.id}: {e}")

        time.sleep(reconnect_delay)

