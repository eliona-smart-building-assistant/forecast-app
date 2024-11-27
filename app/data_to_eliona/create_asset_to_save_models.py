import datetime
from eliona.api_client2 import ApiClient, Configuration, AssetsApi
from eliona.api_client2.models import Asset, Attachment
import os
from eliona.api_client2.rest import ApiException
import base64
import tempfile
import mimetypes
import logging
import time

# Initialize the logger
logger = logging.getLogger(__name__)
ELIONA_API_KEY = os.getenv("API_TOKEN")
ELIONA_HOST = os.getenv("API_ENDPOINT")

configuration = Configuration(host=ELIONA_HOST, api_key={"ApiKeyAuth": ELIONA_API_KEY})


def create_asset_to_save_models():

    with ApiClient(configuration) as api_client:
        assets_api = AssetsApi(api_client)

        asset = Asset(
            global_asset_identifier="forecast_models",
            project_id="1",
            asset_type="Space",
            name="Forecast Models",
            description="This asset is used to store the trained models for the forecasting App.",
        )

        asset = assets_api.put_asset(asset)
        logger.info(f"{asset}")
        return asset


def get_asset_info_and_attachments(asset_id):
    with ApiClient(configuration) as api_client:
        assets_api = AssetsApi(api_client)

        try:
            return assets_api.get_asset_by_id(
                asset_id=asset_id, expansions=["Asset.attachments"]
            )
        except ApiException as e:
            return None


def save_model_to_eliona(model, file_name, max_retries=3, backoff_factor=2):
    """
    Adds a serialized TensorFlow model as an attachment to a specified Eliona asset.
    If an attachment with the same name exists, it replaces it.

    Args:
        model (tf.keras.Model): The TensorFlow model to attach.
        file_name (str): The name to assign to the attachment.
        max_retries (int): Maximum number of retry attempts.
        backoff_factor (int): Factor by which the wait time increases after each retry.
    """
    gai = "forecast_models"
    asset_id = get_asset_id_by_gai(gai)
    if not asset_id:
        logger.info("Asset not found.")
        return

    # Serialize the model to bytes using an in-memory buffer
    model_bytes = serialize_model_to_bytes(model)

    # Encode the bytes to a base64 string
    encoded_content = base64.b64encode(model_bytes).decode("utf-8")

    # Determine the MIME type
    mime_type, _ = mimetypes.guess_type(file_name)
    if not mime_type:
        mime_type = "application/octet-stream"  # Default MIME type

    # Create the Attachment object
    attachment = Attachment(
        name=file_name,
        content_type=mime_type,
        content=encoded_content,
    )

    attempt = 0
    while attempt < max_retries:
        try:
            with ApiClient(configuration) as api_client:
                assets_api = AssetsApi(api_client)
                # Retrieve the existing asset with attachments
                asset = assets_api.get_asset_by_id(
                    asset_id=asset_id, expansions=["Asset.attachments"]
                )

                # Initialize attachments list if necessary
                if asset.attachments is None:
                    asset.attachments = []

                # Check for existing attachment with the same name
                existing_attachments = [
                    att for att in asset.attachments if att.name == file_name
                ]

                if existing_attachments:
                    # Replace the existing attachment
                    for existing_att in existing_attachments:
                        asset.attachments.remove(existing_att)
                    asset.attachments.append(attachment)
                    logger.info(
                        f"Replaced existing attachment '{attachment.name}' in asset ID {asset_id}."
                    )
                else:
                    # Add the new attachment
                    asset.attachments.append(attachment)
                    logger.info(
                        f"Added new attachment '{attachment.name}' to asset ID {asset_id}."
                    )

                # Update the asset with the new attachments list
                updated_asset = assets_api.put_asset_by_id(
                    asset_id=asset_id, asset=asset, expansions=["Asset.attachments"]
                )
                logger.info(
                    f"Successfully updated attachments for asset ID {asset_id}."
                )
                return  # Exit after successful execution

        except ApiException as e:
            attempt += 1
            logger.warning(f"Attempt {attempt} - Error adding attachment: {e}")
            if attempt < max_retries:
                sleep_time = backoff_factor**attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(
                    f"Max retries exceeded for adding attachment to asset ID {asset_id}."
                )
                return


def get_asset_id_by_gai(gai, max_retries=5, backoff_factor=2):
    """
    Fetches the asset ID by its Global Asset Identifier (GAI) with retry logic.

    Args:
        gai (str): Global Asset Identifier.
        max_retries (int): Maximum number of retry attempts.
        backoff_factor (int): Factor by which the wait time increases after each retry.

    Returns:
        int or None: The asset ID if found, else None.
    """
    attempt = 0
    sleep_time = 1
    while attempt < max_retries:
        try:
            with ApiClient(configuration) as api_client:
                assets_api = AssetsApi(api_client)
                assets = assets_api.get_assets()
                for asset in assets:
                    if asset.global_asset_identifier == gai:
                        logging.info(f"Found asset ID {asset.id} for GAI {gai}")
                        return asset.id
                logging.warning(f"Asset with GAI {gai} not found.")
                return None
        except ApiException as e:
            attempt += 1
            logging.warning(
                f"Attempt {attempt} - Exception when calling AssetsApi->get_asset_by_gai: {e}"
            )
            if attempt < max_retries:
                sleep_time = backoff_factor**attempt
                logging.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error(f"Max retries exceeded for GAI {gai}.")
                return None


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"File {file_path} deleted.")
    else:
        logger.info(f"File {file_path} does not exist.")


import io
import tensorflow as tf


def serialize_model_to_bytes(model):
    """
    Serializes a TensorFlow model to bytes using a temporary file.

    Args:
        model (tf.keras.Model): The TensorFlow model to serialize.

    Returns:
        bytes: The serialized model in bytes.
    """
    import tempfile
    import os

    # Create a temporary file with .keras extension
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
        temp_filename = tmp_file.name

    try:
        # Save the model to the temporary file
        model.save(temp_filename)
        # Read the bytes from the file
        with open(temp_filename, "rb") as f:
            model_bytes = f.read()
    finally:
        # Delete the temporary file
        os.remove(temp_filename)

    return model_bytes


def load_model_from_eliona(file_name):
    """
    Loads a TensorFlow model from Eliona asset attachments using a temporary file.

    Args:
        file_name (str): The name of the attachment file to load.

    Returns:
        tf.keras.Model: The loaded TensorFlow model.
    """
    import base64
    import tensorflow as tf
    import tempfile
    import os

    # Define the Global Asset Identifier
    gai = "forecast_models"

    # Retrieve the asset ID
    asset_id = get_asset_id_by_gai(gai)
    if not asset_id:
        logger.info("Asset not found.")
        return None

    # Retrieve asset information along with attachments
    asset = get_asset_info_and_attachments(asset_id)
    if not asset:
        logger.info("Failed to retrieve asset information.")
        return None

    # Find the attachment with the specified file name
    attachment = next((att for att in asset.attachments if att.name == file_name), None)

    if not attachment:
        logger.info(f"Attachment '{file_name}' not found in asset ID {asset_id}.")
        return None

    try:
        # Decode the base64 content
        model_bytes = base64.b64decode(attachment.content)
        logger.info(f"Decoded model bytes for '{file_name}'.")

        # Write the bytes to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
            temp_filename = tmp_file.name
            tmp_file.write(model_bytes)

        try:
            # Load the model from the temporary file
            model = tf.keras.models.load_model(temp_filename)
            logger.info(f"TensorFlow model '{file_name}' loaded successfully.")
            model.compile()
            logger.info("Model compiled with saved parameters.")
            return model
        finally:
            # Delete the temporary file
            os.remove(temp_filename)

    except Exception as e:
        logger.info(f"Error loading model '{file_name}': {e}")
        return None


def model_exists(file_name):
    """
    Checks if a TensorFlow model with the specified filename exists in Eliona asset attachments.

    Args:
        file_name (str): The name of the model file to check.

    Returns:
        bool: True if the model exists, False otherwise.
    """
    # Define the Global Asset Identifier
    gai = "forecast_models"

    # Retrieve the asset ID
    asset_id = get_asset_id_by_gai(gai)
    if not asset_id:
        logger.info("Asset not found.")
        return False

    # Retrieve asset information along with attachments
    asset = get_asset_info_and_attachments(asset_id)
    if not asset:
        logger.info("Failed to retrieve asset information.")
        return False

    # Check if attachments exist
    if not asset.attachments:
        logger.info("No attachments found in the asset.")
        return False

    # Iterate through attachments to find a match
    for attachment in asset.attachments:
        if attachment.name == file_name:
            logger.info(f"Model '{file_name}' exists in asset ID {asset_id}.")
            return True

    # If no matching attachment is found
    logger.info(f"Model '{file_name}' does not exist in asset ID {asset_id}.")
    return False
