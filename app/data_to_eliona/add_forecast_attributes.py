import os
from eliona.api_client2 import (
    AssetsApi,
    AssetTypesApi,
    ApiException,
    AssetTypeAttribute,
    Configuration,
    ApiClient,
)
import logging

logger = logging.getLogger(__name__)
host = os.getenv("API_ENDPOINT")
api_key = os.getenv("API_TOKEN")
configuration = Configuration(host=host)
configuration.api_key["ApiKeyAuth"] = api_key
api_client = ApiClient(configuration)
assets_api = AssetsApi(api_client)
asset_types_api = AssetTypesApi(api_client)


def get_asset_id_by_gai(gai):
    try:
        assets = assets_api.get_assets()
        for asset in assets:
            if asset.global_asset_identifier == gai:
                return asset.id
        return asset.id
    except ApiException as e:
        logger.warning(f"Exception when calling AssetsApi->get_asset_by_gai for gai {gai}: {e}")
        return None


def get_asset_type_name(asset_id):
    try:
        asset = assets_api.get_asset_by_id(asset_id)
        return asset.asset_type
    except ApiException as e:
        logger.warning(f"Exception when calling AssetsApi->get_asset_by_id for asset_id: {asset_id}: {e}")
        return None


def get_all_attribute_names(asset_type_name):
    try:
        expansions = ["AssetType.attributes", "AssetType.asset_type_name"]
        asset_type = asset_types_api.get_asset_type_by_name(
            asset_type_name, expansions=expansions
        )
        if hasattr(asset_type, "attributes"):
            attribute_names = [attr.name for attr in asset_type.attributes]
            return attribute_names
        else:
            logger.warning(f"No attributes found for the asset type: {asset_type_name}")
            return []
    except ApiException as e:
        logger.warning(
            f"Exception when calling AssetTypesApi->get_asset_type_by_name for {asset_type_name}: {e}"
        )
        return []


def get_all_attribute_info(asset_type_name):
    try:
        expansions = ["AssetType.attributes", "AssetType.asset_type_name"]
        asset_type = asset_types_api.get_asset_type_by_name(
            asset_type_name, expansions=expansions
        )
        if hasattr(asset_type, "attributes"):
            attribute_info = {
                attr.name: attr.to_dict() for attr in asset_type.attributes
            }
            return attribute_info
        else:
            logger.warning(f"No attributes found for the asset type: {asset_type_name}")
            return {}
    except ApiException as e:
        logger.warning(
            f"Exception when calling AssetTypesApi->get_asset_type_by_name for {asset_type_name}: {e}"
        )
        return {}


def add_attribute_to_asset_type(asset_type_name, attribute_info):
    attribute = AssetTypeAttribute(**attribute_info)
    try:
        asset_types_api.post_asset_type_attribute(
            asset_type_name, attribute
        )
        logger.info(
            f"Successfully added attribute '{attribute.name}' to asset type '{asset_type_name}'."
        )
    except ApiException as e:
        logger.warning(
            f"Exception when calling AssetTypesApi->post_asset_type_attribute: {e}"
        )


def add_forecast_attributes(gai, attribute_to_forecast, forecast_name_suffix):
    asset_id = get_asset_id_by_gai(gai)
    asset_type_name = get_asset_type_name(asset_id)
    asset_type_all_attributes = get_all_attribute_names(asset_type_name)
    forecast_attribute = attribute_to_forecast + forecast_name_suffix
    if (
        attribute_to_forecast in asset_type_all_attributes
        and forecast_attribute not in asset_type_all_attributes
    ):
        all_attribute_info = get_all_attribute_info(asset_type_name)

        if attribute_to_forecast in all_attribute_info:
            forecast_attr_info = all_attribute_info[attribute_to_forecast]
            forecast_attr_info["name"] = forecast_attribute
            forecast_attr_info["subtype"] = "output"
            forecast_attr_info["translation"] = None
            add_attribute_to_asset_type(asset_type_name, forecast_attr_info)
    return asset_id


