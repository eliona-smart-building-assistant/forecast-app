
from fastapi import FastAPI, HTTPException, Depends

from typing import List  
from sqlalchemy import MetaData, Table, create_engine
from sqlalchemy.orm import sessionmaker, Session
import os
import shutil

import yaml
import re
from api.api_calls import get_asset_by_id
from app.data_to_eliona.add_forecast_attributes import get_asset_id_by_gai
from app.forecast.forecast import forecast
from app.get_data.api_calls import set_forecast_bool, set_forecast_status, set_train_bool, set_training_status
from app.train_and_retrain.train_and_retrain import train_and_retrain
from api.models import AssetModel, ForecastStatus, ModelModel, ThreadInfo, TrainingStatus
import threading

BASE_PATH = os.getenv("HYPERPARAMETER_SEARCH_PATH", "./hyperparameter_search")
DATABASE_URL = os.getenv("CONNECTION_STRING", "postgresql://user:password@localhost/dbname")
DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")
MODELS_DIR = "/tmp"

def create_api(DATABASE_URL: str) -> FastAPI:
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()
    Asset = Table(
        "assets_to_forecast", metadata, autoload_with=engine, schema="forecast"
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    app = FastAPI(
        title="Forecast App API",
        description="API for managing forecsting assets and models.",
        version="1.0.0",
        openapi_url="/v1/version/openapi.json",
        openapi_version="3.1.0",
    )

    app.state.thread_map = {}
    with open("openapi.yaml", "r") as f:
        openapi_yaml = yaml.safe_load(f)

    def custom_openapi():
        app.openapi_schema = openapi_yaml
        return app.openapi_schema

    app.openapi = custom_openapi
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @app.post("/v1/assets/{id}/forecast/start", response_model=dict)
    def start_forecast(id: int, db: Session = Depends(get_db)):
        existing_asset = db.execute(Asset.select().where(Asset.c.id == id)).first()
        if existing_asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        asset_model = AssetModel.from_orm(existing_asset)

        if id not in app.state.thread_map:
            app.state.thread_map[id] = ThreadInfo()
        
        if app.state.thread_map[id].forecast_running:
            raise HTTPException(status_code=400, detail="Forecast is already running for this asset")
        asset = get_asset_by_id(id=id)
        asset_id = get_asset_id_by_gai(asset.gai)
        forecast_thread = threading.Thread(target=forecast, args=(asset_model, asset_id))
        forecast_thread.start()

        set_forecast_bool(asset=asset_model, bool=True)
        app.state.thread_map[id].forecast_running = True
        app.state.thread_map[id].forecast_thread_id = forecast_thread.ident

        return {"message": f"Started forecast thread for asset {id}"}


    @app.post("/v1/assets/{id}/forecast/stop", response_model=dict)
    def stop_forecast(id: int):
        asset = get_asset_by_id(id=id)
        set_forecast_bool(asset=asset, bool=False)
        set_forecast_status(asset, ForecastStatus.STOPPING)
        if id not in app.state.thread_map or not app.state.thread_map[id].forecast_running:
            raise HTTPException(status_code=404, detail="No running forecast thread for this asset")
        app.state.thread_map[id].forecast_running = False
        return {"message": f"Signaled forecast thread for asset {id} to stop"}
        
        return {"message": f"Signaled forecast thread for asset {id} to stop"}
    @app.post("/v1/assets/{id}/train/start", response_model=dict)
    def start_training(id: int, db: Session = Depends(get_db)):
        existing_asset = db.execute(Asset.select().where(Asset.c.id == id)).first()
        if existing_asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        asset_dict = dict(existing_asset._mapping)

        if id not in app.state.thread_map:
            app.state.thread_map[id] = ThreadInfo()

        if app.state.thread_map[id].train_running:
            raise HTTPException(status_code=400, detail="Training is already running for this asset")
        asset_model = AssetModel.from_orm(existing_asset)

        train_thread = threading.Thread(target=train_and_retrain, args=(asset_model,))
        train_thread.start()

        app.state.thread_map[id].train_running = True
        app.state.thread_map[id].train_thread_id = train_thread.ident
        set_train_bool(asset=asset_model, bool=True)


        return {"message": f"Started training thread for asset {id}"}

    @app.post("/v1/assets/{id}/train/stop", response_model=dict)
    def stop_training(id: int):
        asset = get_asset_by_id(id=id)
        set_train_bool(asset=asset, bool=False)
        set_training_status(asset, TrainingStatus.STOPPING)
        if id not in app.state.thread_map or not app.state.thread_map[id].train_running:
            raise HTTPException(status_code=404, detail="No running training thread for this asset")
        app.state.thread_map[id].train_running = False
        return {"message": f"Signaled training thread for asset {id} to stop"}

    @app.get("/v1/models", response_model=List[ModelModel])
    def list_models():
        """
        Retrieve a list of all model filenames in the /tmp directory.
        """
        try:
            files = os.listdir(MODELS_DIR)
            model_files = [
                f
                for f in files
                if os.path.isfile(os.path.join(MODELS_DIR, f))
                and f.startswith("LSTM_model_")
                and f.endswith(".keras")
            ]
            return [ModelModel.from_filename(f) for f in model_files]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing models: {e}")

    @app.delete("/v1/models/{filename}", response_model=dict)
    def delete_model(filename: str):
        """
        Delete a specific model file from the /tmp directory.
        """
        if not re.match(r"^LSTM_model_\d+_\w+_\d+\.keras$", filename):
            raise HTTPException(status_code=400, detail="Invalid filename format")

        model_path = os.path.join(MODELS_DIR, filename)

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")

        try:
            os.remove(model_path)
            return {"message": f"Model '{filename}' deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting model: {e}")

    @app.delete("/v1/hyperparameter_search/{directory_name}", response_model=dict)
    def delete_hyperparameter_search(directory_name: str):
        directory_path = os.path.join(BASE_PATH, directory_name)

        if not os.path.exists(directory_path):
            raise HTTPException(status_code=404, detail="Directory not found")

        try:
            shutil.rmtree(directory_path)
            return {"message": f"Directory {directory_name} deleted successfully"}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error deleting directory: {e}"
            )

    @app.get("/v1/hyperparameter_search", response_model=list)
    def list_hyperparameter_search_directories():

        try:
            directories = [
                d
                for d in os.listdir(BASE_PATH)
                if os.path.isdir(os.path.join(BASE_PATH, d))
            ]
            return directories
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error listing directories: {e}"
            )

    @app.get("/v1/assets", response_model=list[AssetModel])
    def read_assets(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
        result = db.execute(Asset.select().offset(skip).limit(limit))
        assets = result.fetchall()

        # Convert the rows to Pydantic models with encoded bytes
        assets_list = [AssetModel.from_orm(row) for row in assets]

        return assets_list

    @app.get("/v1/assets/search", response_model=AssetModel)
    def read_asset_by_gai_target_forecast(
        gai: str,
        target_attribute: str,
        forecast_length: int,
        db: Session = Depends(get_db),
    ):
        result = db.execute(
            Asset.select()
            .where(Asset.c.gai == gai)
            .where(Asset.c.target_attribute == target_attribute)
            .where(Asset.c.forecast_length == forecast_length)
        ).first()

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Asset with GAI {gai}, target attribute {target_attribute}, and forecast length {forecast_length} not found",
            )

        return AssetModel.from_orm(result)

    @app.get("/v1/assets/{id}", response_model=AssetModel)
    def read_asset(id: int, db: Session = Depends(get_db)):
        result = db.execute(Asset.select().where(Asset.c.id == id)).first()
        if result is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return AssetModel.from_orm(result)

    @app.get("/v1/assets/gai/{gai}", response_model=AssetModel)
    def read_asset_by_gai(gai: str, db: Session = Depends(get_db)):
        result = db.execute(Asset.select().where(Asset.c.gai == gai)).first()
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Asset with GAI {gai} not found"
            )
        return AssetModel.from_orm(result)

    # API to create a new asset
    @app.post("/v1/assets", response_model=AssetModel)
    def create_asset(asset: AssetModel, db: Session = Depends(get_db)):
        asset.forecast_status = ForecastStatus.INACTIVE
        asset.train_status = TrainingStatus.INACTIVE
        db_asset = asset.to_db_model()
        db_asset.pop("id", None)
        
        try:
            insert_stmt = Asset.insert().values(**db_asset)
            result = db.execute(insert_stmt)
            db.commit()
            inserted_id = result.inserted_primary_key[0]
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error creating asset: {e}")
        
        try:
            inserted_asset = db.execute(
                Asset.select().where(Asset.c.id == inserted_id)
            ).first()
            return AssetModel.from_orm(inserted_asset)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving asset: {e}")

    @app.put("/v1/assets/{id}", response_model=AssetModel)
    def update_asset(id: int, asset: AssetModel, db: Session = Depends(get_db)):
        existing_asset = db.execute(Asset.select().where(Asset.c.id == id)).first()
        if existing_asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        new_data = asset.to_db_model()
        update_data = {k: v for k, v in new_data.items() if v is not None}

        try:
            update_query = Asset.update().where(Asset.c.id == id).values(**update_data)
            db.execute(update_query)
            db.commit()
            updated_asset = db.execute(Asset.select().where(Asset.c.id == id)).first()
            return AssetModel.from_orm(updated_asset)
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error updating asset: {e}")

    @app.delete("/v1/assets/{id}", response_model=AssetModel)
    def delete_asset(id: int, db: Session = Depends(get_db)):
        db_asset = db.execute(Asset.select().where(Asset.c.id == id)).first()
        if db_asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")

        delete_query = Asset.delete().where(Asset.c.id == id)
        db.execute(delete_query)
        db.commit()

        return AssetModel.from_orm(db_asset)

    return app  

app = create_api(DATABASE_URL)
