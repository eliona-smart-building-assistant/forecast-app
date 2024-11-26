# FILE: api/openapi.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List  # Import Optional for making fields optional
from sqlalchemy import MetaData, Table, create_engine
from sqlalchemy.orm import sessionmaker, Session
import os
import shutil
from sqlalchemy.engine import Row
import base64


# Pydantic model for the 'assets_to_forecast' table
class AssetModel(BaseModel):
    id: Optional[int] = None
    gai: str
    target_attribute: str
    feature_attributes: Optional[List[str]] = None
    forecast_length: int
    start_date: Optional[str] = None
    parameters: Optional[dict] = None
    datalength: Optional[int] = None
    hyperparameters: Optional[dict] = None
    trainingparameters: Optional[dict] = None
    latest_timestamp: Optional[str] = None
    context_length: Optional[int] = None
    processing_status: Optional[str] = None
    scaler: Optional[str] = None  # Changed from bytes to str
    state: Optional[str] = None  # Changed from bytes to str

    @classmethod
    def from_orm(cls, asset):
        asset_dict = asset._asdict() if isinstance(asset, Row) else dict(asset)
        if asset_dict.get("scaler"):
            asset_dict["scaler"] = base64.b64encode(asset_dict["scaler"]).decode(
                "utf-8"
            )
        if asset_dict.get("state"):
            asset_dict["state"] = base64.b64encode(asset_dict["state"]).decode("utf-8")
        return cls(**asset_dict)

    def to_db_model(self):
        asset_dict = self.model_dump()
        if asset_dict.get("scaler"):
            asset_dict["scaler"] = base64.b64decode(asset_dict["scaler"])
        if asset_dict.get("state"):
            asset_dict["state"] = base64.b64decode(asset_dict["state"])
        return asset_dict


# Retrieve DATABASE_URL from environment variables
DATABASE_URL = os.getenv(
    "CONNECTION_STRING", "postgresql://user:password@localhost/dbname"
)

# Ensure the dialect is correct
DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")


def create_api(DATABASE_URL: str) -> FastAPI:
    # Create the database engine with dynamic DATABASE_URL
    engine = create_engine(DATABASE_URL)

    # Reflect the assets_to_forecast table from the forecast schema
    metadata = MetaData()
    Asset = Table(
        "assets_to_forecast", metadata, autoload_with=engine, schema="forecast"
    )

    # Create a new session for database interactions
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # FastAPI app instance
    app = FastAPI()

    # Dependency to get DB session
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @app.delete("/hyperparameter_search/{directory_name}", response_model=dict)
    def delete_hyperparameter_search(directory_name: str):
        base_path = (
            "C:/Users/sti/eliona/pythonscriptesting/forecast-app/hyperparameter_search"
        )
        directory_path = os.path.join(base_path, directory_name)

        if not os.path.exists(directory_path):
            raise HTTPException(status_code=404, detail="Directory not found")

        try:
            shutil.rmtree(directory_path)
            return {"message": f"Directory {directory_name} deleted successfully"}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error deleting directory: {e}"
            )

    @app.get("/hyperparameter_search", response_model=list)
    def list_hyperparameter_search_directories():
        base_path = (
            "C:/Users/sti/eliona/pythonscriptesting/forecast-app/hyperparameter_search"
        )
        try:
            directories = [
                d
                for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))
            ]
            return directories
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error listing directories: {e}"
            )

    # API to get all assets
    @app.get("/assets", response_model=list[AssetModel])
    def read_assets(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
        result = db.execute(Asset.select().offset(skip).limit(limit))
        assets = result.fetchall()

        # Convert the rows to Pydantic models with encoded bytes
        assets_list = [AssetModel.from_orm(row) for row in assets]

        return assets_list

    @app.get("/assets/search", response_model=AssetModel)
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

    # API to get an asset by ID
    @app.get("/assets/{id}", response_model=AssetModel)
    def read_asset(id: int, db: Session = Depends(get_db)):
        result = db.execute(Asset.select().where(Asset.c.id == id)).first()
        if result is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return AssetModel.from_orm(result)

    @app.get("/assets/gai/{gai}", response_model=AssetModel)
    def read_asset_by_gai(gai: str, db: Session = Depends(get_db)):
        result = db.execute(Asset.select().where(Asset.c.gai == gai)).first()
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Asset with GAI {gai} not found"
            )
        return AssetModel.from_orm(result)

    # API to create a new asset
    @app.post("/assets", response_model=AssetModel)
    def create_asset(asset: AssetModel, db: Session = Depends(get_db)):
        # Convert the AssetModel to a dictionary suitable for the database
        db_asset = asset.to_db_model()

        # Ensure 'processing_status' is set to "new" regardless of input
        db_asset["processing_status"] = "new"

        # **Important:** Remove 'id' if it's None to allow the database to auto-generate it
        if db_asset.get("id") is None:
            db_asset.pop("id")

        try:
            # Insert new asset data
            new_asset = Asset.insert().values(**db_asset)
            result = db.execute(new_asset)
            db.commit()

            # Retrieve the inserted asset to return
            inserted_id = result.inserted_primary_key[0]
            inserted_asset = db.execute(
                Asset.select().where(Asset.c.id == inserted_id)
            ).first()

            return AssetModel.from_orm(inserted_asset)
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error creating asset: {e}")

    # API to update an existing asset
    @app.put("/assets/{id}", response_model=AssetModel)
    def update_asset(id: int, asset: AssetModel, db: Session = Depends(get_db)):
        db_asset = db.execute(Asset.select().where(Asset.c.id == id)).first()
        if db_asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")

        # Convert the AssetModel to a dictionary suitable for the database
        db_asset_updated = asset.to_db_model()

        # Only update 'processing_status' if it's provided; otherwise, keep existing value
        if not asset.processing_status:
            db_asset_updated["processing_status"] = db_asset.processing_status

        try:
            # Update the asset in the database
            update_query = (
                Asset.update().where(Asset.c.id == id).values(**db_asset_updated)
            )
            db.execute(update_query)
            db.commit()

            # Retrieve the updated asset to return
            updated_asset = db.execute(Asset.select().where(Asset.c.id == id)).first()
            return AssetModel.from_orm(updated_asset)
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error updating asset: {e}")

    # API to delete an asset by ID
    @app.delete("/assets/{id}", response_model=AssetModel)
    def delete_asset(id: int, db: Session = Depends(get_db)):
        db_asset = db.execute(Asset.select().where(Asset.c.id == id)).first()
        if db_asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")

        delete_query = Asset.delete().where(Asset.c.id == id)
        db.execute(delete_query)
        db.commit()

        return AssetModel.from_orm(db_asset)

    return app  # Return the app instance without calling `uvicorn.run()`


# Create the FastAPI app at the module level
app = create_api(DATABASE_URL)
