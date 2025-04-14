import db
from register_app import Initialize
import uvicorn
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_api():
    port = int(os.getenv("API_SERVER_PORT", 3000))
    uvicorn.run("api.openapi:app", host="0.0.0.0", port=port)

print("Startup message")
db.create_schema_and_table()
logger.info("Database schema and table created.")
Initialize()
logger.info("App initialized.")
logger.info("Starting API server")
start_api()
