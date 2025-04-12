import motor.motor_asyncio
from .config import settings
import logging

logger = logging.getLogger(__name__)

client = None
db = None

async def connect_to_mongo():
    """Connects to MongoDB using Motor."""
    global client, db
    logger.info("Connecting to MongoDB...")
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.MONGODB_URI,
            # Optional: Add server selection timeout if needed
            # serverSelectionTimeoutMS=5000
        )
        # Verify connection
        await client.admin.command('ping')
        db = client[settings.MONGO_DB_NAME]
        logger.info("MongoDB connection successful.")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        # Depending on your strategy, you might want to raise the exception
        # or handle it gracefully (e.g., allow app to start but log errors)
        raise

async def close_mongo_connection():
    """Closes the MongoDB connection."""
    global client
    if client:
        logger.info("Closing MongoDB connection...")
        client.close()
        logger.info("MongoDB connection closed.")

def get_database() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    """Returns the database instance."""
    if db is None:
        # This situation should ideally not happen if connect_to_mongo is called at startup
        logger.warning("Database instance requested before connection was established.")
        # Consider raising an error or attempting connection here based on desired behavior
        raise RuntimeError("Database not connected. Ensure connect_to_mongo() is called at application startup.")
    return db

def get_recordings_collection() -> motor.motor_asyncio.AsyncIOMotorCollection:
    """Returns the specific collection for recordings."""
    database = get_database()
    return database.get_collection("audio_recordings") # Collection name

def get_speakers_collection() -> motor.motor_asyncio.AsyncIOMotorCollection:
    """Returns the specific collection for speakers."""
    database = get_database()
    return database.get_collection("speakers") # New collection name
