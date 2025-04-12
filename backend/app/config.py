import os
# Import field_validator instead of validator
from pydantic import Field, AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings
from typing import List, Union, Any

class Settings(BaseSettings):
    CLOUDFLARE_ACCOUNT_ID: str = Field(...)
    CLOUDFLARE_ACCESS_KEY_ID: str = Field(...)
    CLOUDFLARE_SECRET_ACCESS_KEY: str = Field(...)
    R2_BUCKET_NAME: str = Field(...)

    MONGODB_URI: str = Field(...)
    MONGO_DB_NAME: str = Field(...)

    # The field type remains the target Python type
    FRONTEND_ORIGIN: str = Field("*")

    # Use field_validator with mode='before'

    @property
    def r2_endpoint_url(self) -> str:
        return f"https://{self.CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com"

    @property
    def allowed_origins(self) -> List[Union[AnyHttpUrl, str]]:
        """Parses the FRONTEND_ORIGIN string into a list of origins."""
        if not self.FRONTEND_ORIGIN:
            # Handle case where env var might be empty
            return ["*"]
        # Split the comma-separated string and strip whitespace
        origins = [origin.strip() for origin in self.FRONTEND_ORIGIN.split(',') if origin.strip()]
        # Return the list of strings. Pydantic won't validate these further
        # here, but they will be used by the CORS middleware.
        # If the string was empty or only whitespace/commas, return default
        return origins if origins else ["*"]

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


# Create a single instance to be imported elsewhere
settings = Settings()

# --- Example Usage (for testing config loading) ---
if __name__ == "__main__":
    print("Loaded Settings:")
    print(f"R2 Endpoint: {settings.r2_endpoint_url}")
    print(f"R2 Bucket: {settings.R2_BUCKET_NAME}")
    print(f"MongoDB URI (masked): {settings.MONGODB_URI[:15]}...{settings.MONGODB_URI[-5:]}")
    print(f"MongoDB DB Name: {settings.MONGO_DB_NAME}")
    # Update to use the new field name
    print(f"Allowed Origins: {settings.FRONTEND_ORIGIN}")
