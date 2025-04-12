# Twi Speech Data Collection Backend API

This FastAPI application serves as the backend for collecting Twi speech audio recordings. It handles audio file uploads to Cloudflare R2 and stores associated metadata in MongoDB.

## Features

-   Upload endpoint (`/upload/audio`) accepting `multipart/form-data`.
-   Stores audio files securely in Cloudflare R2 object storage.
-   Stores recording metadata (participant info, prompt details, R2 URL) in MongoDB.
-   Uses Pydantic for data validation and configuration management.
-   Asynchronous processing with FastAPI and Motor.
-   Configured for deployment on Render.com.
-   Handles CORS for frontend integration.

## Project Structure

```
twi_speech_backend/
├── .env              # Local environment variables (DO NOT COMMIT)
├── .gitignore        # Git ignore file
├── requirements.txt  # Python dependencies
├── render.yaml       # Render deployment configuration
├── app/              # Main application code
│   ├── __init__.py
│   ├── main.py       # FastAPI app setup and main endpoint
│   ├── config.py     # Configuration loading (Pydantic settings)
│   ├── models.py     # Pydantic models for request/response
│   ├── crud.py       # Database interaction logic (Create operations)
│   ├── database.py   # MongoDB connection setup (Motor)
│   └── r2.py         # Cloudflare R2 interaction logic (boto3)
└── README.md         # Project instructions
```

## Setup and Running Locally

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd twi_speech_backend
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**
    Copy the example below into a file named `.env` in the project root and fill in your actual credentials. **DO NOT COMMIT THIS FILE.**
    ```dotenv
    # Cloudflare R2 Credentials
    CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
    CLOUDFLARE_ACCESS_KEY_ID=your_new_r2_access_key_id
    CLOUDFLARE_SECRET_ACCESS_KEY=your_new_r2_secret_access_key
    R2_BUCKET_NAME=your-twi-audio-bucket-name

    # MongoDB Credentials
    MONGODB_URI=mongodb+srv://<user>:<password>@<cluster-url>/<db_name>?retryWrites=true&w=majority
    MONGO_DB_NAME=twi_speech_data

    # Optional: CORS Origins for local development (comma-separated)
    FRONTEND_ORIGIN="http://localhost:3000,http://localhost:8081,*"
    ```

5.  **Run the application:**
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    The API will be available at `http://localhost:8000`. Access the interactive documentation at `http://localhost:8000/docs`.

## Deployment to Render.com

1.  **Push your code** to a GitHub or GitLab repository (ensure `.env` is in `.gitignore`!).
2.  **Create a new Web Service** on Render.com, connecting it to your repository.
3.  **Configure the service:**
    *   Use the settings provided in `render.yaml` (Render might detect some automatically).
    *   Set the `Start Command`: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
    *   Set the `Build Command`: `pip install --upgrade pip && pip install -r requirements.txt`
4.  **Add Environment Variables:**
    *   Go to the "Environment" tab for your service on Render.
    *   Add **Secret Files** or **Environment Variables** for:
        *   `CLOUDFLARE_ACCOUNT_ID`
        *   `CLOUDFLARE_ACCESS_KEY_ID`
        *   `CLOUDFLARE_SECRET_ACCESS_KEY`
        *   `R2_BUCKET_NAME`
        *   `MONGODB_URI`
        *   `MONGO_DB_NAME`
        *   `FRONTEND_ORIGIN` (Set this to the URL of your deployed frontend application, e.g., `https://your-app-name.onrender.com` or specific domains).
    *   **IMPORTANT**: Mark sensitive variables like keys and URIs as "Secret".
5.  **Deploy:** Trigger a manual deploy or let Render deploy automatically on code changes.

## Cloudflare R2 Bucket Configuration

**IMPORTANT:** For the `file_url` returned by the API to be directly accessible, your R2 bucket needs to be configured for public access.

1.  Go to your Cloudflare Dashboard -> R2 -> Your Bucket -> Settings.
2.  Find the "Public access" section.
3.  Ensure "Allow Access" is enabled for your bucket URL (e.g., `pub-XXXX.r2.dev`).
4.  Alternatively, connect a custom domain to your R2 bucket and enable public access for that domain.

If your bucket remains private, you will need to modify the backend to generate pre-signed URLs for accessing the files instead of returning the direct public URL.

## API Usage

-   **POST `/upload/audio`**
    -   **Request Type:** `multipart/form-data`
    -   **Form Fields:** Include fields matching the `AudioMetadataForm` model (e.g., `participant_code`, `prompt_id`, `dialect`, etc.).
    -   **File Part:** Include the audio file under the field name `file`.
    -   **Response:** `UploadResponse` model containing success message, R2 URL, participant/prompt IDs, and MongoDB document ID.

See the interactive API documentation at `/docs` when running locally or deployed.
