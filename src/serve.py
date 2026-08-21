from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

CLOUD_BUCKET = os.environ.get("CLOUD_BUCKET") or os.environ.get("GCS_BUCKET") or os.environ.get("S3_BUCKET", "")
MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu Cloud Storage ve may khi server khoi dong.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # Try S3
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.download_file(CLOUD_BUCKET, MODEL_KEY, MODEL_PATH)
        print("Model da duoc tai xuong tu S3.")
        return
    except Exception:
        pass

    # Try GCS
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(CLOUD_BUCKET)
        blob   = bucket.blob(MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print("Model da duoc tai xuong tu GCS.")
        return
    except Exception:
        pass


if CLOUD_BUCKET or not os.path.exists(MODEL_PATH):
    try:
        download_model()
    except Exception as e:
        print(f"Download model notice: {e}")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=500, detail="Model not loaded")

    pred = int(model.predict([req.features])[0])
    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = label_map.get(pred, "khong_xac_dinh")

    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

