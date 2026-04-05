from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import traceback

# import predict function from inference module 
from Backend.inference_module import predict_one_image 

app = FastAPI(title="Alzheimer's MRI Classifier")

# add CORS middleware to allow requests from frontend (running on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# simple health check endpoint 
@app.get("/")
def root():
    return {"message": "Alzheimer's API ready"}

# main prediction endpoint
# accepts a single uploaded image file from the frontend 
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # validate its an image
        try:
            Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

        # call local inference function 
        model_data = predict_one_image(contents)

        # convert images dict from predict_one_image response to an array for frontend
        images_dict = model_data.get("images", {})
        images_array = [
            images_dict.get("original_png_b64"),
            images_dict.get("vgg_gradcam_png_b64"),
            images_dict.get("eff_gradcam_png_b64"),
        ]

        # return response in JSON frindly format with predicted class, confidence, probabilities, and base64 encoded images for display
        return {
            "class": model_data.get("predicted_class"),
            "confidence": model_data.get("confidence"),
            "probabilities": model_data.get("probabilities"),
            "images": images_array,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("FULL TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

