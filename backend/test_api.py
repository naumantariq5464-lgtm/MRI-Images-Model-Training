import requests, os

test_img_dir = r"D:\Model Train\dataset\test\Pituitary_Tumor"
img_file = os.path.join(test_img_dir, os.listdir(test_img_dir)[0])
print(f"Testing with: {img_file}")

with open(img_file, "rb") as f:
    r = requests.post("http://localhost:8000/predict", files={"file": ("test.jpg", f, "image/jpeg")})

data = r.json()
print("Prediction:", data["prediction"])
print("Confidence:", data["confidence_percent"])
print("Raw Score: ", data["raw_score"])
print("Grad-CAM Base64 Length:", len(data["gradcam_image"]), "chars")
print("API TEST PASSED!")
