from model import FakeImageCNN
from torchvision import transforms
from PIL import Image
import torch
from config import IMAGE_SIZE, MODEL_SAVE_PATH

def predict(image_path, model_path=MODEL_SAVE_PATH):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    model = FakeImageCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    return "REAL" if predicted.item() == 0 else "FAKE"