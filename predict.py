import torch
from model import AntiSpoofNet
from PIL import Image
import torchvision.transforms as transforms

def predict_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    model = AntiSpoofNet()
    model.load_state_dict(torch.load("antispoof_model.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        output = model(image)
        prediction = output.item()

    return "Spoof" if prediction > 0.7 else "Live"

# Example usage
print(predict_image("frames/f1.jpg"))
