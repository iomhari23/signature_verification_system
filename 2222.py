import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2


MODEL_PATH = "siamese_signature_model.pth"
THRESHOLD = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(pretrained=False)
        base.fc = nn.Identity()
        self.base = base

    def forward_once(self, x):
        return self.base(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded successfully")

def preprocess(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = cv2.resize(img, (224, 224))
    img = transform(img)
    img = img.unsqueeze(0)

    return img

def verify(img1_path, img2_path, threshold=THRESHOLD):
    img1 = preprocess(img1_path).to(device)
    img2 = preprocess(img2_path).to(device)

    with torch.no_grad():
        out1, out2 = model(img1, img2)
        out1 = nn.functional.normalize(out1, p=2, dim=1)
        out2 = nn.functional.normalize(out2, p=2, dim=1)

        dist = nn.functional.pairwise_distance(out1, out2)

    distance = dist.item()

    print(f"\nDistance: {distance:.4f}")

    if distance < threshold:
        print("Result: Genuine Signature (MATCH)")
    else:
        print("Result: Forged Signature (NOT MATCH)")

    return distance

if __name__ == "__main__":
    img1 = r"C:\dataset\proof\train\om\signature (3).png"
    img2 = r"C:\dataset\sign\test\049_forg\01_0114049.PNG"

    verify(img1, img2)
