import os
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

class SignatureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.genuine = {}
        self.forged = {}

        for folder in os.listdir(root_dir):
            path = os.path.join(root_dir, folder)

            if "_forg" in folder:
                person_id = folder.split("_")[0]
                self.forged[person_id] = [
                    os.path.join(path, img) for img in os.listdir(path)
                ]
            else:
                person_id = folder
                self.genuine[person_id] = [
                    os.path.join(path, img) for img in os.listdir(path)
                ]

        self.person_ids = [
            pid for pid in self.genuine.keys() if pid in self.forged
        ]

    def __len__(self):
        return 2000

    def __getitem__(self, index):
        person = random.choice(self.person_ids)
        choice = random.random()

        if choice < 0.33:
            img1_path, img2_path = random.sample(self.genuine[person], 2)
            label = 1
        elif choice < 0.66:
            img1_path = random.choice(self.genuine[person])
            img2_path = random.choice(self.forged[person])
            label = 0
        else:
            other_person = random.choice(self.person_ids)
            while other_person == person:
                other_person = random.choice(self.person_ids)
            img1_path = random.choice(self.genuine[person])
            img2_path = random.choice(self.genuine[other_person])
            label = 0

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(pretrained=True)
        base.fc = nn.Identity()
        self.base = base

    def forward_once(self, x):
        return self.base(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        distance = nn.functional.pairwise_distance(out1, out2)
        loss = (label * distance**2 +
               (1 - label) * torch.clamp(self.margin - distance, min=0.0)**2)
        return loss.mean()
    #def trueScore(self,out1,out2):
        if distance < self.margin:
            return 0
        else:
            pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = SignatureDataset(r"C:\dataset\sign\train", transform=transform)
test_dataset = SignatureDataset(r"C:\dataset\sign\test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for i, (img1, img2, label) in enumerate(train_loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(f"Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} Completed | Avg Loss: {running_loss/len(train_loader):.4f}\n")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for img1, img2, label in test_loader:
        img1, img2 = img1.to(device), img2.to(device)
        out1, out2 = model(img1, img2)
        dist = nn.functional.pairwise_distance(out1, out2)
        pred = (dist < 0.5).float()
        correct += (pred == label.to(device)).sum().item()
        total += label.size(0)

print("Test Accuracy:", correct / total)

torch.save(model.state_dict(), "siamese_signature_model.pth")
