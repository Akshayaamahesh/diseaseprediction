import os
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Transforms
transformer = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Hyperparameters
num_epochs = 2
num_classes = 2
batch_size_train = 64
batch_size_test = 32
learning_rate = 0.001

# Update paths for training and testing directories on your local system
train_path = "D:/chest_xray/train"  # Adjust this path to your actual training data directory
test_path = "D:/chest_xray/test"    # Adjust this path to your actual testing data directory

# Calculate the size of training and testing images
train_count = len(glob.glob(os.path.join(train_path, '**/*.jpeg')))
test_count = len(glob.glob(os.path.join(test_path, '**/*.jpeg')))
print(f"Training images: {train_count}, Testing images: {test_count}")

class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None, is_valid_file=None):
        self.root = root
        self.transform = transform
        self.is_valid_file = is_valid_file

        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        classes = [d.name for d in pathlib.Path(dir).iterdir() if d.is_dir() and d.name != '__MACOSX']
        classes = sorted(classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(self, directory, class_to_idx):
        images = []
        directory = os.path.expanduser(directory)
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(directory, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if self.is_valid_file(path):
                        item = (path, class_to_idx[target])
                        images.append(item)
        return images

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def loader(self, path):
        return Image.open(path).convert('RGB')

def is_valid_image(filename):
    # Exclude macOS-specific files
    return not ("__MACOSX" in filename or ".DS_Store" in filename)

# Train dataset
train_dataset = CustomImageFolder(train_path, transform=transformer, is_valid_file=is_valid_image)
print(type(train_dataset))

# Test dataset
test_dataset = CustomImageFolder(test_path, transform=transformer, is_valid_file=is_valid_image)
print(type(test_dataset))

# Train dataset
train_dataset = CustomImageFolder(train_path, transform=transformer, is_valid_file=is_valid_image)

# Convert labels 2 and 3 to 1 in the training dataset
train_dataset.targets = [label if label < 2 else 1 for label in train_dataset.targets]

#Test dataset
test_dataset = CustomImageFolder(test_path, transform=transformer, is_valid_file=is_valid_image)

# Convert labels 2 and 3 to 1 in the test dataset
test_dataset.targets = [label if label < 2 else 1 for label in test_dataset.targets]

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)

# Test loader
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=True)

# Categories
root = pathlib.Path(train_path)
classes = [j.name for j in root.iterdir() if j.is_dir() and j.name != '__MACOSX']
classes = sorted(classes)
print(classes)




# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 82 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model_xray_new.ckpt')


# Define the function to predict pneumonia
def predict_pneumonia(model, image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Apply transformations
    image = transformer(image)

    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Get the predicted class (0: Normal, 1: Pneumonia)
    _, predicted_class = torch.max(output.data, 1)

    if predicted_class.item() == 0:
        return "Normal"
    else:
        return "Pneumonia"

# Load the trained model
model_path = 'model_xray_new.ckpt'
model = ConvNet(num_classes)
model.load_state_dict(torch.load(model_path))
model.to(device)

# Example usage
jpeg_file_path = "img.jpg" # Replace with the actual path
prediction = predict_pneumonia(model, jpeg_file_path)
print(f"The model predicts: {prediction}")
