import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import streamlit as st

# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loading and Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./dataset", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./dataset", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Neural Network Model (Multilayer Perceptron)
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training and Evaluation
model = MNISTNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

# Save the trained model
torch.save(model.state_dict(), "mnist_model.pt")  # Replace with your desired path

# Streamlit App

# Load the trained model
model = MNISTNet().to(device)
model.load_state_dict(torch.load("mnist_model.pt"))

def predict_digit(image):
    # Preprocess image (assuming a PIL image is uploaded)
    image = transforms.ToTensor()(image)
    image.unsqueeze_(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

st.title("MNIST Digit Recognizer")

uploaded_file = st.file_uploader("Upload an image of a handwritten digit:", type="jpeg, png")

if uploaded_file is not None:
    image = st.image(uploaded_file, width=280)  # Display uploaded image

    if st.button("Predict Digit"):
        prediction = predict_digit(image)
        st.success(f"Predicted Digit: {prediction}")
