import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import PATH_PERSO_TRAIN_NEW, PATH_PERSO_VALID_NEW, PATH_PERSO_TEST_NEW

train_img_dir = PATH_PERSO_TRAIN_NEW
valid_img_dir = PATH_PERSO_VALID_NEW
test_img_dir = PATH_PERSO_TEST_NEW

# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

train_dataset = datasets.ImageFolder(train_img_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

validation_dataset = datasets.ImageFolder(valid_img_dir, transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)


# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(
            32 * 75 * 75, 128
        )  # Adjust based on input size after conv/pool layers
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 75 * 75)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


model = SimpleCNN()

# Define loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        predicted = outputs.squeeze().round()
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = correct_train / total_train
    print(
        f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy}"
    )

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            val_loss += loss.item()
            predicted = outputs.squeeze().round()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    val_accuracy = correct / total
    print(
        f"Validation Loss: {val_loss / len(validation_loader)}, Validation Accuracy: {val_accuracy}"
    )
