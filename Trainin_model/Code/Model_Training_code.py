# ===================== IMPORTING ALL REQUIRED LIBRARIES =====================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tkinter import Tk, filedialog  


# ===================== DEFINING PATHS FOR DATA AND MODEL =====================
train_dir = ("C:\\Users\\Prathamesh\\Downloads\\PlantVillage\\PlantVillage\\train")
val_dir = ("C:\\Users\\Prathamesh\\Downloads\\PlantVillage\\PlantVillage\\val")
model_path = 'plant_disease_model.pth'

# ===================== IMAGE PREPROCESSING PIPELINE =====================
# Resize, convert to tensor, and normalize images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ===================== LOADING TRAIN AND VALIDATION DATASETS =====================
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
val_data = datasets.ImageFolder(root=val_dir, transform=transform)

# Create data loaders for batching and shuffling
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Extract class names and count number of classes
class_names = train_data.classes
num_classes = len(class_names)

print(f"\n Loaded {len(train_data)} Training Images and {len(val_data)} Validation Images.")
print(f"All Classes Present In PlantVillage Dataset : {class_names}")


# ===================== DEVICE CONFIGURATION (GPU OR CPU) =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== FUNCTION TO LOAD A SAVED MODEL =====================
def load_model(load_path=model_path):
    # If model file does not exist, return None
    if not os.path.exists(load_path):
        print(f" Model File Not Found At {load_path}")
        return None, None
    
    # Load checkpoint (saved model state + class names)
    checkpoint = torch.load(load_path, map_location=device)
    
    # Rebuild model architecture and load saved weights
    loaded_model = models.resnet18(pretrained=False)
    loaded_model.fc = nn.Linear(loaded_model.fc.in_features, len(checkpoint['class_names']))
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    
    print("\n Your Trained Model Loaded Successfully!")
    print("\n Model Architecture:")
    print(loaded_model)
    return loaded_model.to(device), checkpoint['class_names']

# ===================== FUNCTION TO SAVE TRAINED MODEL =====================
def save_model(model, class_names, save_path=model_path):
    # Save model weights, class names, and input size
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'input_size': 128
    }, save_path)
    print(f"\n Model Saved To {save_path}")


# ===================== CHECK IF MODEL ALREADY EXISTS =====================
# If model file is found → load it, else train a new one
if os.path.exists(model_path):
    model, class_names = load_model(model_path)
    skip_training = True
else:
    # Create new ResNet18 model with pretrained ImageNet weights
    model = models.resnet18(pretrained=True)
    # Replace final layer for PlantVillage number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    skip_training = False

    print("\n Model Architecture:")
    print(model)


# ===================== DEFINE LOSS FUNCTION AND OPTIMIZER =====================
criterion = nn.CrossEntropyLoss()                     # Multi-class loss
optimizer = optim.Adam(model.parameters(), lr=0.0001) # Adam optimizer with small learning rate


# ===================== TRAINING CONFIGURATIONS =====================
epochs = 5
train_losses = []     # To store training loss per epoch
val_accuracies = []   # To store validation accuracy per epoch

# ===================== TRAINING LOOP =====================
if not skip_training:
    print("\n Training Model Using Device's CPU. It will take some time...\n")
    for epoch in range(epochs):
        model.train()         # Set model to training mode
        running_loss = 0.0    # Track loss per epoch
        
        # ---- Training on batches ----
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()            # Reset gradients
            outputs = model(inputs)          # Forward pass
            loss = criterion(outputs, labels)# Compute loss
            loss.backward()                  # Backpropagation
            optimizer.step()                 # Update weights
            running_loss += loss.item()      # Accumulate loss

        # Average training loss for the epoch
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # ---- Validation after each epoch ----
        model.eval()          # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad(): # Disable gradient computation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)  # Get highest probability class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        val_accuracies.append(accuracy)

        # Print loss and accuracy for this epoch
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Accuracy: {accuracy:.4f}")

    # Save model after training completes
    save_model(model, class_names)
else:
    print("\n Skipping Training because We Already Trained a Model")


# ===================== PLOT VALIDATION ACCURACY OVER EPOCHS =====================
if val_accuracies:
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title("Validation Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


# ===================== CLASSIFICATION REPORT ON VALIDATION SET =====================
print("\n Classification Report on Validation Set:")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
print(classification_report(all_labels, all_preds, target_names=class_names))


# ===================== FUNCTION TO PREDICT SINGLE IMAGE =====================
def predict_image(image_path, model=model, class_names=class_names):
    model.eval()  # Set model to evaluation mode

    # Load and show the image
    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)
    plt.axis('off')
    plt.title("Uploaded Image For Making Predictions")
    plt.show()

    # Apply same preprocessing as training images
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        pred_class = class_names[predicted.item()]
        confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()
        print(f"\n Predicted Class Of Your Uploaded Image: {pred_class}")
        print(f"\n Our Model Is {confidence:.2%} Confident In Its Prediction.")


# ===================== FUNCTION TO UPLOAD IMAGE AND RUN PREDICTION =====================
def upload_and_predict():
    print("\n Please Select an Image From Your Device... If Selected , Kindly Wait.. Our Model Is Making Prediction!")
    root = Tk()
    root.withdraw() 
    root.wm_attributes('-topmost', 1)  

    # Open file dialog for image selection
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    root.destroy()  
    
    # Run prediction if file is selected
    if file_path:
        try:
            predict_image(file_path)
        except Exception as e:
            print(f" Error Processing Image: {e}")
    else:
        print(" No Image Selected.")


# ===================== MAIN EXECUTION: ASK USER TO UPLOAD IMAGE =====================
if input("\n Would You Like to Upload an Plant Image for Prediction (y/n): ").lower() == 'y':
    upload_and_predict()