import streamlit as st
import pandas as pd
import os
import cv2
from PIL import Image
import torch
from datetime import datetime
import shutil
from ultralytics import YOLO
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'license_plate_detector.pt'
yolo_model = YOLO(model_path)
CSV_PATH = 'license_plate_database.csv'
IMAGES_DIR = 'car_images/'
characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)   # Output: 64 x 28 x 28
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # Output: 128 x 14 x 14
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) # Output: 256 x 7 x 7

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)  # Adjusted input size based on pooling
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 36)  # 36 classes (0-9 and A-Z)

        # Activation function
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.relu(self.conv1(x))  # [64, 28, 28]
        x = self.pool(x)              # [64, 14, 14]

        x = self.relu(self.conv2(x))  # [128, 14, 14]
        x = self.pool(x)              # [128, 7, 7]

        x = self.relu(self.conv3(x))  # [256, 7, 7]
        x = self.pool(x)              # [256, 3, 3]

        x = x.view(x.size(0), -1)     # [256*3*3 = 2304]

        x = self.relu(self.fc1(x))    # [1024]
        x = self.dropout(x)            # Apply dropout

        x = self.relu(self.fc2(x))    # [512]
        x = self.dropout(x)            # Apply dropout

        x = self.fc3(x)                # [36]

        return x


def crop_image(image, zoom_factor=0.05):

    results = yolo_model(image)
    cropped_plates = []
    
    for result in results:
        boxes = result.boxes
    
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0].cpu().numpy())
            label = yolo_model.model.names[class_id]
    
            if label.lower() == 'license_plate':
                width = x2 - x1
                height = y2 - y1
                
                new_x1 = max(0, x1 + int(zoom_factor * width))
                new_y1 = max(0, y1 + int(zoom_factor * height))
                new_x2 = min(image.shape[1], x2 - int(zoom_factor * width))
                new_y2 = min(image.shape[0], y2 - int(zoom_factor * height))
                
                plate_img = image[new_y1:new_y2, new_x1:new_x2]
                cropped_plates.append(plate_img)
    
    return cropped_plates

@st.cache
def load_data(csv_path):
    return pd.read_csv(csv_path)

def save_to_csv(csv_path, license_plate, image_label, timestamp):
    df = pd.read_csv(csv_path)
    new_entry = pd.DataFrame([[license_plate, image_label, timestamp]], columns=["License Plate", "Image Label", "Timestamp"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(csv_path, index=False)

def display_car_image(image_file):
    image_path = os.path.join(IMAGES_DIR, image_file)
    if os.path.exists(image_path):
        img = Image.open(image_path)
        st.image(img, caption=f"Car Image: {image_file}", use_column_width=True)
    else:
        st.error(f"Image {image_file} not found!")

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')  

    transform = transforms.Compose([
        transforms.Resize((28, 28)),  
        transforms.ToTensor(),        
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    
    return transform(image).unsqueeze(0)  

def detect_license_plate(image, save_output=False, output_path='output.jpg'):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo_model(image_rgb)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            label = yolo_model.model.names[class_id]
            if label.lower() == 'license_plate':
                if save_output:
                    cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    output_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, output_image)
                return image_rgb, (int(x1), int(y1), int(x2), int(y2))  
    return None, None

def find_contours(dimensions, img):

    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    x_cntr_list = []
    img_res = []
    
    img_height = img.shape[0]
    bottom_threshold = int(img_height * 0.7)  

    for cntr in cntrs:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        if lower_width < intWidth < upper_width and lower_height < intHeight < upper_height:
            x_cntr_list.append((intX, intY, intWidth, intHeight))  
            
            char_copy = np.zeros((44, 24))
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            char = cv2.subtract(255, char)

            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)  

    top_row = []
    bottom_row = []
    
    for i, (intX, intY, intWidth, intHeight) in enumerate(x_cntr_list):
        bottom_of_char = intY + intHeight
        
        if bottom_of_char > bottom_threshold:
            bottom_row.append((intX, i))
        else:
            top_row.append((intX, i))

    if len(top_row) == 0:
        sorted_indices = sorted(bottom_row, key=lambda x: x[0])  
    else:
        sorted_top = sorted(top_row, key=lambda x: x[0])
        sorted_bottom = sorted(bottom_row, key=lambda x: x[0])
        sorted_indices = sorted_top + sorted_bottom  

    img_res_copy = [img_res[idx[1]] for idx in sorted_indices]  
    img_res = np.array(img_res_copy)

    img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)  
    for idx in sorted_indices:
        intX, intY, intWidth, intHeight = x_cntr_list[idx[1]]
        cv2.rectangle(img_copy, (intX, intY), (intX + intWidth, intY + intHeight), (0, 255, 0), 2)  # Green rectangles

    img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_copy_rgb)
    plt.title('Sorted Contours (Bounding Boxes)')
    plt.axis('off')  
    plt.show()

    return img_res

def segment_characters(image):
    img_lp = cv2.resize(image, (150, 90))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))
    border_size = 10
    img_binary_lp = cv2.copyMakeBorder(img_binary_lp, border_size, border_size, border_size, border_size,
                                       cv2.BORDER_CONSTANT, value=[255, 255, 255])
    dimensions = [img_binary_lp.shape[0] // 6, 2 * img_binary_lp.shape[0] // 2.5, img_binary_lp.shape[1] // 10, 2 * img_binary_lp.shape[1] // 3]
    return find_contours(dimensions, img_binary_lp)

def predict_license_plate(img_path):
    model = torch.load('cnn.pth')
    model.eval()
    img = cv2.imread(img_path)
    detected_lp, coords = detect_license_plate(img, save_output=False)
    if detected_lp is None:
        return None
    
    cropped_lp = crop_image(detected_lp, zoom_factor=0)
    
    segments = segment_characters(cropped_lp[0])
    lp_number = ''
    for char_image in segments:
        char_image_pil = Image.fromarray(char_image)
        img_tensor = preprocess_image(char_image_pil)

        with torch.no_grad():
            output_tensor = model(img_tensor)
            predicted_class = output_tensor.argmax(dim=1).item()
            lp_number += characters[predicted_class]
    return lp_number

st.title("License Plate Recognition and Search")
st.write("Search by license plate number or upload a new image for recognition.")

df = load_data(CSV_PATH)

license_plate_input = st.text_input("Enter the License Plate Number:")

if license_plate_input:
    result = df[df['License Plate'] == license_plate_input]

    if not result.empty:

        st.success(f"License Plate {license_plate_input} found!")
        for index, row in result.iterrows():
            st.write(f"**Timestamp**: {row['Timestamp']}")
            display_car_image(row['Image Label'])
    else:
        st.error(f"No record found for License Plate {license_plate_input}.")

st.header("Upload a New Image for License Plate Recognition")
uploaded_file = st.file_uploader("Upload an image of the car", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:

    file_path = os.path.join(IMAGES_DIR, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(file_path, caption='Uploaded Image', use_column_width=True)

    predicted_lp = predict_license_plate(file_path)

    if predicted_lp is not None:
        st.success(f"License Plate Detected: {predicted_lp}")

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        save_to_csv(CSV_PATH, predicted_lp, uploaded_file.name, timestamp)

        st.write(f"**Timestamp**: {timestamp}")
        display_car_image(uploaded_file.name)
    else:
        st.error("No license plate detected. The image has been removed.")
        os.remove(file_path)
