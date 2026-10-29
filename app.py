import os
import json

import torch
from torch import nn
from torchvision import models, transforms

import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/resnet18_best.pth"
CLASS_PATH = "class_to_idx.json"

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Image Classification",
    page_icon="🦋",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("Image Classifier")

st.write(
    "Upload an image and the trained ResNet18 model "
    "will predict its class."
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "Model file not found!"
    )

    st.info(
        "Please run train.py first."
    )

    st.stop()


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(CLASS_PATH, "r") as f:

    class_to_idx = json.load(f)


# Reverse mapping

idx_to_class = {
    int(index): class_name
    for class_name, index in class_to_idx.items()
}


NUM_CLASSES = len(idx_to_class)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Create ResNet18
    model = models.resnet18(weights=None)

    # Replace final layer
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    # Load checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Handle checkpoint format
    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(DEVICE)

    model.eval()

    return model


model = load_model()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Read image
    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Display image
    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        use_container_width=True
    )


    # Transform image
    image_tensor = transform(
        image
    ).unsqueeze(0)


    image_tensor = image_tensor.to(
        DEVICE
    )


    # Prediction
    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]


    # Get top prediction
    confidence, predicted_index = torch.max(
        probabilities,
        dim=0
    )


    predicted_class = idx_to_class[
        predicted_index.item()
    ]


    confidence_percentage = (
        confidence.item() * 100
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("Prediction")

    st.success(
        f"Prediction: {predicted_class.upper()}"
    )

    st.metric(
        "Confidence",
        f"{confidence_percentage:.2f}%"
    )


    # ========================================================
    # ALL CLASS PROBABILITIES
    # ========================================================

    st.subheader(
        "Class Probabilities"
    )


    probabilities_dict = {}

    for index, probability in enumerate(
        probabilities
    ):

        class_name = idx_to_class[index]

        probabilities_dict[
            class_name
        ] = probability.item() * 100


    # Sort descending
    probabilities_dict = dict(
        sorted(
            probabilities_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )


    # Display probabilities
    for class_name, probability in probabilities_dict.items():

        st.write(
            f"**{class_name}**: "
            f"{probability:.2f}%"
        )

        st.progress(
            int(probability)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Model: ResNet18 | "
    f"Device: {DEVICE}"
)
