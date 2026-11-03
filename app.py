
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import torch.nn.functional as F

from PIL import Image


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="FreshHarvest | Fruit Inspection",
    page_icon="🍎",
    layout="wide"
)


# ==========================================
# 2. HEADER
# ==========================================

st.title("🍎 FreshHarvest")

st.markdown(
    "### AI-Powered Fruit Freshness Inspection"
)

st.write(
    "Upload a fruit image to identify its type and freshness condition."
)

st.divider()


# ==========================================
# 3. DEVICE CONFIGURATION
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 4. LOAD TRAINED RESNET50 MODEL
# ==========================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        "models/freshharvest_resnet50.pth",
        map_location=device,
        weights_only=True
    )

    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            model.fc.in_features,
            checkpoint["num_classes"]
        )
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    class_names = checkpoint["class_names"]

    return model, class_names


try:

    model, class_names = load_model()

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# ==========================================
# 5. IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# 6. FORMAT PREDICTION
# ==========================================

def format_prediction(class_name):

    if class_name.startswith("S_"):

        fruit_name = class_name[2:]
        freshness = "Spoiled"

    elif class_name.startswith("F_"):

        fruit_name = class_name[2:]
        freshness = "Fresh"

    else:

        fruit_name = class_name
        freshness = "Unknown"

    return fruit_name, freshness


# ==========================================
# 7. IMAGE UPLOAD
# ==========================================

st.subheader("📸 Upload Fruit Image")

uploaded_file = st.file_uploader(
    "Drag and drop your fruit image here",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear fruit image."
)


# ==========================================
# 8. IMAGE PREVIEW AND PREDICTION
# ==========================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2, gap="large")

        # ----------------------------------
        # Uploaded Image
        # ----------------------------------

        with col1:

            st.subheader("Uploaded Image")

            st.image(
                image,
                caption="Selected Fruit Image",
                use_container_width=True
            )

        # ----------------------------------
        # Prediction
        # ----------------------------------

        with col2:

            st.subheader("Prediction")

            inspect_button = st.button(
                "🔍 Inspect Fruit",
                type="primary",
                use_container_width=True
            )

            if inspect_button:

                input_tensor = transform(image).unsqueeze(0)

                input_tensor = input_tensor.to(device)

                # Model inference
                with torch.inference_mode():

                    outputs = model(input_tensor)

                    probabilities = F.softmax(
                        outputs,
                        dim=1
                    )

                    confidence, predicted = torch.max(
                        probabilities,
                        dim=1
                    )

                # Get predicted class
                predicted_class = class_names[
                    predicted.item()
                ]

                fruit_name, freshness = format_prediction(
                    predicted_class
                )

                confidence_score = confidence.item() * 100

                # ----------------------------------
                # DISPLAY ACTUAL PREDICTION
                # ----------------------------------

                st.divider()

                st.markdown(
                    f"<h2 style='text-align:center'>{fruit_name}</h2>",
                    unsafe_allow_html=True
                )

                if freshness == "Fresh":

                    st.success(
                        "🍏 FRESH"
                    )

                elif freshness == "Spoiled":

                    st.error(
                        "⚠️ SPOILED"
                    )

                else:

                    st.warning(
                        "❔ UNKNOWN"
                    )

                st.metric(
                    label="Prediction Confidence",
                    value=f"{confidence_score:.2f}%"
                )

                if confidence_score < 60:

                    st.warning(
                        "The model is uncertain about this prediction. "
                        "Consider uploading a clearer image."
                    )

                st.divider()

    except Exception as e:

        st.error(
            f"Unable to process image: {e}"
        )


else:

    st.info(
        "Upload a fruit image to start freshness inspection."
    )


# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.caption(
    "FreshHarvest | CNN-Based Fruit Freshness Inspection"
)

st.caption(
    "Powered by PyTorch, ResNet50 and Streamlit"
)
