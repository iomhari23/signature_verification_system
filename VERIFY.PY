import streamlit as st
import torch

from main import (
    SiameseSignatureModel,
    load_signature_image,
    preprocess_signature,
    extract_topological_features,
    DEVICE,
)

@st.cache_resource
def load_model():
    model = SiameseSignatureModel(
        pretrained_weights_path="C:\\dataset\\sign\\signet.pth"
    ).to(DEVICE)

    checkpoint = torch.load("siamese_best_loss.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


model = load_model()

st.title("✍Signature Verification System")

img1 = st.file_uploader("Upload Signature 1", type=["png", "jpg"])
img2 = st.file_uploader("Upload Signature 2", type=["png", "jpg"])


def prepare(file):
    img = load_signature_image(file)
    processed = preprocess_signature(img)

    img_tensor = torch.tensor(processed, dtype=torch.float32).unsqueeze(0)
    topo_tensor = extract_topological_features(file).unsqueeze(0)

    return img_tensor.to(DEVICE), topo_tensor.to(DEVICE)


if img1 and img2:
    img1_tensor, topo1 = prepare(img1)
    img2_tensor, topo2 = prepare(img2)

    with torch.no_grad():
        emb1, emb2, logit = model(img1_tensor, img2_tensor, topo1, topo2)

    confidence = torch.sigmoid(logit).item()
    similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()

    verdict = " Forged" if confidence >= 0.5 else " Genuine"

    st.subheader("Result")
    st.write(f"**Verdict:** {verdict}")
    st.write(f"**Confidence:** {confidence:.4f}")
    st.write(f"**Similarity:** {similarity:.4f}")
