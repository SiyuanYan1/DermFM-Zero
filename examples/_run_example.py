"""Faithful run of examples/zero-shot-classification.ipynb (GPU)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import torch
from PIL import Image

project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "..", "src")
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

import open_clip
print("open_clip:", open_clip.__file__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:redlessone/DermFM-Zero")
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer("hf-hub:redlessone/DermFM-Zero")

img_path = os.path.join(project_root, "PAT_8_15_820.png")
image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

PAD_CLASSNAMES = [
    "nevus", "basal cell carcinoma", "actinic keratosis",
    "seborrheic keratosis", "squamous cell carcinoma", "melanoma",
]
template = lambda c: f"This is a skin image of {c}"
text = tokenizer([template(c) for c in PAD_CLASSNAMES]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

final_prediction = PAD_CLASSNAMES[torch.argmax(text_probs[0])]
print(f"This images is diagnosised as {final_prediction}.")
print("\nLabel probs:")
for i, label in enumerate(PAD_CLASSNAMES):
    print(f"{label}: {text_probs[:, i].item():.3f}")
