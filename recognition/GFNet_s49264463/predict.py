import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataset import get_transforms, group_by_patient
from modules import ConvNetwork, GFNet

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gfnet", choices=["gfnet", "cnn"])
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data-root", default="/content/AD_NC")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--threshold", type=float, default=0.79)
ap.add_argument("--out", default="predictions.png")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = (GFNet() if args.model == "gfnet" else ConvNetwork()).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device))
model.eval()

test = group_by_patient(args.data_root, "test")
items = [item for pid in sorted(test) for item in test[pid]]
step = max(1, len(items) // args.n)
sample = items[::step][:args.n]

transform = get_transforms(train=False)
names = {0: "NC", 1: "AD"}
cols = 4
rows = (len(sample) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows))
n_correct = n_review = 0

with torch.no_grad():
    for ax, (path, label) in zip(axes.ravel(), sample):
        image = Image.open(path)
        probs = torch.softmax(model(transform(image).unsqueeze(0).to(device)), dim=1)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])
        review = conf < args.threshold
        n_correct += pred == label
        n_review += review
        flag = "SEND TO HUMAN" if review else "auto"
        print(f"{path.split('/')[-1]}: true {names[label]}, predicted {names[pred]}, "
              f"confidence {conf:.2f} -> {flag}")
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        ax.set_title(f"true {names[label]} / pred {names[pred]}\nconf {conf:.2f} ({flag})",
                     color="green" if pred == label else "red", fontsize=8)

for ax in axes.ravel()[len(sample):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"{n_correct}/{len(sample)} correct on this sample; {n_review} flagged for human review; "
      f"figure saved to {args.out}")
