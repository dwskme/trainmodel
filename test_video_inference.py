import cv2
import torch

from model.unet import UNet
from postprocess import process_prediction
from transforms import apply_filters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "lane_unet.pth"
VIDEO_IN = "test_input.mp4"  # place your test video here
VIDEO_OUT = "output_lanes.mp4"
FPS = 20

# Load model
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Video I/O
cap = cv2.VideoCapture(VIDEO_IN)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, FPS, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (256, 256))
    filt = apply_filters(img)
    tensor = (
        torch.from_numpy(filt.transpose(2, 0, 1) / 255.0)
        .float()
        .unsqueeze(0)
        .to(DEVICE)
    )

    # Inference + post
    with torch.no_grad():
        clean, lines, angle = process_prediction(model(tensor))

    # Resize masks back
    clean_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    lines_bgr = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
    clean_bgr = cv2.resize(clean_bgr, (w, h))
    lines_bgr = cv2.resize(lines_bgr, (w, h))

    # Overlay
    combo = cv2.addWeighted(frame, 0.6, clean_bgr, 0.4, 0)
    combo = cv2.addWeighted(combo, 0.8, lines_bgr, 0.2, 0)
    cv2.putText(
        combo,
        f"Angle: {angle:.1f}°",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    out.write(combo)
    cv2.imshow("Lane Detection", combo)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[✅] Saved inference video to {VIDEO_OUT}")
