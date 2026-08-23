import os

base = r"D:\realtime-opencv-detection\runs\detect"
folders = ["helmet_train", "helmet_train-3"]  # adjust based on actual names above

for folder in folders:
    path = os.path.join(base, folder)
    if os.path.exists(path):
        print(f"{folder}: {os.listdir(path)}")
        weights_path = os.path.join(path, "weights")
        if os.path.exists(weights_path):
            print(f"   weights/: {os.listdir(weights_path)}")
    else:
        print(f"{folder}: NOT FOUND")