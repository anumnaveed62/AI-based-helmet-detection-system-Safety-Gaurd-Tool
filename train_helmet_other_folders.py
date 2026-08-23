import os
base = "runs/detect"
for folder in ["helmet_train", "helmet_train2", "helmet_train3"]:
    path = os.path.join(base, folder)
    if os.path.exists(path):
        print(f"{folder}: {os.listdir(path)}")
    else:
        print(f"{folder}: NOT FOUND")