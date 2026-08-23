import os

base = r"D:\realtime-opencv-detection\runs\detect"
if os.path.exists(base):
    print("Folders found in runs/detect:")
    for f in os.listdir(base):
        print(" -", f)
else:
    print("Base path does not exist:", base)