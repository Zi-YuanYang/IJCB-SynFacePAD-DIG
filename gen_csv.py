import csv
import pandas as pd
import os
from tqdm import tqdm

root_path = "../dataset/MSU-SFD"

# folders = ["/BonaFide", "/attack", "/PAs/PrintAttack", "/PAs/Samsung_ReplayAttack", "/PAs/Webcam_ReplayAttack"]
folders = ["/BonaFide", "/attack"]

with open("MSU_SFD.csv", "w") as csvfile:
    writer = csv.writer(csvfile, lineterminator='\n')
    for folder in folders:
        label = folder.split("/")[-1]
        print("Generating.. {}".format(folder.split("/")[-1]))
        path = root_path + folder
        img_names = os.listdir(path)
        for img_name in tqdm(img_names):
            filepath = os.path.join(path, img_name)
            writer.writerow([filepath, label.lower()])