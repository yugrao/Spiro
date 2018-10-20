import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pickle as p

os.chdir("PARKINSON_HW/hw_dataset")

for datatype in ["control", "parkinson", "../new_dataset/parkinson"]:
    for datafile in os.listdir(datatype):
        if datafile=="train_model":
            continue
        df = pd.read_csv(datatype+"/"+datafile, sep=";", names=["X", "Y", "Z", "Pressure", "GripAngle", "Timestamp", "Test ID"])
        df = df.loc[df["Test ID"] == 0][["X","Y"]]
        if not len(df):
            continue
        center_val = df.iloc[0]

        df["X"] = [value-center_val[0] for value in df["X"]]
        df["Y"] = [-value+center_val[1] for value in df["Y"]]
        plt.plot(df["X"], df["Y"], "ko-", linewidth=5, markersize=0)
        plt.axis([-300, 300, 300, -300])
        #plt.show()
        plt.axis("off")
        fig = plt.gcf()
        fig.set_size_inches(5,5)
        fig.savefig("../../pictures/"+datafile.split(".")[0]+".png", dpi=25)
        plt.clf()
os.chdir("../..")

picarray = []
for pic in os.listdir("pictures"):
    a = np.asarray(Image.open("pictures/"+pic))
    a = [[list(map(lambda rgbx: (rgbx[0] * 299.0/1000 + rgbx[1] * 587.0/1000 + rgbx[2] * 114.0/1000)/255, row)) for row in a]]
    a = np.array(a)
    diseased = False if pic[0]=="C" else True
    picarray.append([a, diseased])
    print(pic, a.shape)
p.dump( picarray, open( "train_model/picarray.pickle", "wb" ) )
