from simpful import *
from helper  import areaof

import pandas             as pd
import numpy              as np
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import os
import re
import glob

pandas = "/Users/marina/introcs/Bachelorarbeit/opencv/results/figure/12/"

def fuzzy_infer(matrix, coords):
    FS = FuzzySystem()

    # Calculate the density by working out the l2 norm of the matrix slice.
    density = np.average(matrix.flatten())

    # Calculate the size of the detected object from the coordinates.
    size = areaof(coords)
    
    # Define linguistic variable density and its trapezoidal membership function.
    # TODO: Redefine density function.
    D_1 = FuzzySet(points=[[0, 1.], [0.015, 1.], [0.08, 0]], term="low")
    D_2 = FuzzySet(points=[[0.01, 0], [0.07, 1.], [0.1, 1.], [0.5, 0]], term="medium")
    D_3 = FuzzySet(points=[[0.4, 0], [0.45, 1.], [0.6, 1.]], term="high")
    FS.add_linguistic_variable("Density", LinguisticVariable([D_1, D_2, D_3], concept="Path density", universe_of_discourse=[0,0.6]))
    
    # Define linguistic variable size and its trapezoidal membership function.
    S_1 = FuzzySet(points=[[0, 1.], [1800, 1.], [2100, 0.]], term="small")
    S_2 = FuzzySet(points=[[1200, 0], [2000, 1.], [2500, 1.], [3300, 0.]], term="medium")
    S_3 = FuzzySet(points=[[3200, 0], [3342, 1.], [4000, 1.]], term="large")
    FS.add_linguistic_variable("Size", LinguisticVariable([S_1, S_2, S_3], concept="Object size", universe_of_discourse=[0,4000]))
    
    # Define output fuzzy set risk and its trapezoidal membership function.
    R_1 = FuzzySet(function=Triangular_MF(a=0,b=0,c=1), term="low")
    R_2 = FuzzySet(function=Triangular_MF(a=0,b=1,c=2), term="medium")
    R_3 = FuzzySet(function=Trapezoidal_MF(a=1,b=2,c=3, d=3), term="high")
    FS.add_linguistic_variable("Risk", LinguisticVariable([R_1, R_2, R_3], concept="Risk level", universe_of_discourse=[0,3]))

    # Fuzzy rules.
    RULE1 = "IF (Density IS low) AND (Size IS small)  THEN (Risk IS low)"
    RULE2 = "IF (Density IS low) AND (Size IS medium)  THEN (Risk IS low)"
    RULE3 = "IF (Density IS low) AND (Size IS large)  THEN (Risk IS medium)"
    RULE4 = "IF (Density IS medium) AND (Size IS small)  THEN (Risk IS medium)"
    RULE5 = "IF (Density IS medium) AND (Size IS medium)  THEN (Risk IS medium)"
    RULE6 = "IF (Density IS medium) AND (Size IS large)  THEN (Risk IS high)"
    RULE7 = "IF (Density IS high) AND (Size IS small)  THEN (Risk IS high)"
    RULE8 = "IF (Density IS high) AND (Size IS medium)  THEN (Risk IS high)"
    RULE9 = "IF (Density IS high) AND (Size IS large)  THEN (Risk IS high)"
    FS.add_rules([RULE1, RULE2, RULE3, RULE4, RULE5, RULE6, RULE7, RULE8, RULE9])

    # Set antecedents values.
    FS.set_variable("Density", density)
    FS.set_variable("Size", size)

    # Perform inference.
    risk = FS.Mamdani_inference(["Risk"])
    result = risk["Risk"]

    #FS.plot_variable("Density")

    return round(result)
    
def save_norms(matrix, frame):
    # Only save non-zero values.
    if not np.all((matrix == 0)):
        filename = pandas + str(frame) + ".csv"
        df = pd.DataFrame(matrix)
        df.fillna(0, inplace=True)

        df.to_csv(filename)
        return df

# Returns the max value, min value, norm, mean, and standard deviation for the last frame in each video.
def get_values():
    keylist = ['video', 'norm', 'max', 'min', 'mean', 'std']
    values_dict = dict.fromkeys(keylist)

    videos = []
    norms = []
    max_values = []
    min_values = []
    means = []
    stds = []

    folder = "/Users/marina/introcs/Bachelorarbeit/opencv/results/figure/"
    subdirectories = [x for x in os.listdir(folder) if x.isdigit()]
    subdirectories.sort()
    
    for subfolder in subdirectories:
        # Get highest frame file in directory.
        files = [float(os.path.basename(os.path.splitext(filename)[0])) for filename in [f for f in glob.glob(os.path.join(os.path.join(folder, subfolder), '*')) if os.path.isfile(f)]]
        filename = os.path.join(folder, subfolder, str(max(files)) + ".csv")

        df = pd.read_csv(filename)
        df = df.drop(['Unnamed: 0'], axis=1)

        norm = round(np.linalg.norm(df), 4)
        max_value = round(max(df.max(axis=1)), 4)
        min_value = round(np.nanmin(df.replace(0,np.nan).values), 4)
        mean = round(df.values.mean(), 4)
        std = round(df.values.std(ddof=1), 4)

        videos.append(subfolder)
        norms.append(norm)
        max_values.append(max_value)
        min_values.append(min_value)
        means.append(mean)
        stds.append(std)
    
    values_dict['video'] = videos
    values_dict['norm'] = norms
    values_dict['max'] = max_values
    values_dict['min'] = min_values
    values_dict['mean'] = means
    values_dict['std'] = stds

    df = pd.DataFrame(data=values_dict)
    df.to_csv("/Users/marina/introcs/Bachelorarbeit/opencv/results/figure/values.csv", sep=';', decimal=',')

    return df


# Make a histogram of values in the dataframe to help with creating membership function for classification.
def make_histogram(df, frame):
    plt.hist(df.to_numpy().flatten())
    plt.title("Frame " + str(frame))
    #plt.show()
    #plt.savefig(os.path.join(pandas, "histogram" + str(frame) + ".png"))
    plt.clf()

def make3dhist(matrix, frame):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.nonzero(matrix)
    
    top = [matrix[i][j] for i,j in np.transpose(np.nonzero(matrix))]

    bottom = np.zeros_like(top)
    width = depth = 0.5

    ax.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax.set_ylim(ymax = 300, ymin = 0)
    ax.set_xlim(xmax = 400, xmin = 0)
    #ax.set_zlim(0, 0.008)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')

    plt.title("Frame " + str(frame))
    #plt.show()
    #plt.savefig(os.path.join(pandas, "histogram3d", str(frame) + ".png"))
    plt.close()

def draw_density():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)

    # Trapezoid small
    x1 = [0.0, 0.04, 0.015, 0.0]
    y1 = [0.0, 0.0, 1, 1]
    ax.add_patch(patches.Polygon(xy=list(zip(x1,y1)), fill=False, color='blue', label="small"))

    # Trapezoid medium
    x2 = [0.01, 0.07, 0.06, 0.02]
    y2 = [0.0, 0.0, 1, 1]
    ax.add_patch(patches.Polygon(xy=list(zip(x2,y2)), fill=False, color='red', label="medium"))

    # Trapezoid high
    x3 = [0.06, 0.09, 0.09, 0.08]
    y3 = [0.0, 0.0, 1, 1]
    ax.add_patch(patches.Polygon(xy=list(zip(x3,y3)), fill=False, color='green', label="large"))

    ax.set_ylim(ymax = 1, ymin = 0)
    ax.set_xlim(xmax = 0.09, xmin = 0)
    ax.xaxis.set_ticks(np.arange(0, 0.09, 0.01))
    plt.legend()

    #plt.show()
    #plt.savefig("/Users/marina/introcs/Bachelorarbeit/opencv/results/pandas/density.png")
    plt.close()

def draw_size():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)

    # Trapezoid small
    x1 = [0.0, 2100, 1800, 0.0]
    y1 = [0.0, 0.0, 1, 1]
    ax.add_patch(patches.Polygon(xy=list(zip(x1,y1)), fill=False, color='blue', label="small"))

    # Trapezoid medium
    x2 = [1200, 3300, 2500, 2000]
    y2 = [0.0, 0.0, 1, 1]
    ax.add_patch(patches.Polygon(xy=list(zip(x2,y2)), fill=False, color='red', label="medium"))

    # Trapezoid high
    x3 = [3200, 3820, 3820, 3342]
    y3 = [0.0, 0.0, 1, 1]
    ax.add_patch(patches.Polygon(xy=list(zip(x3,y3)), fill=False, color='green', label="large"))

    ax.set_ylim(ymax = 1, ymin = 0)
    ax.set_xlim(xmax = 3820, xmin = 0)
    ax.xaxis.set_ticks(np.arange(0, 3820, 300))
    plt.legend()

    #plt.show()
    #plt.savefig("/Users/marina/introcs/Bachelorarbeit/opencv/results/pandas/density.png")
    plt.close()

