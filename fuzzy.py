from simpful import *
from helper  import areaof

import pandas             as pd
import numpy              as np
import os
import re
import glob

# Perform fuzzy inference to determine the risk level of a detected object.
def fuzzy_infer(matrix: np.ndarray, prop: float) -> int:
    FS = FuzzySystem()

    # Calculate the density by working out the l2 norm of the matrix slice.
    density = np.average(matrix.flatten())
    
    # Define density and its fuzzy sets.
    D_1 = FuzzySet(function=Trapezoidal_MF(a=0,b=0,c=0.003,d=0.04), term="low")
    D_2 = FuzzySet(function=Trapezoidal_MF(a=0.003,b=0.006,c=0.015,d=0.02), term="medium")
    D_3 = FuzzySet(function=Trapezoidal_MF(a=0.01,b=0.02,c=0.03,d=0.03), term="high")
    FS.add_linguistic_variable("Density", LinguisticVariable([D_1, D_2, D_3], concept="Path density", universe_of_discourse=[0,0.03]))
    
    # Define object size relative to humans in scene and fuzzy sets.
    S_1 = FuzzySet(function=Trapezoidal_MF(a=0,b=0,c=0.09,d=0.1), term="small")
    S_2 = FuzzySet(function=Triangular_MF(a=0.08,b=0.1,c=0.15), term="medium")
    S_3 = FuzzySet(function=Trapezoidal_MF(a=0.1,b=0.15,c=1,d=1), term="large")
    FS.add_linguistic_variable("Size", LinguisticVariable([S_1, S_2, S_3], concept="Object size", universe_of_discourse=[0,1]))
    
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
    FS.set_variable("Size", prop)

    # Perform inference.
    risk = FS.Mamdani_inference(["Risk"])
    result = risk["Risk"]
    return round(result)
