import numpy as np
import pandas as pd
import joblib
def predictRuns(s):
    prediction = 0
    ### Your Code Here ###
    with open("linear_regression.joblib","rb") as f:
        reg=joblib.load(f)
    with open("venue_encoder.joblib","rb") as f:
        ve=joblib.load(f)
    with open("team_encoder.joblib","rb") as f:
        te=joblib.load(f)
    tdf=pd.read_csv("inputFile.csv")
    tdf=tdf.drop(columns=["innings",'batsmen',"bowlers"])
    tdf["venue"]=ve.transform(tdf["venue"])
    tdf["batting_team"]=te.transform(tdf["batting_team"])
    tdf["bowling_team"]=te.transform(tdf["bowling_team"])
    tdf=tdf[["venue","batting_team","bowling_team"]]
    predictruns=int(reg.predict(tdf))
    return predictruns


