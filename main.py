import seaborn as sns
from numpy.polynomial.polynomial import polyfit
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from runexplorer import RunExplorer
from runvisualizer import RunVisualizer


filePath = '../data/'
limbs = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip",
         "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Reye", "Leye", "Rear", "Lear"]





def dataFrameBuilder(dataFiles):
    for file in dataFiles:
        if not file[-3:] == 'csv':
            dataFiles.remove(file)
    fileInfo = pd.DataFrame(
        {'file': dataFiles}
    )
    return fileInfo


def fileProcessor(df):
    for index, row in df.iterrows():
        # create new RunExplorer instance based on the current athlete CSV file.
        explorer = RunExplorer(filePath + row['file'])

        # Safe overall quality based on total number of NaN's
        df.at[index, 'quality'] = explorer.getFileQuality()

        #Safe detected start frame
        df.at[index, 'startFrame'] = explorer.startRunOne
        #Safe detected end frame 
        df.at[index, 'endFrame'] = explorer.endRunFour
        #Safe covered distance in pixels
        df.at[index, 'runDistance'] = abs(
            explorer.startXVal - explorer.endXVal)
        
        # Performance metric to predict as discussed in section 4.1
        df.at[index, 'speedPixelsPerFrame'] = explorer.speedPixelsPerFrame

        # Body and leg length as discussed in section 4.2
        df.at[index, 'length'] = explorer.bodyLength
        df.at[index, 'legLength'] = explorer.legLength

        for limb in limbs:
            featureName = limb + '_ver_std'
            limbQuality = limb + '_qual'
            limbQuality1 = limb + '_qual1'
            limbQuality2 = limb + '_qual2'
            limbQuality3 = limb + '_qual3'
            limbQuality4 = limb + '_qual4'

            # Calculate quality scores overall, and per run for this limb.
            # Support for chapter threee figures
            df.at[index, limbQuality] = explorer.getLimbQuality(limb)
            df.at[index, limbQuality1] = explorer.getLimbQuality(
                limb, explorer.startRunOne, explorer.endRunOne)
            df.at[index, limbQuality2] = explorer.getLimbQuality(
                limb, explorer.startRunTwo, explorer.endRunTwo)
            df.at[index, limbQuality3] = explorer.getLimbQuality(
                limb, explorer.startRunThree, explorer.endRunThree)
            df.at[index, limbQuality4] = explorer.getLimbQuality(
                limb, explorer.startRunFour, explorer.endRunFour)

            # Standard deviation of this limb, discussed in section 4.3
            df.at[index, featureName] = explorer.verticalDisplacementDeviation(
                limb)

        #Neck parabola feature, discussed in section 4.4
        df.at[index, 'm1'] = explorer.neckParabola(
            explorer.startRunOne, explorer.endRunOne)
        df.at[index, 'm2'] = explorer.neckParabola(
            explorer.startRunTwo, explorer.endRunTwo)
        df.at[index, 'm3'] = explorer.neckParabola(
            explorer.startRunThree, explorer.endRunThree)
        df.at[index, 'm4'] = explorer.neckParabola(
            explorer.startRunFour, explorer.endRunFour)


        # Acceleration phase detector as discussed in section 4.5
        df.at[index, 'phaseRun1'] = explorer.runPhaseSeperator(
            explorer.startRunOne, explorer.endRunOne)
        df.at[index, 'phaseRun2'] = explorer.runPhaseSeperator(
            explorer.startRunTwo, explorer.endRunTwo)
        df.at[index, 'phaseRun3'] = explorer.runPhaseSeperator(
            explorer.startRunThree, explorer.endRunThree)
        df.at[index, 'phaseRun4'] = explorer.runPhaseSeperator(
            explorer.startRunFour, explorer.endRunFour)


        # Acceleration phase detector as discussed in section 4.5
        df.at[index, 'stepFrequency1'] = explorer.stepFrequency(
            explorer.startRunOne, explorer.endRunOne)/explorer.legLength
        df.at[index, 'stepFrequency2'] = explorer.stepFrequency(
            explorer.startRunTwo, explorer.endRunTwo)/explorer.legLength
        df.at[index, 'stepFrequency3'] = explorer.stepFrequency(
            explorer.startRunThree, explorer.endRunThree)/explorer.legLength
        df.at[index, 'stepFrequency4'] = explorer.stepFrequency(
            explorer.startRunFour, explorer.endRunFour)/explorer.legLength

        #visualizations made for individual athlete files
        visualizer = RunVisualizer(explorer, row['file'])
        visualizer.visualizeRuns()
        visualizer.visualizeParabolaRun()
        visualizer.visualizeAccelerationPhase()
        visualizer.visualizeVerticalDisplacement()


    df = qualityCheck(df)
    df = speedCheck(df)
    df = removeStartAtFrameZero(df)
    df = removeEndAtLastFrame(df)

    df = df.sort_values('speedPixelsPerFrame').reset_index(drop=True)

    return df


def qualityCheck(df):
    print('mean and standard deviation')
    print(df['quality'])
    print(df['quality'].mean())
    print(df['quality'].std())

    newdf = df[
        (df['quality'] > (df['quality'].mean()-df['quality'].std()))
    ]
    numberDelete = df.shape[0] - newdf.shape[0]
    print()
    print('Executing quality control')
    print("Number of files before quality control: {0}".format(df.shape[0]))
    print("Number of files that don't match quality control: {0}".format(
        numberDelete))
    print("Number of files remaing after removal: {0}".format(newdf.shape[0]))
    return newdf


def removeStartAtFrameZero(df):
    newdf = df[
        (df['startFrame'] > 0)
    ]
    numberDelete = df.shape[0] - newdf.shape[0]
    print()
    print('Executing start frame check')
    print("Number of files before startFrameCheck: {0}".format(df.shape[0]))
    print("Number of files that don't have a startFrame: {0}".format(
        numberDelete))
    print("Number of files remaing after removal: {0}".format(newdf.shape[0]))
    return newdf


def removeEndAtLastFrame(df):
    newdf = df[
        (df['startFrame'] > 0)
    ]
    numberDelete = df.shape[0] - newdf.shape[0]
    print()
    print('Executing start frame check')
    print("Number of files before startFrameCheck: {0}".format(df.shape[0]))
    print("Number of files that don't have a startFrame: {0}".format(
        numberDelete))
    print("Number of files remaing after removal: {0}".format(newdf.shape[0]))
    return newdf


def speedCheck(df):
    newdf = df[
        df['speedPixelsPerFrame'] > (
            df['speedPixelsPerFrame'].mean()-(2*df['speedPixelsPerFrame'].std()))
    ]
    numberDelete = df.shape[0] - newdf.shape[0]
    print()
    print('Executing speed check')
    print("Number of files before speed outlier check: {0}".format(
        df.shape[0]))
    print("Number of speed outliers: {0}".format(numberDelete))
    print("Number of files remaing after removal: {0}".format(newdf.shape[0]))
    return newdf


def main():
    # Get list of files based on the file path.
    dataFiles = os.listdir(filePath)
    # Create dataframe based on csv files in data folder
    fileInfo = dataFrameBuilder(dataFiles)
    # Based on this dataframe process all files
    fileInfo = fileProcessor(fileInfo)
    # Safe result with all feature information to csv. 
    fileInfo.to_csv('fileInfo.csv')

if __name__ == '__main__':
    main()


