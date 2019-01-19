import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.signal import find_peaks_cwt
from scipy.optimize import fmin
from peakdet import peakdet
from scipy import fftpack


import matplotlib.pyplot as plt


class RunExplorer():

    def __init__(self, filename):
        print('RunExplorer processing file: {0}'.format(filename))

        self.df = self.readFile(filename)
        self.df = self.replaceMinusOneWithNan()
        self.df = self.changeCoordinateSystem()
        self.horizontalCorrection()
        self.findTurningPoints()
        self.bodyLength = self.bodyLength()
        self.legLength = self.legLength()
        self.durationInFrames = self.endFrame - self.startFrame
        self.speedPixelsPerFrame = abs(
            self.startXVal - self.endXVal)*4 / self.durationInFrames
        self.runOneDuration = self.endRunOne - self.startRunOne
        self.runTwoDuration = self.endRunTwo - self.startRunTwo
        self.runThreeDuration = self.endRunThree - self.startRunThree
        self.runFourDuration = self.endRunFour - self.startRunFour

    # QUALITY AND CORRRECTIONS

    def getKeypointQuality(self, keyPoint, startRun=None, endRun=None):
        keyPoint = self.getKeypointSelection(keyPoint)
        keyPoint = keyPoint[startRun:endRun]
        return 1-(keyPoint['keyPointScore'].isna().sum()/keyPoint.shape[0])

    def readFile(self, filename):
        """Read CSV (comma-separated) file into DataFrame.

         Parameters
        ----------
        filename : string
        The filename of the csv file.

        Returns
        -------
        pandas.DataFrame
            The DataFrame that was extracted from the csv file.
        """
        columnNames = ["fileIndex", "frameIndex",
                       "keyPointIndex", "keyPointX", "keyPointY", "keyPointScore"]
        return pd.read_csv(filename, names=columnNames)

    def replaceMinusOneWithNan(self):
        """Function that replaces all the minus one values with NaN.

        Returns
        -------
        pandas.DataFrame
        The DataFrame contains no more -1 values.
        """

        return self.df.replace(-1, np.nan)

    def getStartFrame(self):
        startFrame = self.startIndex.frameIndex
        return startFrame

    def getStartX(self):
        startX = self.startIndex.keyPointX
        return startX

    def leftRightComparison(self, RKeypoint, LKeypoint, axis):
        if axis == 'X':
            hipR = self.getKeypointSelection(RKeypoint)['X_rot']
            hipL = self.getKeypointSelection(LKeypoint)['X_rot']
        if axis == 'Y':
            hipR = self.getKeypointSelection(RKeypoint)['Y_rot']
            hipL = self.getKeypointSelection(LKeypoint)['Y_rot']
        return hipR, hipL

    def LRComparisonLeftSideVisible(self, RKeypoint, LKeypoint, axis):
        if axis == 'X':
            hipR = self.getKeypointSelection(RKeypoint)['X_rot']
            hipL = self.getKeypointSelection(LKeypoint)['X_rot']
            hipR1 = hipR[self.startRunOne:self.endRunOne]
            hipR3 = hipR[self.startRunThree:self.endRunThree]
            hipL1 = hipL[self.startRunOne:self.endRunOne]
            hipL3 = hipL[self.startRunThree:self.endRunThree]

        return hipR1, hipR3, hipL1, hipL3

    def LRComparisonRightSideVisible(self, RKeypoint, LKeypoint, axis):
        if axis == 'X':
            hipR = self.getKeypointSelection(RKeypoint)['X_rot']
            hipL = self.getKeypointSelection(LKeypoint)['X_rot']
            hipR2 = hipR[self.startRunTwo:self.endRunTwo]
            hipR4 = hipR[self.startRunFour:self.endRunFour]
            hipL2 = hipL[self.startRunTwo:self.endRunTwo]
            hipL4 = hipL[self.startRunFour:self.endRunFour]

        return hipR2, hipR4, hipL2, hipL4

    def horizontalCorrection(self):
        ankle = self.getKeypointSelection('Rank')
        startYValue = ankle.iloc[0].keyPointY
        startX = ankle.iloc[0].keyPointX
        middleOfMovie = 1080 - ankle
        # select middle image 120 pixel range
        middleAnkle = ankle[(ankle['keyPointX'] > 900) &
                            (ankle['keyPointX'] < 1020)]
        minimaIndexes = find_peaks_cwt(
            (1080 - middleAnkle['keyPointY']), np.arange(1, 20))
        minimumYValue = 1080
        xValue = startX

        for i in minimaIndexes:
            frameYValue = middleAnkle.iloc[i].keyPointY
            if frameYValue < minimumYValue:
                minimumYValue = frameYValue
                xValue = middleAnkle.iloc[i].keyPointX

        oppositeSide = abs(startYValue-minimumYValue)
        adjacentSide = abs(startX - xValue)
        tangentInRadians = np.tan(oppositeSide/adjacentSide)
        tangentInDegrees = np.degrees(tangentInRadians)

        if minimumYValue < startYValue:
            tangentInDegrees = -1*tangentInDegrees

        rotated_df = self.applyRotationMatrix(tangentInDegrees)
        return rotated_df

    def applyRotationMatrix(self, degree):
        theta = np.radians(degree)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, s), (s, c)))

        rotated_df = self.df

        rotated_df['X_rot'] = rotated_df['keyPointX'] * \
            c - rotated_df['keyPointY'] * s
        rotated_df['Y_rot'] = rotated_df['keyPointX'] * \
            s + rotated_df['keyPointY'] * c

        return rotated_df

    def findTurningPoints(self):
        Rhip = self.getKeypointSelection('Rhip')
        Lhip = self.getKeypointSelection('Lhip')
        hips = pd.concat([Rhip, Lhip], axis=1, sort=False)
        hips['center'] = hips['X_rot'].mean(axis=1)
        x = hips['center'].shift(-7).rolling(15).median()

        xt = x.interpolate()
        maxtab, mintab = peakdet(xt, 100)
        minPeaks = []
        b, a = np.argmin(mintab, axis=0)
        minPeaks.append(mintab[a][0].astype(int))
        mintab = np.delete(mintab, a, 0)
        b, a = np.argmin(mintab, axis=0)
        minPeaks.append(mintab[a][0].astype(int))
        minPeaks = sorted(minPeaks)
        offsetMiddlePeak = 1000
        for i in range(0, maxtab.shape[0]):
            if abs(600 - maxtab[i][0]) < offsetMiddlePeak:
                offsetMiddlePeak = abs(600 - maxtab[i][0])
                middlePeak = maxtab[i][0]

        middlePeak = middlePeak.astype(int)

        flagXValue = hips.iloc[middlePeak].center
        # find start by walking back from first valley
        for i in range(minPeaks[0], 0, -1):
            searchX = hips.iloc[i].center
            self.startRunOne = 0
            if (searchX > flagXValue) & (i < 250):
                self.startRunOne = hips.iloc[i].frameIndex.values[0]
                break
        # find end by walking forward from last valley
        for i in range(minPeaks[1], hips.shape[0]):
            searchX = hips.iloc[i].center
            self.endRunFour = 1200
            if (searchX > flagXValue) & (i > 800):
                self.endRunFour = hips.iloc[i].frameIndex.values[0]
                break

        self.endRunOne = int(hips.iloc[minPeaks[0]].frameIndex.values[0])
        self.endRunTwo = int(hips.iloc[middlePeak].frameIndex.values[0])
        self.endRunThree = int(hips.iloc[minPeaks[1]].frameIndex.values[0])
        self.endRunFour = int(self.endRunFour)
        self.startRunOne = int(self.startRunOne)
        self.startRunTwo = int(self.endRunOne + 1)
        self.startRunThree = int(self.endRunTwo + 1)
        self.startRunFour = int(self.endRunThree + 1)
        self.startIndex = 6
        self.startFrame = int(self.startRunOne)
        self.endFrame = int(self.endRunFour)
        self.startXVal = int(hips.iloc[(self.startRunTwo)].center)
        self.endXVal = hips.iloc[(self.endRunTwo)].center

    def changeCoordinateSystem(self):
        self.df['keyPointY'] = 1080 - self.df['keyPointY']
        return self.df

    def getKeypointSelection(self, keyPointOfInterest):
        """Private method that selects only those rows of a dataframe that belong to a certain keyPoint

        """
        # List of keyPoints
        keyPoints = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip",
                     "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Reye", "Leye", "Rear", "Lear"]
        # Find the index of the keyPoint of interest
        keyPointIndex = keyPoints.index(keyPointOfInterest)
        # Filter the dataframe to on the keyPoint
        keyPointDF = self.df[self.df.keyPointIndex == keyPointIndex]
        # Reset the index of the df to match the keyPoint
        keyPointDF = keyPointDF.reset_index(drop=True)
        return keyPointDF

    def getFileQuality(self):
        return 1-(self.df['keyPointScore'].isna().sum()/self.df.shape[0])

    def getEnd(self, keyPoint):
        keyPointSelection = self.getKeypointSelection(keyPoint)
        try:
            end = keyPointSelection[(keyPointSelection['keyPointX'] > self.startX) & (
                keyPointSelection['frameIndex'] > 800)].iloc[0].frameIndex
        except IndexError:
            end = keyPointSelection.shape[0]
        return end

    """
    The remaing functions are deticated to the features described in chapter 4.
    Section 4.8 of the thesis also dives into these as well
    """

    def bodyLength(self):
        """Calculate the body length of the athlete. 

        Returns
        -------
        float
            The length in number of pixels. A float rather than a integer since rotation can cause decimal pixels.
        """
        # Select the neck
        neck = self.getKeypointSelection('neck')
        # Select the left ankle
        ankle = self.getKeypointSelection('Lank')
        # Calclate the body length as the median observation of the first twenty frames.
        bodyLength = ((neck['Y_rot'][0:20]) - (ankle['Y_rot'][:20])).median()
        return bodyLength

    def legLength(self):
        """Calculate the leg length of the athlete. 

        Returns
        -------
        float
            The length in number of pixels. A float rather than a integer since rotation can cause decimal pixels.
        """
        # Select the left hip
        hip = self.getKeypointSelection('Lhip')
        # Select the left ankle
        ankle = self.getKeypointSelection('Lank')
        # Calclate the leg length as the median observation of the first twenty frames.
        legLength = ((hip['Y_rot'][0:20]) - (ankle['Y_rot'][:20])).median()
        return legLength

    def verticalDisplacementDeviation(self, keyPoint, startFrame=None, endFrame=None,  visualize=False):
        """Calculate the vertical displacement of a keyPoint as the standard deviation of the signal. 

        Returns
        -------
        float
            The standard deviation of the vertical postion of the keyPoint. 
        """
        keyPointSelection = self.getKeypointSelection(keyPoint)
        keyPointSelection = keyPointSelection.interpolate()
        if startFrame is not None and endFrame is not None:
            keyPointSelection = keyPointSelection[
                (keyPointSelection['frameIndex'] >= startFrame) &
                (keyPointSelection['frameIndex'] <= endFrame)]
        else:
            keyPointSelection = keyPointSelection[
                (keyPointSelection['frameIndex'] >= self.startFrame) &
                (keyPointSelection['frameIndex'] <= self.endFrame)]
        verticalDisplacement = keyPointSelection['Y_rot'].std()
        if visualize == True:
            return keyPointSelection['Y_rot']
        else:
            return float(verticalDisplacement)

    def neckParabola(self, startFrame, endFrame,  visualize=False):
        """Calculate the leg length of the athlete. 

        Returns
        -------
        float
            The a*2 value of a in  a x + b, which is the first order derivative of the parabolic function. 
        """
        # Select the neck
        neck = self.getKeypointSelection('neck')
        # Apply an interpolation function, thereafter shift two and apply a median smoothing filter.
        neck['Y_rot'] = neck['Y_rot'].interpolate().shift(-2).rolling(5).median()
        # Fill nan opbservations with zeros (polyfit does not accept NaN values)
        neck = neck.fillna(0)
        # Make a selection if the start and end of a run are defined
        neck = neck[(neck['frameIndex'] >= startFrame)
                    & (neck['frameIndex'] <= endFrame)]
        # Select the Y position of the neck as pandas series to use.
        neck = neck['Y_rot']
        x = np.arange(0, neck.shape[0])
        fitFunction = np.poly1d(np.polyfit(x, neck, 2))
        twoA = fitFunction.c[0]*2
        if visualize == True:
            f = np.poly1d(fitFunction)
            l = f(x)
            return(x, neck, l)
        else:
            return(twoA)

    def runPhaseSeperator(self,  startFrame, endFrame,  visualize=False):
        """Seperate the run phases based on a the second order derivative of a polynomial fit. 

        Returns
        -------
        int
            The index of the end of the acceleration phase.
        """

        # Select the right hip
        rHip = self.getKeypointSelection('Rhip')
        # Select the left hip
        lHip = self.getKeypointSelection('Lhip')
        # Combine both in one dataframe
        hips = pd.concat([rHip, lHip], axis=1, sort=False)
        # Add a new column
        hips['center'] = hips['X_rot'].mean(axis=1)
        hips['center'] = hips['center'].interpolate(
        ).shift(-2).rolling(5).median()
        hips = hips.fillna(0)
        hips = hips.query('frameIndex >= ' + str(startFrame) +
                          ' and frameIndex <= ' + str(endFrame))
        hip = hips['center']
        x = np.arange(0, hip.shape[0])
        p = np.poly1d(np.polyfit(x, hip, 3))
        fitFunction = np.poly1d(p)
        filledFunction = fitFunction(x)
        secondOrderDerivative = np.diff(filledFunction, n=2)
        index = (np.abs(secondOrderDerivative-0)).argmin()
        phaseEnd = x[index]
        if visualize == True:
            f = np.poly1d(fitFunction)
            l = f(x)
            return(x, hip, l, phaseEnd)
        else:
            return(phaseEnd)

    def stepFrequency(self, startRun=None, endRun=None):
        """Calculate the step frequceny based on the most dominant frequency after a fast fourier transform. 

        Returns
        -------
        float
            The most dominant frequency represented as a float. 
        """
        rHip = self.getKeypointSelection('Rhip')
        lHip = self.getKeypointSelection('Lhip')
        hips = pd.concat([rHip, lHip], axis=1, sort=False)
        hips['center'] = hips['Y_rot'].mean(axis=1)
        hips['center'] = hips['center'].interpolate(
        ).shift(-2).rolling(5).median()
        hips = hips.fillna(0)
        hips = hips.query('frameIndex >= ' + str(startRun) + ' & '
                          + str(endRun) + ' <= frameIndex')
        hip = hips['center']
        sig = hip
        # The FFT of the signal
        sig_fft = fftpack.fft(sig)
        # And the power (sig_fft is of complex dtype)
        power = np.abs(sig_fft)
        # The corresponding frequencies
        sample_freq = fftpack.fftfreq(sig.size, d=1/60)
        # Find the peak frequency: we can focus on only the positive frequencies
        pos_mask = np.where(sample_freq > 5)
        freqs = sample_freq[pos_mask]
        peak_freq = freqs[power[pos_mask].argmax()]
        return float(peak_freq)
