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


class RunVisualizer():
 
	def __init__(self, explorer=None, filename=None, resultFile=None):

		self.explorer = explorer
		self.filename = filename
		self.resultFile = resultFile

	def visualizeRuns(self):
		rHip = self.explorer.getLimbSelection('Rhip')
		lHip = self.explorer.getLimbSelection('Lhip')
		hips = pd.concat([rHip, lHip], axis=1, sort=False)
		hips['center'] = hips['X_rot'].mean(axis=1)
		hips = hips.fillna(0)
		hip = hips['center']
		hips['X_ori'] = hip
		hips['X_smooth'] = hip.shift(-7).rolling(15).median()
		x = hips

		x1 = hips.query('frameIndex >= ' + str(self.explorer.startRunOne) + ' and frameIndex <= ' + str(self.explorer.endRunOne))
		x1 = x1['X_smooth'].interpolate()
		y1 = np.arange(0, x1.shape[0])


		x2 = hips.query('frameIndex >= ' + str(self.explorer.startRunTwo) + ' and frameIndex <= ' + str(self.explorer.endRunTwo))
		x2 = x2['X_smooth'].interpolate()
		y2 = np.arange(0, x2.shape[0])

		x3 = hips.query('frameIndex >= ' + str(self.explorer.startRunThree) + ' and frameIndex <= ' + str(self.explorer.endRunThree))
		x3 =  x3['X_smooth'].interpolate()
		y3 = np.arange(0, x3.shape[0])

		x4 = hips.query('frameIndex >= ' + str(self.explorer.startRunFour) + ' and frameIndex <= ' + str(self.explorer.endRunFour))
		x4 = x4['X_smooth'].interpolate()
		y4 = np.arange(0, x4.shape[0])

		x = x.interpolate()
		y = np.arange(len(x))




		fig = plt.figure(tight_layout={'h_pad': 5}, figsize=(16,8))
		gs = gridspec.GridSpec(2, 4)

		ax = fig.add_subplot(gs[0, :])

		ax.plot(y, x['X_ori'], '-',  color="#999999", label='Original data')
		ax.plot(y, x['X_smooth'], '-', color="#2c3e50", label='Smoothed data')
		ax.set_xlim(0, 1250)
		ax.set_ylim(0, 1920)


		for xc in [self.explorer.startFrame, self.explorer.startRunTwo, self.explorer.startRunThree, self.explorer.startRunFour, self.explorer.endFrame]:
		    if xc == self.explorer.startFrame:
		        ax.axvline(x=xc, linestyle='--', color="#e74c3c", label='Track cut-off point')
		    else:
		        ax.axvline(x=xc, linestyle='--', color="#e74c3c")


		tRunOneXPos = self.explorer.startFrame + 5
		tRunTwoXPos = self.explorer.startRunTwo + 5
		tRunThreeXPos = self.explorer.startRunThree + 5
		tRunFourXPos = self.explorer.startRunFour + 5 

		ax.text(tRunOneXPos, 50, "Track 1", fontsize=10)
		ax.text(tRunTwoXPos, 50, "Track 2", fontsize=10)
		ax.text(tRunThreeXPos, 50, "Track 3", fontsize=10)
		ax.text(tRunFourXPos, 50, "Track 4", fontsize=10)

		ax.set_xlabel('frame index in video')
		ax.set_ylabel('X position in video')
		ax.set_title('Horizontal Right hip displacement - including seperation of the four tracks')
		ax.legend()


		ax = fig.add_subplot(gs[1, 0])
		ax.plot(y1, x1, '-', color="#2c3e50")
		ax.set_xlim(0, 250)
		ax.set_ylim(0, 1920)
		ax.invert_yaxis()
		ax.set_xlabel('duration in frames')
		ax.set_ylabel('X position in video')
		ax.set_title('Track 1 - Right to Left')

		ax = fig.add_subplot(gs[1, 1])
		ax.plot(y2, x2, '-', color="#2c3e50")
		ax.set_xlim(0, 250)
		ax.set_ylim(0, 1920)
		ax.set_xlabel('duration in frames')
		ax.set_ylabel('X position in video')
		ax.set_title('Track 2 - Left to Right')


		ax = fig.add_subplot(gs[1, 2])
		ax.plot(y3, x3, '-', color="#2c3e50")
		ax.set_xlim(0, 250)
		ax.set_ylim(0, 1920)
		ax.invert_yaxis()
		ax.set_xlabel('duration in frames')
		ax.set_ylabel('X position in video')
		ax.set_title('Track 3 - Right to Left')


		ax = fig.add_subplot(gs[1, 3])
		ax.plot(y4, x4, '-', color="#2c3e50")
		ax.set_xlim(0, 250)
		ax.set_ylim(0, 1920)
		ax.set_xlabel('duration in frames')
		ax.set_ylabel('X position in video')
		ax.set_title('Track 4 - Left to Right')

		plt.savefig('visualization output/run overview/'+ str(self.filename[:-4] + '.jpg'),
		            dpi=600)
		plt.close('all')

	def visualizeParabolaRun(self):

		f, ax = plt.subplots(1, 4, figsize=(24, 6))

		for i in range(0, 4):
		    if i == 0:
		        x, neck, l = self.explorer.neckParabola(
		            self.explorer.startRunOne, self.explorer.endRunOne, visualize=True)
		    if i == 1:
		        x, neck, l = self.explorer.neckParabola(
		            self.explorer.startRunTwo, self.explorer.endRunTwo, visualize=True)
		    if i == 2:
		        x, neck, l = self.explorer.neckParabola(
		            self.explorer.startRunThree, self.explorer.endRunThree, visualize=True)
		    if i == 3:
		        x, neck, l = self.explorer.neckParabola(
		            self.explorer.startRunFour, self.explorer.endRunFour, visualize=True)
		    ax[i].plot(x, neck, color='#2980b9')
		    ax[i].plot(x, l, color='#e67e22')
		    ax[i].set_xlabel('Time (number of frames)')
		    ax[i].set_ylabel('Y-position of neck coordinate')
		    ax[i].set_title('Track ' + str(i+1))
		    patch_blue = mpatches.Patch(
		        color='#2980b9', label='Original signal')
		    patch_orange = mpatches.Patch(
		        color='#e67e22', label='Polynomial fit')
		    ax[i].legend(handles=[patch_blue, patch_orange], loc=4)
		plt.savefig('visualization output/parabola/'+ str(self.filename[:-4] + '.jpg'),
		            dpi=600)
		plt.close('all')

	def visualizeAccelerationPhase(self):

		f, ax = plt.subplots(1, 4, figsize=(24, 6))

		for i in range(0, 4):
			if i == 0:
			    x, hips, l, phaseEnd = self.explorer.runPhaseSeperator(
			        self.explorer.startRunOne, self.explorer.endRunOne, visualize=True)
			if i == 1:
			    x, hips, l, phaseEnd = self.explorer.runPhaseSeperator(
			        self.explorer.startRunTwo, self.explorer.endRunTwo, visualize=True)
			if i == 2:
			    x, hips, l, phaseEnd = self.explorer.runPhaseSeperator(
			        self.explorer.startRunThree, self.explorer.endRunThree, visualize=True)
			if i == 3:
			    x, hips, l, phaseEnd = self.explorer.runPhaseSeperator( 
			        self.explorer.startRunFour, self.explorer.endRunFour, visualize=True)
			ax[i].plot(x, hips, color='#2980b9')
			ax[i].plot(x, l, color='#e67e22')
			ax[i].axvline(x=phaseEnd, linestyle='--', color="#e74c3c", label='Phase cut-off point')
			ax[i].set_xlabel('Time (number of frames)')
			ax[i].set_ylabel('X-position of center of hips')
			ax[i].set_title('Track ' + str(i+1))
			ax[i].set_xlim(0, 275)
			ax[i].set_ylim(0, 1920)
			patch_blue = mpatches.Patch(
			    color='#2980b9', label='Original signal')
			patch_orange = mpatches.Patch(
			    color='#e67e22', label='Polynomial fit')
			patch_red = mpatches.Patch(
			    color='#e74c3c', label='Phase cut-off point')
			ax[i].legend(handles=[patch_blue, patch_orange, patch_red ], loc=4)
		plt.savefig('visualization output/phase seperator/'+ str(self.filename[:-4] + '.jpg'),
			dpi=600)
		plt.close('all')




	def visualizeVerticalDisplacement(self):
	    f, ax = plt.subplots(1, 4, figsize=(24, 6))

	    for i in range(0, 4):
	        if i == 0:
	            neck = self.explorer.verticalDisplacementDeviation(
	                'neck', self.explorer.startRunOne, self.explorer.endRunOne, visualize=True)
	        if i == 1:
	            neck = self.explorer.verticalDisplacementDeviation(
	                'neck', self.explorer.startRunTwo, self.explorer.endRunTwo, visualize=True)
	        if i == 2:
	            neck = self.explorer.verticalDisplacementDeviation(
	                'neck', self.explorer.startRunThree, self.explorer.endRunThree, visualize=True)
	        if i == 3:
	            neck = self.explorer.verticalDisplacementDeviation(
	                'neck',  self.explorer.startRunFour, self.explorer.endRunFour, visualize=True)
	        ax[i].plot(neck, color='#2980b9')
	        ax[i].set_xlabel('Time (number of frames)')
	        ax[i].set_ylabel('Y-position of neck')
	        ax[i].set_title('Track ' + str(i+1))
	    plt.savefig('visualization output/vertical displacement/'+ str(self.filename[:-4] + '.jpg'),
	            dpi=600)
	    plt.close('all')