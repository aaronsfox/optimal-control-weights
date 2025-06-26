# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    ADD NOTES...

    TODO:
        > Adjusting sphere stiffness could work - but is it appropriate (requires huge drop in stiffness to become appropriate and leads to sliding...)
        > Find anthropometric data somewhere?
        > Add metabolics to muscles during scaling
        > Use path point set for muscle paths?
        > Finalise main part of code with function use


"""

# =========================================================================
# Import packages
# =========================================================================

import opensim as osim
import os
import glob
import shutil
import numpy as np
import pandas as pd
import pickle
from scipy.constants import g
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# =========================================================================
# Import tools
# =========================================================================

# Add muscle parameter optimisation tool path
curr_dir = os.path.dirname(__file__)
tools_path = os.path.join(curr_dir,'..','..','..','tools','MuscleParamOptimizer','Python_tool')
sys.path.append(tools_path)

# Add functions from tool
from optimMuscleParams import *

# =========================================================================
# Modify to run different parts of code
# =========================================================================

# Set which processing steps to run
runScaling = False
runMuscleParamOpt = False
runIK = False
runTracking = False
runContactAdjustment = False

# =========================================================================
# Set-up
# =========================================================================

# Plot settings
# -------------------------------------------------------------------------

# Set matplotlib parameters
from matplotlib import rcParams
import matplotlib
matplotlib.use('TkAgg')
plt.ion()

# rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 16
rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['legend.fontsize'] = 10
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['legend.framealpha'] = 0.0
rcParams['savefig.dpi'] = 300
rcParams['savefig.format'] = 'pdf'

# General settings
# -------------------------------------------------------------------------

# Get participant list from data folder
participant_list = [ii for ii in os.listdir(os.path.join('..', 'data')) if os.path.isdir(os.path.join('..', 'data', ii))]

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', 'model', 'Geometry'))

# Create dictionaries for tools to avoid over-writing
scaleTool = {participantId: osim.ScaleTool() for participantId in participant_list}
scaleTool_2d = {participantId: osim.ScaleTool() for participantId in participant_list}
ikTool = {participantId: osim.InverseKinematicsTool() for participantId in participant_list}

# Set kinematics filter frequency
kinematicFiltFreq = 6

# Create dictionary specifying condition colouring
# Set both HEX and RGB colours (https://www.rapidtables.com/convert/color/hex-to-rgb.html)
# RGB are only used as osim Vec3 objects so they can be set that way here
condition_col = {
    '3D': {'HEX': '#4885ed', 'RGB': osim.Vec3(0.2823529411764706, 0.5215686274509804, 0.9294117647058824)},  # 3D = blue,
    '2D': {'HEX': '#ffa600', 'RGB': osim.Vec3(1, 0.6509803921568628, 0)}  # 2D = gold
}

# OpenSim settings
# -------------------------------------------------------------------------

# Create measurement set for scaling

# Set the parameters for the measurement sets
measurementSetParams = {
    # Torso
    'torso_depth': {'markerPairs': [['C7', 'CLAV'], ['T10', 'STRN'], ], 'bodyScale': ['torso'], 'axes': 'X'},
    'torso_height': {'markerPairs': [['C7', 'T10'], ['CLAV', 'STRN'], ], 'bodyScale': ['torso'], 'axes': 'Y'},
    'torso_width': {'markerPairs': [['RSHO', 'LSHO'],], 'bodyScale': ['torso'], 'axes': 'Z'},
    # Pelvis
    'pelvis_depth': {'markerPairs': [['RASI', 'SACR'], ['LASI', 'SACR'], ], 'bodyScale': ['pelvis'], 'axes': 'X'},
    'pelvis_height': {'markerPairs': [['RASI', 'LASI'], ], 'bodyScale': ['pelvis'], 'axes': 'Y'},
    'pelvis_width': {'markerPairs': [['RASI', 'LASI'], ], 'bodyScale': ['pelvis'], 'axes': 'Z'},
    # Left thigh
    'l_thigh': {'markerPairs': [['LASI', 'LKNE'], ], 'bodyScale': ['femur_l'], 'axes': 'XYZ'},
    # Left patella
    'l_patella': {'markerPairs': [['LASI', 'LKNE'], ], 'bodyScale': ['patella_l'], 'axes': 'XYZ'},
    # Left shank
    'l_tibia': {'markerPairs': [['LKNE', 'LANK'], ], 'bodyScale': ['tibia_l', 'talus_l'], 'axes': 'XYZ'},
    # Left foot
    'l_foot': {'markerPairs': [['LHEE', 'LTOE'], ], 'bodyScale': ['calcn_l', 'toes_l'], 'axes': 'XYZ'},
    # Right thigh
    'r_thigh': {'markerPairs': [['RASI', 'RKNE'], ], 'bodyScale': ['femur_r'], 'axes': 'XYZ'},
    # Right patella
    'r_patella': {'markerPairs': [['RASI', 'RKNE'], ], 'bodyScale': ['patella_r'], 'axes': 'XYZ'},
    # Right shank
    'r_tibia': {'markerPairs': [['RKNE', 'RANK'], ], 'bodyScale': ['tibia_r', 'talus_r'], 'axes': 'XYZ'},
    # Right foot
    'r_foot': {'markerPairs': [['RHEE', 'RTOE'], ], 'bodyScale': ['calcn_r', 'toes_r'], 'axes': 'XYZ'},
}

# Create the measurement set
scaleMeasurementSet = osim.MeasurementSet()

# Append the measurements from parameters
for measureName in measurementSetParams.keys():
    # Create the measurement
    measurement = osim.Measurement()
    measurement.setName(measureName)
    # Append the marker pairs
    for ii in range(len(measurementSetParams[measureName]['markerPairs'])):
        measurement.getMarkerPairSet().cloneAndAppend(
            osim.MarkerPair(measurementSetParams[measureName]['markerPairs'][ii][0],
                            measurementSetParams[measureName]['markerPairs'][ii][1]))
    # Append the body scales
    for ii in range(len(measurementSetParams[measureName]['bodyScale'])):
        # Create body scale
        bodyScale = osim.BodyScale()
        bodyScale.setName(measurementSetParams[measureName]['bodyScale'][ii])
        # Create and set axis names
        axes = osim.ArrayStr()
        for jj in range(len(measurementSetParams[measureName]['axes'])):
            axes.append(measurementSetParams[measureName]['axes'][jj])
        bodyScale.setAxisNames(axes)
        # Apppend to body scale set
        measurement.getBodyScaleSet().cloneAndAppend(bodyScale)
    # Append the measurement to the set
    scaleMeasurementSet.cloneAndAppend(measurement)

# Create scale task set

# Set the parameters for the scale marker and joint task sets
markerParams = {
    # Torso
    'CLAV': {'weight': 5.0}, 'STRN': {'weight': 5.0}, 'C7': {'weight': 5.0}, 'T10': {'weight': 5.0},
    'RSHO': {'weight': 0.0}, 'LSHO': {'weight': 0.0},
    # Pelvis
    'LASI': {'weight': 10.0}, 'RASI': {'weight': 10.0}, 'SACR': {'weight': 10.0},
    # Right thigh
    'RTHI': {'weight': 0.0}, 'RKNE': {'weight': 5.0},
    # Right shank
    'RTIB': {'weight': 0.0},
    # Right foot
    'RHEE': {'weight': 5.0}, 'RANK': {'weight': 2.5}, 'RTOE': {'weight': 2.5},
    # Left thigh
    'LTHI': {'weight': 0.0}, 'LKNE': {'weight': 5.0},
    # Right shank
    'LTIB': {'weight': 0.0},
    # Right foot
    'LHEE': {'weight': 5.0}, 'LANK': {'weight': 2.5}, 'LTOE': {'weight': 2.5},
}
jointParams = {
    'lumbar_ext': 0.001, 'lumbar_bend': 0.001, 'lumbar_rota': 0.001,
    'pelvis_tilt': 0.001, 'pelvis_list': 0.001, 'pelvis_rotation': 0.001,
    'hip_flexion_r': 0.001, 'hip_adduction_r': 0.001, 'hip_rotation_r': 0.001,
    'knee_angle_r': 0.001, 'ankle_angle_r': 0.001,  'subtalar_angle_r': 0.001, 'mtp_angle_r': 0.001,
    'hip_flexion_l': 0.001, 'hip_adduction_l': 0.001, 'hip_rotation_l': 0.001,
    'knee_angle_l': 0.001, 'ankle_angle_l': 0.001, 'subtalar_angle_l': 0.001, 'mtp_angle_l': 0.001,
    }

# Create the task set
scaleTaskSet = osim.IKTaskSet()

# Append the tasks from the parameters
for taskName in markerParams.keys():
    # Create the task and add details
    task = osim.IKMarkerTask()
    task.setName(taskName)
    task.setWeight(markerParams[taskName]['weight'])
    if markerParams[taskName]['weight'] == 0.0:
        task.setApply(False)
    # Append to task set
    scaleTaskSet.cloneAndAppend(task)

# Append the joints from the parameters
for jointName in jointParams:
    # Create the task and add details
    jointTask = osim.IKCoordinateTask()
    jointTask.setName(jointName)
    jointTask.setWeight(jointParams[jointName])
    # Append to task set
    scaleTaskSet.cloneAndAppend(jointTask)

# Create the IK task set for tracking
""" NOTE: Defaulting to all 1's as subsequent steps will adjust kinematics a little """

# Set the parameters for the IK task sets
ikTaskSetParams = {
    # Torso
    'CLAV': {'weight': 1}, 'STRN': {'weight': 1}, 'C7': {'weight': 1}, 'T10': {'weight': 1},
    'RSHO': {'weight': 1}, 'LSHO': {'weight': 1},
    # Pelvis
    'LASI': {'weight': 1}, 'RASI': {'weight': 1}, 'SACR': {'weight': 1},
    # Right thigh
    'RTHI': {'weight': 1}, 'RKNE': {'weight': 1},
    # Right shank
    'RTIB': {'weight': 1},
    # Right foot
    'RHEE': {'weight': 1}, 'RANK': {'weight': 1}, 'RTOE': {'weight': 1},
    # Left thigh
    'LTHI': {'weight': 1}, 'lKNE': {'weight': 1},
    # Left shank
    'LTIB': {'weight': 1},
    # Left foot
    'LHEE': {'weight': 1}, 'LANK': {'weight': 1}, 'LTOE': {'weight': 1},
}

# Create the task set
ikTaskSet = osim.IKTaskSet()

# Append the tasks from the parameters
for taskName in ikTaskSetParams.keys():
    # Create the task and add details
    task = osim.IKMarkerTask()
    task.setName(taskName)
    task.setWeight(ikTaskSetParams[taskName]['weight'])
    if ikTaskSetParams[taskName]['weight'] == 0.0:
        task.setApply(False)
    # Append to task set
    ikTaskSet.cloneAndAppend(task)

# Set actuator forces for torque-driven tracking simulations
actForces = {
    'pelvis_tx': {'actuatorType': 'residual', 'optForce': 2.5e0},
    'pelvis_ty': {'actuatorType': 'residual', 'optForce': 2.5e0},
    'pelvis_tz': {'actuatorType': 'residual', 'optForce': 2.5e0},
    'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 2.5e0},
    'pelvis_list': {'actuatorType': 'residual', 'optForce': 2.5e0},
    'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 2.5e0},
    'hip_flexion_r': {'actuatorType': 'torque', 'optForce': 300.0},
    'hip_adduction_r': {'actuatorType': 'torque', 'optForce': 200.0},
    'hip_rotation_r': {'actuatorType': 'torque', 'optForce': 100.0},
    'knee_angle_r': {'actuatorType': 'torque', 'optForce': 300.0},
    'ankle_angle_r': {'actuatorType': 'torque', 'optForce': 200.0},
    # 'subtalar_angle_r': {'actuatorType': 'torque', 'optForce': 100.0},
    # 'mtp_angle_r': {'actuatorType': 'torque', 'optForce': 50.0},
    'hip_flexion_l': {'actuatorType': 'torque', 'optForce': 300.0},
    'hip_adduction_l': {'actuatorType': 'torque', 'optForce': 200.0},
    'hip_rotation_l': {'actuatorType': 'torque', 'optForce': 100.0},
    'knee_angle_l': {'actuatorType': 'torque', 'optForce': 300.0},
    'ankle_angle_l': {'actuatorType': 'torque', 'optForce': 200.0},
    # 'subtalar_angle_l': {'actuatorType': 'torque', 'optForce': 100.0},
    # 'mtp_angle_l': {'actuatorType': 'torque', 'optForce': 50.0},
    'lumbar_ext': {'actuatorType': 'torque', 'optForce': 300.0},
    'lumbar_bend': {'actuatorType': 'torque', 'optForce': 200.0},
    'lumbar_rota': {'actuatorType': 'torque', 'optForce': 100.0},
    }

# Set list of muscles to consider elastic tendons for
elasticTendons = {
    '2d': ['gastroc_r', 'gastroc_l', 'soleus_r', 'soleus_l'],
    '3d': ['gaslat_r', 'gaslat_l', 'gasmed_r', 'gasmed_l', 'soleus_r', 'soleus_l']
}

# Set tracking weights
globalMarkerTrackingWeight = 1.0e1
globalStateTrackingWeight = 1.0e0
globalControlEffortWeight = 1.0e-3

# Set mesh interval
meshInterval = 50

# =========================================================================
# Define functions
# =========================================================================

# Accessory functions for sine-wave fitting to pelvis_ty data
# -------------------------------------------------------------------------

# Generate sine wave from parameters
def sine_wave(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(frequency * x + phase) + offset

# Estimate sine wave frequency from data using FFT
def estimate_frequency(x, y):
    y_detrended = y - np.mean(y)
    fft = np.fft.fft(y_detrended)
    freqs = np.fft.fftfreq(len(x), d=(x[1] - x[0]))
    positive = freqs > 0
    freq_guess = freqs[positive][np.argmax(np.abs(fft[positive]))]
    return 2 * np.pi * freq_guess  # convert to angular frequency

# Fit sine wave constraining start and end values to the same
def fit_sine_with_constraint(x_data, y_data):
    # Estimate initial frequency from FFT
    freq_guess = estimate_frequency(x_data, y_data)

    # Initial guess
    amplitude_guess = (np.max(y_data) - np.min(y_data)) / 2
    offset_guess = np.mean(y_data)
    phase_guess = 0
    initial_guess = [amplitude_guess, freq_guess, phase_guess, offset_guess]

    # Loss function with periodicity constraint
    def loss(params):
        amplitude, frequency, phase, offset = params
        y_fit = sine_wave(x_data, amplitude, frequency, phase, offset)
        mse = np.mean((y_fit - y_data) ** 2)
        endpoint_penalty = 1e4 * (y_fit[0] - y_fit[-1]) ** 2
        return mse + endpoint_penalty

    # Optional: Add parameter bounds to avoid crazy values
    bounds = [
        (0, 2 * amplitude_guess),  # amplitude
        (1e-3, 10 * freq_guess),  # frequency
        (-2 * np.pi, 2 * np.pi),  # phase
        (offset_guess - 2 * amplitude_guess, offset_guess + 2 * amplitude_guess),  # offset
    ]

    result = minimize(loss, initial_guess, method='L-BFGS-B', bounds=bounds)

    if result.success:
        return result.x
    else:
        print("Optimization failed:", result.message)
        return None


# Scale participant model
# -------------------------------------------------------------------------
def run_scaling(participant_id):

    """
    :param participant_id: participant ID to run scaling for
    :return:
    """

    # Create scaling directory for files
    os.makedirs(os.path.join('..', 'data', participant_id, 'scaling'), exist_ok=True)

    # Get static trial file
    static_trc_file = os.path.join('..', 'data', participant_id, 'static', 'static.trc')
    static_c3d_file = os.path.join('..', 'data', participant_id, 'static', 'static.c3d')

    # Construct opensim c3d object
    c3dFile = osim.C3DFileAdapter()
    c3dFile.setLocationForForceExpression(osim.C3DFileAdapter.ForceLocation_CenterOfPressure)

    # Read in the static trial
    staticC3D = c3dFile.read(static_c3d_file)

    # Get forces table for mass
    staticForces = c3dFile.getForcesTable(staticC3D).flatten()

    # Get mass from the plate the participant is clearly standing on
    vert_force_labels = [col for col in staticForces.getColumnLabels() if col.startswith('f') and col.endswith('_3')]
    vert_force_static = [staticForces.getDependentColumn(col).to_numpy().mean() / g for col in vert_force_labels]
    mass_kg = np.array(vert_force_static).max()

    # Set-up and run the scale tool for the 3D model
    # -------------------------------------------------------------------------

    # Set participant mass
    scaleTool[participant_id].setSubjectMass(mass_kg)

    # Set generic model file
    scaleTool[participant_id].getGenericModelMaker().setModelFileName(os.path.join('..', 'model',
                                                                                   'Miller2021_Crienkinge2023.osim'))

    # Set measurement set in model scaler
    scaleTool[participant_id].getModelScaler().setMeasurementSet(scaleMeasurementSet)

    # Set scale tasks in tool
    for ii in range(scaleTaskSet.getSize()):
        scaleTool[participant_id].getMarkerPlacer().getIKTaskSet().cloneAndAppend(scaleTaskSet.get(ii))

    # Set marker file
    scaleTool[participant_id].getMarkerPlacer().setMarkerFileName(static_trc_file)
    scaleTool[participant_id].getModelScaler().setMarkerFileName(static_trc_file)

    # Set options
    scaleTool[participant_id].getModelScaler().setPreserveMassDist(True)
    scaleOrder = osim.ArrayStr()
    scaleOrder.set(0, 'measurements')
    scaleTool[participant_id].getModelScaler().setScalingOrder(scaleOrder)

    # Set time ranges
    timeRange = osim.ArrayDouble()
    timeRange.set(0, osim.TimeSeriesTableVec3(static_trc_file).getIndependentColumn()[0])  # initial time
    timeRange.set(1, osim.TimeSeriesTableVec3(static_trc_file).getIndependentColumn()[-1])  # final time
    scaleTool[participant_id].getMarkerPlacer().setTimeRange(timeRange)
    scaleTool[participant_id].getModelScaler().setTimeRange(timeRange)

    # Set output files
    scaleTool[participant_id].getModelScaler().setOutputModelFileName(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModel_3d.osim'))
    scaleTool[participant_id].getModelScaler().setOutputScaleFileName(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaleSet_3d.xml'))

    # Set marker adjustment parameters
    scaleTool[participant_id].getMarkerPlacer().setOutputMotionFileName(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_staticMotion_3d.mot'))
    scaleTool[participant_id].getMarkerPlacer().setOutputModelFileName(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModelAdjusted_3d.osim'))

    # Save and run scale tool
    scaleTool[participant_id].printToXML(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaleSetup_3d.xml'))
    scaleTool[participant_id].run()

    # Set-up and run the scale tool for the 2D model (no marker adjustment)
    # -------------------------------------------------------------------------

    # Set participant mass
    scaleTool_2d[participant_id].setSubjectMass(mass_kg)

    # Turn off marker placer
    scaleTool_2d[participant_id].getMarkerPlacer().setApply(False)

    # Set generic model file
    scaleTool_2d[participant_id].getGenericModelMaker().setModelFileName(os.path.join('..', 'model',
                                                                                      'Denton2023_Crienkinge2023.osim'))

    # Set measurement set in model scaler
    scaleTool_2d[participant_id].getModelScaler().setMeasurementSet(scaleMeasurementSet)

    # Set marker file
    scaleTool_2d[participant_id].getMarkerPlacer().setMarkerFileName(static_trc_file)
    scaleTool_2d[participant_id].getModelScaler().setMarkerFileName(static_trc_file)

    # Set options
    scaleTool_2d[participant_id].getModelScaler().setPreserveMassDist(True)
    scaleOrder = osim.ArrayStr()
    scaleOrder.set(0, 'measurements')
    scaleTool_2d[participant_id].getModelScaler().setScalingOrder(scaleOrder)

    # Set time ranges
    timeRange = osim.ArrayDouble()
    timeRange.set(0, osim.TimeSeriesTableVec3(static_trc_file).getIndependentColumn()[0])  # initial time
    timeRange.set(1, osim.TimeSeriesTableVec3(static_trc_file).getIndependentColumn()[-1])  # final time
    scaleTool_2d[participant_id].getModelScaler().setTimeRange(timeRange)

    # Set output files
    scaleTool_2d[participant_id].getModelScaler().setOutputModelFileName(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModel_2d.osim'))
    scaleTool_2d[participant_id].getModelScaler().setOutputScaleFileName(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaleSet_2d.xml'))

    # Save and run scale tool
    scaleTool_2d[participant_id].printToXML(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaleSetup_2d.xml'))
    scaleTool_2d[participant_id].run()

    # Adjust the 2D & 3D model
    # -------------------------------------------------------------------------

    # Load the scaled model back in
    scaledModel = osim.Model(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModelAdjusted_3d.osim'))
    scaledModel_2d = osim.Model(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModel_2d.osim'))

    # Set model name
    scaledModel.setName(f'{participant_id}_3d')
    scaledModel_2d.setName(f'{participant_id}_2d')

    # Scale model muscle forces according to height-mass relationship

    # Get generic model mass and set generic height for 3D model (this is the same as 2D model)
    genModel = osim.Model(os.path.join('..', 'model', 'Miller2021_Crienkinge2023.osim'))
    genModel_2d = osim.Model(os.path.join('..', 'model', 'Denton2023_Crienkinge2023.osim'))
    genModelMass = np.sum([genModel.getBodySet().get(bodyInd).getMass() for bodyInd in range(genModel.getBodySet().getSize())])
    genModelHeight = 1.70

    # Get scaled model height (use mass from earlier)
    # TODO: find anthropometrics somewhere?
    height_m = np.max(np.array([osim.TimeSeriesTableVec3(
        static_trc_file).flatten().getDependentColumn(f'{head_marker}_2').to_numpy().max() for head_marker in [
        'LFHD','RFHD','LBHD','RBHD']]) / 1000)

    # Get muscle volume totals based on mass and heights with linear equation
    genericMuscVol = 47.05 * genModelMass * genModelHeight + 1289.6
    scaledMuscVol = 47.05 * mass_kg * height_m + 1289.6

    # Loop through all muscles and scale according to volume and muscle parameters
    # Use this opportunity to also update contraction velocity as well

    # Set in 3D model
    for muscInd in range(scaledModel.getMuscles().getSize()):
        # Get current muscle name
        muscName = scaledModel.getMuscles().get(muscInd).getName()
        # Get optimal fibre length for muscle from each model
        genericL0 = genModel.getMuscles().get(muscName).getOptimalFiberLength()
        scaledL0 = scaledModel.getMuscles().get(muscName).getOptimalFiberLength()
        # Set force scale factor
        forceScaleFactor = (scaledMuscVol / genericMuscVol) / (scaledL0 / genericL0)
        # Scale current muscle strength
        scaledModel.getMuscles().get(muscInd).setMaxIsometricForce(
            forceScaleFactor * scaledModel.getMuscles().get(muscInd).getMaxIsometricForce())
        # Update max contraction velocity
        scaledModel.getMuscles().get(muscInd).setMaxContractionVelocity(20.0)

    # Set in 2D model
    for muscInd in range(scaledModel_2d.getMuscles().getSize()):
        # Get current muscle name
        muscName = scaledModel_2d.getMuscles().get(muscInd).getName()
        # Get optimal fibre length for muscle from each model
        genericL0 = genModel_2d.getMuscles().get(muscName).getOptimalFiberLength()
        scaledL0 = scaledModel_2d.getMuscles().get(muscName).getOptimalFiberLength()
        # Set force scale factor
        forceScaleFactor = (scaledMuscVol / genericMuscVol) / (scaledL0 / genericL0)
        # Scale current muscle strength
        scaledModel_2d.getMuscles().get(muscInd).setMaxIsometricForce(
            forceScaleFactor * scaledModel_2d.getMuscles().get(muscInd).getMaxIsometricForce())
        # Update max contraction velocity
        scaledModel_2d.getMuscles().get(muscInd).setMaxContractionVelocity(20.0)

    # Adjust contact geometry positions back to original reference locations

    # 3D model
    # Load in unadjusted model and get marker positions
    unadjustedModel = osim.Model(os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModel_3d.osim'))
    contactMarkerPos = {unadjustedModel.getMarkerSet().get(ii).getName(): unadjustedModel.getMarkerSet().get(ii).get_location() \
                        for ii in range(unadjustedModel.getMarkerSet().getSize()) if \
                        unadjustedModel.getMarkerSet().get(ii).getName().startswith('contact_')}
    # Adjusted contact geometry location in the adjusted model
    for contact in contactMarkerPos.keys():
        scaledModel.getContactGeometrySet().get(contact.replace('contact_','')).set_location(contactMarkerPos[contact])

    # 2D model
    # Load in unadjusted model and get marker positions
    unadjustedModel_2d = osim.Model(os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModel_2d.osim'))
    contactMarkerPos_2d = {unadjustedModel_2d.getMarkerSet().get(ii).getName(): unadjustedModel_2d.getMarkerSet().get(ii).get_location() \
                           for ii in range(unadjustedModel_2d.getMarkerSet().getSize()) if \
                           unadjustedModel_2d.getMarkerSet().get(ii).getName().startswith('contact_')}
    # Adjusted contact geometry location in the adjusted model
    for contact in contactMarkerPos_2d.keys():
        scaledModel_2d.getContactGeometrySet().get(contact.replace('contact_', '')).set_location(contactMarkerPos_2d[contact])

    # Adjust non-fixed markers in 2D model with 3D positions
    for adjust_marker in ['RTHI', 'LTHI', 'RTIB', 'LTIB']:
        scaledModel_2d.getMarkerSet().get(adjust_marker).set_location(scaledModel.getMarkerSet().get(adjust_marker).get_location())

    # Loop through the model bodies and set the colouring
    # Also adjust the opacity here

    # 3D model
    for bodyInd in range(scaledModel.updBodySet().getSize()):
        # Loop through the attached geomtries on body
        for gInd in range(scaledModel.updBodySet().get(bodyInd).getPropertyByName('attached_geometry').size()):
            # Set the colour
            scaledModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(condition_col['3D']['RGB'])
            # Set the opacity
            scaledModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.6)

    # 2D model
    for bodyInd in range(scaledModel_2d.updBodySet().getSize()):
        # Loop through the attached geomtries on body
        for gInd in range(scaledModel_2d.updBodySet().get(bodyInd).getPropertyByName('attached_geometry').size()):
            # Set the colour
            scaledModel_2d.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(condition_col['2D']['RGB'])
            # Set the opacity
            scaledModel_2d.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.6)

    # Loop through the muscles and set colouring

    # 3D model
    for muscInd in range(scaledModel.getMuscles().getSize()):
        scaledModel.getMuscles().get(muscInd).getGeometryPath().get_Appearance().set_color(condition_col['3D']['RGB'])

    # 2D model
    for muscInd in range(scaledModel_2d.getMuscles().getSize()):
        scaledModel_2d.getMuscles().get(muscInd).getGeometryPath().get_Appearance().set_color(condition_col['2D']['RGB'])

    # Finalise model connections
    scaledModel.finalizeConnections()
    scaledModel_2d.finalizeConnections()

    # Print to file (overwrites original adjusted model)
    scaledModel.printToXML(os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModelAdjusted_3d.osim'))
    scaledModel_2d.printToXML(os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModelAdjusted_2d.osim'))


# Optimise muscle parameters in scaled model
# -------------------------------------------------------------------------
def run_parameter_opt(participant_id, model_variant='3d', n_eval=10):

    """
    :param participant_id: string of participant to run optimisation for
    :param model_variant: set to optimise 3d or 2d model variant
    :param n_eval: number of evaluations per coordinate in the optimisation (defaults to 10 as per example)
    :return:
    """

    # Set files and directories
    # -------------------------------------------------------------------------

    # Create model directory for files
    os.makedirs(os.path.join('..', 'data', participant_id, 'model'), exist_ok=True)

    # Get the path to the desired scaled model file
    model_path = os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModelAdjusted_{model_variant}.osim')

    # Get the path to the reference model to map scaled model against
    if model_variant == '3d':
        ref_model_path = os.path.join('..', 'model', 'Miller2021_Crienkinge2023.osim')
    elif model_variant == '2d':
        ref_model_path = os.path.join('..', 'model', 'Denton2023_Crienkinge2023.osim')

    # Adjust model to work with tool
    # -------------------------------------------------------------------------
    # Optimiser tool doesn't seem to work with DeGrooteFregley muscle variant
    # Here the muscles are replaced by Millard muscles to ensure they work with the tool

    # Load the models
    ref_model = osim.Model(ref_model_path)
    target_model = osim.Model(model_path)

    # Create lists to store newly created muscles in
    new_ref_muscles = []
    new_target_muscles = []

    # Get muscle sets
    ref_model_muscles = ref_model.getMuscles()
    target_model_muscles = target_model.getMuscles()

    # Loop through reference model muscles to create Millard versions
    for musc_ind in range(ref_model_muscles.getSize()):
        # Get base muscle parameters
        base_musc = ref_model_muscles.get(musc_ind)
        # Create the new muscle
        new_musc = osim.Millard2012EquilibriumMuscle()
        # Perform all the common mappings at base class level (OpenSim::Muscle)
        new_musc.setName(base_musc.getName())
        new_musc.set_appliesForce(base_musc.get_appliesForce())
        new_musc.setMinControl(base_musc.getMinControl())
        new_musc.setMaxControl(base_musc.getMaxControl())
        new_musc.setMaxIsometricForce(base_musc.getMaxIsometricForce())
        new_musc.setOptimalFiberLength(base_musc.getOptimalFiberLength())
        new_musc.setTendonSlackLength(base_musc.getTendonSlackLength())
        new_musc.setPennationAngleAtOptimalFiberLength(base_musc.getPennationAngleAtOptimalFiberLength())
        new_musc.setMaxContractionVelocity(base_musc.getMaxContractionVelocity())
        new_musc.set_ignore_tendon_compliance(base_musc.get_ignore_tendon_compliance())
        new_musc.set_ignore_activation_dynamics(base_musc.get_ignore_activation_dynamics())
        new_musc.updGeometryPath().assign(base_musc.getGeometryPath())
        # Perform Millard muscle specific mappings
        new_musc.set_default_activation(osim.DeGrooteFregly2016Muscle().safeDownCast(
            base_musc).get_default_activation())
        new_musc.set_activation_time_constant(osim.DeGrooteFregly2016Muscle().safeDownCast(
            base_musc).get_activation_time_constant())
        new_musc.set_deactivation_time_constant(osim.DeGrooteFregly2016Muscle().safeDownCast(
            base_musc).get_deactivation_time_constant())
        new_musc.set_fiber_damping(osim.DeGrooteFregly2016Muscle().safeDownCast(
            base_musc).get_fiber_damping())
        new_musc.getFiberForceLengthCurve().set_strain_at_one_norm_force(osim.DeGrooteFregly2016Muscle().safeDownCast(
            base_musc).get_passive_fiber_strain_at_one_norm_force())
        new_musc.getTendonForceLengthCurve().set_strain_at_one_norm_force(osim.DeGrooteFregly2016Muscle().safeDownCast(
            base_musc).get_tendon_strain_at_one_norm_force())
        # Append new muscle to list
        new_ref_muscles.append(new_musc.clone())

    # Loop target model muscles to create Millard versions
    for musc_ind in range(target_model_muscles.getSize()):
        # Get base muscle parameters
        base_musc = target_model_muscles.get(musc_ind)
        # Create the new muscle
        new_musc = osim.Millard2012EquilibriumMuscle()
        # Perform all the common mappings at base class level (OpenSim::Muscle)
        new_musc.setName(base_musc.getName())
        new_musc.set_appliesForce(base_musc.get_appliesForce())
        new_musc.setMinControl(base_musc.getMinControl())
        new_musc.setMaxControl(base_musc.getMaxControl())
        new_musc.setMaxIsometricForce(base_musc.getMaxIsometricForce())
        new_musc.setOptimalFiberLength(base_musc.getOptimalFiberLength())
        new_musc.setTendonSlackLength(base_musc.getTendonSlackLength())
        new_musc.setPennationAngleAtOptimalFiberLength(base_musc.getPennationAngleAtOptimalFiberLength())
        new_musc.setMaxContractionVelocity(base_musc.getMaxContractionVelocity())
        new_musc.set_ignore_tendon_compliance(base_musc.get_ignore_tendon_compliance())
        new_musc.set_ignore_activation_dynamics(base_musc.get_ignore_activation_dynamics())
        new_musc.updGeometryPath().assign(base_musc.getGeometryPath())
        # Perform Millard muscle specific mappings
        new_musc.set_default_activation(osim.DeGrooteFregly2016Muscle().safeDownCast(
            base_musc).get_default_activation())
        new_musc.set_activation_time_constant(osim.DeGrooteFregly2016Muscle().safeDownCast(
            base_musc).get_activation_time_constant())
        new_musc.set_deactivation_time_constant(osim.DeGrooteFregly2016Muscle().safeDownCast(
            base_musc).get_deactivation_time_constant())
        new_musc.set_fiber_damping(osim.DeGrooteFregly2016Muscle().safeDownCast(
            base_musc).get_fiber_damping())
        new_musc.getFiberForceLengthCurve().set_strain_at_one_norm_force(osim.DeGrooteFregly2016Muscle().safeDownCast(
            base_musc).get_passive_fiber_strain_at_one_norm_force())
        new_musc.getTendonForceLengthCurve().set_strain_at_one_norm_force(osim.DeGrooteFregly2016Muscle().safeDownCast(
            base_musc).get_tendon_strain_at_one_norm_force())
        # Append new muscle to list
        new_target_muscles.append(new_musc.clone())

    # Get force set from models to use
    ref_forces = ref_model.getForceSet()
    target_forces = target_model.getForceSet()

    # Get muscle indices from the model force sets
    # Reference model
    ref_muscle_ind = []
    for force_ind in range(ref_forces.getSize()):
        if ref_forces.get(force_ind).getConcreteClassName() == 'DeGrooteFregly2016Muscle':
            ref_muscle_ind.append(force_ind)
    # Target model
    target_muscle_ind = []
    for force_ind in range(target_forces.getSize()):
        if target_forces.get(force_ind).getConcreteClassName() == 'DeGrooteFregly2016Muscle':
            target_muscle_ind.append(force_ind)

    # Remove muscles from the models
    # Reference model
    remove_counter = 0
    for remove_ind in ref_muscle_ind:
        ref_model.updForceSet().remove(remove_ind - remove_counter)
        remove_counter += 1
    # Target model
    remove_counter = 0
    for remove_ind in target_muscle_ind:
        target_model.updForceSet().remove(remove_ind - remove_counter)
        remove_counter += 1

    # Adopt the new Millard muscles
    for new_musc in new_ref_muscles:
        ref_model.updForceSet().cloneAndAppend(new_musc)
    for new_musc in new_target_muscles:
        target_model.updForceSet().cloneAndAppend(new_musc)

    # Finalize model before further edits
    ref_model.finalizeConnections()
    target_model.finalizeConnections()

    # Turn tendon dynamics off in certain muscles
    # Reference model
    for muscle_ind in range(ref_model.getMuscles().getSize()):
        if ref_model.getMuscles().get(muscle_ind).getName() not in elasticTendons[model_variant]:
            # Set to rigid tendon
            ref_model.getMuscles().get(muscle_ind).set_ignore_tendon_compliance(True)
    # Target model
    for muscle_ind in range(target_model.getMuscles().getSize()):
        if target_model.getMuscles().get(muscle_ind).getName() not in elasticTendons[model_variant]:
            # Set to rigid tendon
            target_model.getMuscles().get(muscle_ind).set_ignore_tendon_compliance(True)

    # Finalize model before saving
    ref_model.finalizeConnections()
    target_model.finalizeConnections()

    # Print new models to file
    ref_model.printToXML(ref_model_path.replace('.osim','_MillardMuscles.osim'))
    target_model.printToXML(model_path.replace('.osim', '_MillardMuscles.osim'))

    # Set where to store optimised model and info
    optim_folder = os.path.join('..', 'data', participant_id, 'model')
    log_folder = os.path.join('..', 'data', participant_id, 'model', f'optim_{model_variant}')

    # Optimise the target based on reference model for n_eval points per coord
    model_opt, sims_info = optimMuscleParams(ref_model_path.replace('.osim','_MillardMuscles.osim'),
                                             model_path.replace('.osim', '_MillardMuscles.osim'),
                                             n_eval, log_folder)

    # Convert muscles back to DeGrooteFregley version
    model_proc = osim.ModelProcessor(model_opt)
    model_proc.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    final_model = model_proc.process()
    final_model.setName(f'{participant_id}_{model_variant}_optim')

    # Write optimised model to file
    final_model.printToXML(os.path.join(optim_folder, f'{participant_id}_model_{model_variant}_optim.osim'))


# Run inverse kinematics
# -------------------------------------------------------------------------
def run_ik(participant_id):

    """
    :param participant_id: participant ID to run IK for
    :return:
    """

    # Create scaling directory for files
    os.makedirs(os.path.join('..', 'data', participant_id, 'ik'), exist_ok=True)

    # Get dynamic trial file
    trial_file = os.path.join('..', 'data', participant_id, 'dynamic', 'walk.trc')

    # Run IK on trial
    # -------------------------------------------------------------------------

    # Create a folder for the condition and trial
    os.makedirs(os.path.join('..', 'data', participant_id, 'ik'), exist_ok=True)

    # Set model
    ikTool[participant_id].set_model_file(os.path.join('..', 'data', participant_id, 'scaling',
                                                       f'{participant_id}_scaledModelAdjusted_3d.osim'))

    # Set task set (consistent for all trials)
    for taskInd in range(ikTaskSet.getSize()):
        ikTool[participant_id].getIKTaskSet().adoptAndAppend(ikTaskSet.get(taskInd))

    # Set to report marker locations
    ikTool[participant_id].set_report_marker_locations(True)

    # Set the marker file (relative to setup file location)
    ikTool[participant_id].setMarkerDataFileName(trial_file)

    # Set times
    ikTool[participant_id].setStartTime(osim.TimeSeriesTableVec3(trial_file).getIndependentColumn()[0])
    ikTool[participant_id].setEndTime(osim.TimeSeriesTableVec3(trial_file).getIndependentColumn()[-1])

    # Set output filename (relative to setup file location)
    ikTool[participant_id].setOutputMotionFileName(os.path.join('..', 'data', participant_id, 'ik', f'{participant_id}_walk_ik_3d.mot'))

    # Save IK tool to file
    ikTool[participant_id].printToXML('ikSetup.xml')

    # Bring the tool back in and run it (this seems to avoid Python kernel crashing)
    ikRun = osim.InverseKinematicsTool('ikSetup.xml')
    ikRun.run()

    # Rename supplementary marker outputs
    shutil.move('_ik_marker_errors.sto',
                os.path.join('..', 'data', participant_id, 'ik', f'{participant_id}_walk_ikMarkerErrors_3d.sto'))
    shutil.move('_ik_model_marker_locations.sto',
                os.path.join('..', 'data', participant_id, 'ik', f'{participant_id}_walk_ikModelMarkerLocations_3d.sto'))
    shutil.move('ikSetup.xml',
                os.path.join('..', 'data', participant_id, 'ik', f'{participant_id}_walk_ikSetup_3d.xml'))

    # Run 2D version of IK
    # -------------------------------------------------------------------------

    # The first step is to create an appropriate 2D version of the marker data
    # All z-data therefore aligns to the locked position on the scaled model

    # Read in TRC data
    trc_data = osim.TimeSeriesTableVec3(trial_file)

    # Read in 2D scaled model
    model_2d = osim.Model(os.path.join('..', 'data', participant_id, 'scaling',
                                       f'{participant_id}_scaledModelAdjusted_2d.osim'))

    # Get the markers from the 2D model
    marker_names = [model_2d.getMarkerSet().get(ii).getName() for ii in range(model_2d.getMarkerSet().getSize())]

    # Remove any markers in trc data not in model markerset
    for marker in trc_data.getColumnLabels():
        if marker not in marker_names:
            trc_data.removeColumn(marker)

    # Flatten marker table
    trc_flat = trc_data.flatten()

    # Initialise model state to position (using default position)
    model_state = model_2d.initSystem()
    model_2d.realizePosition(model_state)

    # Adjust the z-axis marker data in the trc file
    for marker_ax in trc_flat.getColumnLabels():
        if marker_ax.endswith('_3'):  # z-axis data
            # Get the marker name
            marker_name = marker_ax[:-2]
            # Get the marker location in ground frame
            marker_loc_ground = model_2d.getMarkerSet().get(marker_name).getLocationInGround(model_state)
            # Remove columns to replace (this is required for consistent ordering when packing)
            # Need to get the relevant data first though
            marker_x_data = trc_flat.getDependentColumn(marker_name + '_1').to_numpy()
            trc_flat.removeColumn(marker_name + '_1')
            marker_y_data = trc_flat.getDependentColumn(marker_name + '_2').to_numpy()
            trc_flat.removeColumn(marker_name + '_2')
            trc_flat.removeColumn(marker_name + '_3')
            # Append new columns to replace (including the unchanged data that was removed)
            # Includes multiplying by 1000 to convert from m to mm
            trc_flat.appendColumn(marker_name + '_1', osim.Vector().createFromMat(marker_x_data))
            trc_flat.appendColumn(marker_name + '_2', osim.Vector().createFromMat(marker_y_data))
            trc_flat.appendColumn(marker_ax, osim.Vector().createFromMat(np.ones(trc_flat.getNumRows()) * (marker_loc_ground.get(2) * 1000)))

    # Pack table back into Vec3 format
    trc_data_2d = trc_flat.packVec3()

    # Write to file
    osim.TRCFileAdapter().write(trc_data_2d, f'{trial_file[:-4]}_2d.trc')

    # Re-use IK tool to get 2D kinematics
    # Only some parameters need to be updated

    # Set model
    ikTool[participant_id].set_model_file(os.path.join('..', 'data', participant_id, 'scaling',
                                                       f'{participant_id}_scaledModelAdjusted_2d.osim'))

    # Set the marker file (relative to setup file location)
    ikTool[participant_id].setMarkerDataFileName(f'{trial_file[:-4]}_2d.trc')

    # Set output filename (relative to setup file location)
    ikTool[participant_id].setOutputMotionFileName(os.path.join('..', 'data', participant_id, 'ik', f'{participant_id}_walk_ik_2d.mot'))

    # Save IK tool to file
    ikTool[participant_id].printToXML('ikSetup.xml')

    # Bring the tool back in and run it (this seems to avoid Python kernel crashing)
    ikRun = osim.InverseKinematicsTool('ikSetup.xml')
    ikRun.run()

    # Rename supplementary marker outputs
    shutil.move('_ik_marker_errors.sto',
                os.path.join('..', 'data', participant_id, 'ik', f'{participant_id}_walk_ikMarkerErrors_2d.sto'))
    shutil.move('_ik_model_marker_locations.sto',
                os.path.join('..', 'data', participant_id, 'ik', f'{participant_id}_walk_ikModelMarkerLocations_2d.sto'))
    shutil.move('ikSetup.xml',
                os.path.join('..', 'data', participant_id, 'ik', f'{participant_id}_walk_ikSetup_2d.xml'))


# Run combined marker and coordinate tracking
# -------------------------------------------------------------------------
def run_tracking(participant_id, model_variant='3d'):

    # Organise and set-up files
    # -------------------------------------------------------------------------

    # Create directory for files
    os.makedirs(os.path.join('..', 'data', participant_id, 'exp'), exist_ok=True)

    # Create directory to store model variant specific results
    os.makedirs(os.path.join('..', 'data', participant_id, 'exp', model_variant), exist_ok=True)

    # Run marker tracking on trial
    # -------------------------------------------------------------------------

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..', 'data', participant_id, 'exp', model_variant))

    # Copy external loads file to simulation directory
    shutil.copyfile(os.path.join('..', '..', 'dynamic', 'walk_grf.mot'), 'walk_grf.mot')
    shutil.copyfile(os.path.join('..', '..', 'dynamic', 'walk_grf.xml'), 'walk_grf.xml')

    # Copy TRC file to simulation directory
    if model_variant == '3d':
        shutil.copyfile(os.path.join('..', '..', 'dynamic', 'walk.trc'),f'walk_{model_variant}.trc')
    elif model_variant == '2d':
        shutil.copyfile(os.path.join('..', '..', 'dynamic', f'walk_{model_variant}.trc'), f'walk_{model_variant}.trc')

    # Copy IK file to simulation directory
    shutil.copyfile(os.path.join('..', '..', 'ik', f'{participant_id}_walk_ik_{model_variant}.mot'),
                    f'{participant_id}_walk_ik_{model_variant}.mot')

    # Copy model file to simulation directory
    shutil.copyfile(os.path.join('..', '..', 'scaling', f'{participant_id}_scaledModelAdjusted_{model_variant}.osim'),
                    f'{participant_id}_scaledModelAdjusted_{model_variant}.osim')

    # Copy plug in gait coordinate data to simulation directory
    shutil.copyfile(os.path.join('..', '..', 'dynamic', 'walk_PiG-angles.mot'), f'walk_PiG_angles_{model_variant}.mot')

    # Set-up model for simulation
    # -------------------------------------------------------------------------

    # Use a processor to remove muscles from model
    model_proc = osim.ModelProcessor(f'{participant_id}_scaledModelAdjusted_{model_variant}.osim')
    model_proc.append(osim.ModOpRemoveMuscles())
    osim_model = model_proc.process()

    # Add coordinate actuators to model
    # Depending on if model is 2d or 3d variant not all actuators will need to be created
    model_coordinates = [osim_model.getCoordinateSet().get(ii).getName() for ii in range(osim_model.getNumCoordinates())]
    # Loop through coordinates
    for coordinate in actForces:
        # Check if in model coordinates
        if coordinate in model_coordinates:
            # Create actuator
            actu = osim.CoordinateActuator()
            # Set name
            actu.setName(f'{coordinate}_{actForces[coordinate]["actuatorType"]}')
            # Set coordinate
            actu.setCoordinate(osim_model.updCoordinateSet().get(coordinate))
            # Set optimal force
            actu.setOptimalForce(actForces[coordinate]['optForce'])
            # Set min and max control
            actu.setMinControl(np.inf * -1)
            actu.setMaxControl(np.inf * 1)
            # Append to model force set
            osim_model.updForceSet().cloneAndAppend(actu)

    # Clear the contact geometry as it won't be used
    osim_model.updContactGeometrySet().clearAndDestroy()

    # Remove contact forces, and passive subtalar and/or toe damping forces given joint will be welded
    remove_force = []
    if model_variant == '3d':
        for forceInd in range(osim_model.getForceSet().getSize()):
            if osim_model.updForceSet().get(forceInd).getName().lower().startswith('contact') or \
                    osim_model.updForceSet().get(forceInd).getName().startswith('PassiveToeMoment') or \
                    osim_model.updForceSet().get(forceInd).getName().startswith('PassiveSubtalarDamping'):
                remove_force.append(forceInd)
    elif model_variant == '2d':
        for forceInd in range(osim_model.getForceSet().getSize()):
            if osim_model.updForceSet().get(forceInd).getName().lower().startswith('contact') or \
                    osim_model.updForceSet().get(forceInd).getName().startswith('mtp_damping'):
                remove_force.append(forceInd)
    counter = 0
    for remove in remove_force:
        osim_model.updForceSet().remove(remove - counter)
        counter += 1

    # Finalise model connections
    osim_model.finalizeConnections()

    # Print model to file
    osim_model.printToXML(f'{participant_id}_exp-tracking_{model_variant}.osim')

    # Adjust kinematic and GRF data from experimental processing
    # -------------------------------------------------------------------------

    # Load in the kinematic data
    kinematics_data = osim.Storage(f'{participant_id}_walk_ik_{model_variant}.mot')

    # Create a copy of the kinematics data to alter the column labels in
    states_data = osim.Storage(f'{participant_id}_walk_ik_{model_variant}.mot')

    # Get the column headers for the storage file
    angle_names = kinematics_data.getColumnLabels()

    # Get the corresponding full paths from the model to rename the angles in the kinematics file
    for angInd in range(angle_names.getSize()):
        currAngle = angle_names.get(angInd)
        if currAngle != 'time':
            # Try getting the full path to coordinate
            # This may fail due to their being marker data included in these files
            try:
                # Loo for full coordinate path
                fullPath = osim_model.updCoordinateSet().get(currAngle).getAbsolutePathString() + '/value'
                # Set angle name appropriately using full path
                angle_names.set(angInd, fullPath)
            except:
                # Print out that current column isn't a coordinate
                print(f'{currAngle} not a coordinate...skipping name conversion...')
                # Set to the same as originaly
                angle_names.set(angInd, currAngle)

    # Set the states storage object to have the updated column labels
    states_data.setColumnLabels(angle_names)

    # Convert from IK default of degrees to radians
    osim_model.initSystem()
    osim_model.getSimbodyEngine().convertDegreesToRadians(states_data)

    # Write the states storage object to file
    states_data.printToXML(f'{participant_id}_ik-walk-coordinates_{model_variant}.sto')

    # Check whether GRF data needs updating for 2D model
    # For 2D model the GRFs need to be updated to eliminate z-axis data
    # In lieu of accurate COP tracking in Z-direction this is normalised to sit underneath the ankle joint centre for z-data
    if model_variant == '2d':
        # Load in GRF data
        grf_data = osim.TimeSeriesTable('walk_grf.mot')
        ex_loads = osim.ExternalLoads('walk_grf.xml', True)
        # Load in IK data for Z-COP reference
        ik_data = osim.TimeSeriesTable(f'{participant_id}_ik-walk-coordinates_{model_variant}.sto')
        # Load in model to pose from IK data
        model_2d = osim.Model(f'{participant_id}_scaledModelAdjusted_{model_variant}.osim')
        model_2d_state = model_2d.initSystem()
        # Get ankle joint position across walking trial
        r_ankle = []
        l_ankle = []
        for ii in range(ik_data.getNumRows()):
            # Pose the model using IK coordinates
            for state_name in ik_data.getColumnLabels():
                model_2d.setStateVariableValue(model_2d_state, state_name, ik_data.getDependentColumn(state_name).to_numpy()[ii])
            model_2d.realizePosition(model_2d_state)
            # Get ankle joint positions at these points in the z-direction
            # Based on origin of joint in ground frame
            r_ankle.append(model_2d.getJointSet().get('ankle_r').getChildFrame().getPositionInGround(model_2d_state).get(2))
            l_ankle.append(model_2d.getJointSet().get('ankle_l').getChildFrame().getPositionInGround(model_2d_state).get(2))
        # Get average position of ankle joint centres
        r_midpoint = np.mean(r_ankle)
        l_midpoint = np.mean(l_ankle)
        # Figure out which column to apply right and left mid-point to based on external loads
        cop_vals = {}
        for load_ind in range(ex_loads.getSize()):
            if ex_loads.get(load_ind).getAppliedToBodyName() == 'calcn_r':
                cop_vals[ex_loads.get(load_ind).get_point_identifier() + 'z'] = r_midpoint
            elif ex_loads.get(load_ind).getAppliedToBodyName() == 'calcn_l':
                cop_vals[ex_loads.get(load_ind).get_point_identifier() + 'z'] = l_midpoint
        # Get GRF data column labels
        grf_columns = list(grf_data.getColumnLabels())
        # Loop through row indices and reset values
        for row_ind in range(grf_data.getNumRows()):
            # First check if all values are zero, as then the row can be skipped
            if not all(grf_data.getRowAtIndex(row_ind).to_numpy() == 0):
                # Create the array of zeros
                replace_data = np.zeros(len(grf_columns))
                # Loop through columns
                for col in grf_columns:
                    # First check if it is a replacement COP value
                    if col in cop_vals.keys():
                        replace_data[grf_columns.index(col)] = cop_vals[col]
                    # Check for x or y data that needs to be retained
                    elif col.endswith('x') or col.endswith('y'):
                        replace_data[grf_columns.index(col)] = grf_data.getDependentColumn(col).to_numpy()[row_ind]
                    # Check for z-axis force data and add a small amount of noise
                    # This is to ensure that normalised GRF tracking doesn't fail later on if desired
                    # This will be random noise between -1 and +1
                    elif col.endswith('vz'):
                        replace_data[grf_columns.index(col)] = np.random.normal(loc=0.0, scale=1)
                # Set the row in the grf data to the replacement values
                grf_data.setRowAtIndex(row_ind, osim.RowVector().createFromMat(replace_data))
        # Write the new grf data to file
        osim.STOFileAdapter().write(grf_data, 'walk_grf.mot')

    # Adjust plug in gait coordinates to work with tracking simulation
    # -------------------------------------------------------------------------

    # Read in plug in gait coordinates
    pig_coordinates = osim.TimeSeriesTable(f'walk_PiG_angles_{model_variant}.mot')

    # Rename column labels to model states
    new_labels = []
    for col in pig_coordinates.getColumnLabels():
        new_labels.append(osim_model.getCoordinateSet().get(col).getAbsolutePathString()+'/value')
    pig_coordinates.setColumnLabels(new_labels)

    # Set in degrees meta-data
    pig_coordinates.addTableMetaDataString('inDegrees','no')

    # Write to file
    osim.STOFileAdapter().write(pig_coordinates, f'walk_PiG_states_{model_variant}.sto')

    # Set up tracking simulation
    # -------------------------------------------------------------------------

    # Create tracking tool
    track = osim.MocoTrack()
    track.setName(f'{participant_id}_exp-tracking_{model_variant}')

    # Create model processor
    track_model_proc = osim.ModelProcessor(f'{participant_id}_exp-tracking_{model_variant}.osim')

    # Append external loads
    track_model_proc.append(osim.ModOpAddExternalLoads('walk_grf.xml'))

    # Weld locked joints for marker tracking
    weld_joints = osim.StdVectorString()
    # MTP joints in both models
    weld_joints.append('mtp_r')
    weld_joints.append('mtp_l')
    # Subtalar only in 3D model
    if model_variant == '3d':
        weld_joints.append('subtalar_r')
        weld_joints.append('subtalar_l')
    track_model_proc.append(osim.ModOpReplaceJointsWithWelds(weld_joints))

    # Set model in tool
    track.setModel(track_model_proc)

    # Set the markers reference file
    track.setMarkersReferenceFromTRC(f'walk_{model_variant}.trc')

    # Set global tracking weight
    track.set_markers_global_tracking_weight(globalMarkerTrackingWeight)

    # Set marker weights
    tracking_weights = osim.MocoWeightSet()
    for marker in ikTaskSetParams:
        if ikTaskSetParams[marker]['weight'] != 0:
            tracking_weights.cloneAndAppend(osim.MocoWeight(marker, ikTaskSetParams[marker]['weight']))
    track.set_markers_weight_set(tracking_weights)

    # Set the coordinates reference file
    table_proc = osim.TableProcessor(f'walk_PiG_states_{model_variant}.sto')
    track.setStatesReference(table_proc)

    # Set global tracking weight
    track.set_states_global_tracking_weight(globalStateTrackingWeight)

    # Set to ignore unused columns
    track.set_allow_unused_references(True)

    # Set the timings
    track.set_initial_time(osim.TimeSeriesTable(f'{participant_id}_ik-walk-coordinates_{model_variant}.sto').getIndependentColumn()[0])
    track.set_final_time(osim.TimeSeriesTable(f'{participant_id}_ik-walk-coordinates_{model_variant}.sto').getIndependentColumn()[-1])

    # Initialise to a Moco study and problem to finalise
    # -------------------------------------------------------------------------

    # Get study and problem
    study = track.initialize()
    problem = study.updProblem()

    # Update control effort goal

    # Get a reference to the MocoControlCost goal and set parameters
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'))
    effort.setWeight(globalControlEffortWeight)
    effort.setExponent(2)

    # Set bounds
    # -------------------------------------------------------------------------

    # Set joint coordinate values to model limits
    track_model = track_model_proc.process()
    track_model.initSystem()
    for coord_ind in range(track_model.getCoordinateSet().getSize()):
        # Get coordinate name and path
        coord_name = track_model.updCoordinateSet().get(coord_ind).getName()
        coord_path = track_model.updCoordinateSet().get(coord_ind).getAbsolutePathString()
        # Skip setting bounds for PFJ constrained knee angle
        if 'beta' not in coord_path:
            # Set bounds in problem to model coordinate bounds
            problem.setStateInfo(f'{coord_path}/value',
                                 [track_model.getCoordinateSet().get(coord_name).getRangeMin(),
                                  track_model.getCoordinateSet().get(coord_name).getRangeMax()],
                                 [], [])

    # Define and configure the solver
    # -------------------------------------------------------------------------
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

    # Set solver options
    solver.set_optim_max_iterations(1000)
    solver.set_num_mesh_intervals(meshInterval)
    solver.set_optim_constraint_tolerance(1e-3)
    solver.set_optim_convergence_tolerance(1e-3)

    # Reset problem to avoid any issues
    solver.resetProblem(problem)

    # Get the initial guess to manipulate
    initial_guess = solver.createGuess()

    # Resample time to match states data
    ik_states = osim.TimeSeriesTable(f'{participant_id}_ik-walk-coordinates_{model_variant}.sto')
    initial_guess.resampleWithNumTimes(ik_states.getNumRows())

    # Fill initial guess with coordinates from IK data
    for col in ik_states.getColumnLabels():
        if col in initial_guess.exportToStatesTable().getColumnLabels():
            initial_guess.setState(col, ik_states.getDependentColumn(col).to_numpy())

    # Set the guess in solver
    solver.setGuess(initial_guess)

    # Reset problem to finalise
    solver.resetProblem(problem)

    # Solve the problem
    # -------------------------------------------------------------------------
    solution = study.solve()

    # # Option to visualise solution
    # study.visualize(solution)

    # Save files and finalize
    # -------------------------------------------------------------------------

    # Write solution to file
    if solution.isSealed():
        solution.unseal()
    solution.write(f'{participant_id}_exp-tracking-solution_{model_variant}.sto')

    # Remove initial tracked states and markers file
    os.remove(f'{participant_id}_exp-tracking_{model_variant}_tracked_markers.sto')
    os.remove(f'{participant_id}_exp-tracking_{model_variant}_tracked_states.sto')

    # Return to home directory
    os.chdir(home_dir)

    # Create a folder of data to use with muscle redundancy solver
    # -------------------------------------------------------------------------

    #  Create the folder to store data in
    os.makedirs(os.path.join('..', 'data', participant_id, 'mrs', model_variant), exist_ok=True)

    # Get and copy across the base scaled version of model
    os.makedirs(os.path.join('..', 'data', participant_id, 'model'), exist_ok=True)
    scaled_model = osim.Model(os.path.join('..', 'data', participant_id, 'scaling',
                                           f'{participant_id}_scaledModelAdjusted_{model_variant}.osim'))
    scaled_model.initSystem()
    scaled_model.printToXML(os.path.join('..', 'data', participant_id, 'model', f'{participant_id}_model_{model_variant}.osim'))

    # Get the Moco trajectory
    trajectory = osim.MocoTrajectory(os.path.join('..', 'data', participant_id, 'exp', model_variant,
                                                  f'{participant_id}_exp-tracking-solution_{model_variant}.sto'))

    # Export kinematic values table and edit to format useable by muscle redundancy solver
    # Converts back to degrees and simplifies column labels
    values_table = trajectory.exportToValuesTable()
    values_table.setColumnLabels([col.split('/')[3] for col in values_table.getColumnLabels()])
    scaled_model.getSimbodyEngine().convertRadiansToDegrees(values_table)
    osim.STOFileAdapter().write(values_table,
                                os.path.join('..', 'data', participant_id, 'mrs', model_variant,
                                             f'{participant_id}_walk_ik_{model_variant}.mot'))

    # Convert tracking solution controls to ID like file
    tracking_controls = trajectory.exportToControlsTable()

    # Loop through column labels to update data
    # This process converts the control to a torque and then deletes the original data
    for col in tracking_controls.getColumnLabels():
        # Check for pelvis translations
        if any([ii in col for ii in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']]):
            # Create label by isolating coordinate and adding force label
            new_label = col.split('/')[-1].replace('_residual','_force')
            # Convert control signal to torque
            force_val = actForces[col.split('/')[-1].replace('_residual','')]['optForce']
            new_vals = tracking_controls.getDependentColumn(col).to_numpy() * force_val
        # Check for pelvis rotations
        elif any([ii in col for ii in ['pelvis_list', 'pelvis_tilt', 'pelvis_rotation']]):
            # Create label by isolating coordinate and adding moment label
            new_label = col.split('/')[-1].replace('_residual','_moment')
            # Convert control signal to torque
            force_val = actForces[col.split('/')[-1].replace('_residual','')]['optForce']
            new_vals = tracking_controls.getDependentColumn(col).to_numpy() * force_val
        # Otherwise it will be a torque actuated joint
        else:
            # Create label by isolating coordinate and adding moment label
            new_label = col.split('/')[-1].replace('_torque', '_moment')
            # Convert control signal to torque
            force_val = actForces[col.split('/')[-1].replace('_torque', '')]['optForce']
            new_vals = tracking_controls.getDependentColumn(col).to_numpy() * force_val
        # Add column to controls table
        tracking_controls.appendColumn(new_label, osim.Vector().createFromMat(new_vals))
        # Remove original column
        tracking_controls.removeColumn(col)

    # Filter signals before writing to file using table processor
    table_proc = osim.TableProcessor(tracking_controls)
    table_proc.append(osim.TabOpLowPassFilter(kinematicFiltFreq))
    torque_signals = table_proc.process()

    # Filtering pads time frame out so need to trim new table
    torque_signals.trim(tracking_controls.getIndependentColumn()[0],tracking_controls.getIndependentColumn()[-1])

    # Write newly created ID like data to file
    osim.STOFileAdapter().write(torque_signals,
                                os.path.join('..', 'data', participant_id, 'mrs', model_variant,
                                             f'{participant_id}_walk_id_{model_variant}.sto'))


# Adjust vertical position of model in kinematics based on static trial data
# Position is adjusted based on desired sphere penetration in static trial motion
# TODO: adjust this for new dataset...needed?
# -------------------------------------------------------------------------
def adjust_model_position(participant_id, model_variant='3d', sphere_penetration_per = 0.00):

    # Determine vertical offset needed to apply to data
    # -------------------------------------------------------------------------

    # Read in scaled model
    if model_variant == '3d':
        osim_model = osim.Model(os.path.join('..', 'data', participant_id, 'scaling',
                                             f'{participant_id}_scaledModelAdjusted.osim'))
    elif model_variant == '2d':
        osim_model = osim.Model(os.path.join('..', 'data', participant_id, 'scaling',
                                             f'{participant_id}_scaledModelAdjusted_{model_variant}.osim'))

    # Read in static motion data
    static_motion = osim.TimeSeriesTable(os.path.join('..', 'data', participant_id, 'scaling',
                                                      f'{participant_id}_staticMotion_3d.mot'))

    # Initialise the model state
    model_state = osim_model.initSystem()

    # Get model state variable names
    model_state_names = [osim_model.getStateVariableNames().get(ii) for ii in range(osim_model.getStateVariableNames().size())]

    # Get contact sphere names from model
    contact_spheres = []
    for geom_ind in range(osim_model.getContactGeometrySet().getSize()):
        # Check for contact sphere component
        if 'sphere' in osim_model.getContactGeometrySet().get(geom_ind).getConcreteClassName().lower():
            # Get contact sphere name and append to list
            contact_spheres.append(osim_model.getContactGeometrySet().get(geom_ind).getName())

    # Get sphere parameters from model
    sphere_params = {sphere: {'body': osim.ContactSphere.safeDownCast(osim_model.getContactGeometrySet().get(sphere)).getBody().getName(),
                              'loc': osim.ContactSphere.safeDownCast(osim_model.getContactGeometrySet().get(sphere)).get_location(),
                              'radius': osim.ContactSphere.safeDownCast(osim_model.getContactGeometrySet().get(sphere)).getRadius()}
                     for sphere in contact_spheres}

    # Create arrays to store contact sphere vertical edge positions from static motion
    vert_sphere_pos = {sphere: 0 for sphere in contact_spheres}

    # Set model states from motion. Only values are done here as we realise to position
    for state_name in static_motion.getColumnLabels():
        if state_name.endswith('/value') and state_name in model_state_names:
            osim_model.setStateVariableValue(model_state, state_name, static_motion.getDependentColumn(state_name).to_numpy()[0])

    # Realize model to position stage
    osim_model.realizePosition(model_state)

    # Extract the vertical position of the spheres in the ground frame
    for sphere in contact_spheres:
        # Get the vertical position of the sphere in the ground frame
        # Subtract the radius to get the edge of the spheres position
        vert_sphere_pos[sphere] = osim_model.getBodySet().get(sphere_params[sphere]['body']).findStationLocationInGround(
            model_state, sphere_params[sphere]['loc']).get(1) - sphere_params[sphere]['radius']

    # Identify the offset required to have the contact sphere penetrate the ground by the desired percentage
    desired_value = {sphere: (sphere_params[sphere]['radius'] * 2) * sphere_penetration_per * -1 for sphere in contact_spheres}

    # Calculate the offset required to achieve the desired value
    offset_amount = {sphere: desired_value[sphere] - vert_sphere_pos[sphere] for sphere in contact_spheres}

    # Calculate average offset amount to apply to vertical pelvis position
    offset_avg = np.mean([offset_amount[sphere] for sphere in contact_spheres])

    # Read in tracking data and apply offset
    # -------------------------------------------------------------------------

    # Set tracking file solution
    tracking_file = os.path.join('..', 'data', participant_id, 'exp', model_variant,
                                 f'{participant_id}_exp-tracking-solution_{model_variant}.sto')

    # Load in solution trajectory
    solution_data = osim.MocoTrajectory(tracking_file)

    # Solution data for pelvis_ty tends to skew and not be periodic
    # As a first step, fit a constrained sine wave to ensure similar periodicity in vertical translation
    pelvis_ty_state = osim_model.getCoordinateSet().get('pelvis_ty').getAbsolutePathString() + '/value'
    ty = solution_data.getState(pelvis_ty_state).to_numpy()
    time = solution_data.getTime().to_numpy()
    params = fit_sine_with_constraint(time, ty)
    ty_fit = sine_wave(time, *params)
    # plt.plot(time, ty)
    # plt.plot(time, ty_fit)

    # Next, adjust vertical pelvis position by average offset
    solution_data.setState(pelvis_ty_state, ty_fit + offset_avg)

    # Write adjusted data to file
    solution_data.write(os.path.join('..', 'data', participant_id, 'exp', model_variant,
                                     f'{participant_id}_exp-tracking-solution-adjusted_{model_variant}.sto'))

    # Adjust tracking data and contact sphere parameters further to optimise GRF prediction
    # -------------------------------------------------------------------------

    # # Read in GRF and external loads data
    # grf_data = osim.TimeSeriesTable(os.path.join('..', 'data', participant_id, 'exp', model_variant, 'walk_grf.mot'))
    # ex_loads = osim.ExternalLoads(os.path.join('..', 'data', participant_id, 'exp', model_variant, 'walk_grf.xml'), True)
    # grf_time = np.array(grf_data.getIndependentColumn())
    #
    # # Interpolate solution data to the same number of times as the cropped GRF data
    # solution_data.resampleWithNumTimes(grf_data.getNumRows())
    # solution_data.getNumTimes()
    #
    # # Get the left and right forces from trial data to optimise against
    # limb_labels = {'left': [], 'right': []}
    # for load_ind in range(ex_loads.getSize()):
    #     if ex_loads.get(load_ind).getAppliedToBodyName() == 'calcn_r':
    #         limb_labels['right'].append(ex_loads.get(load_ind).get_force_identifier())
    #     elif ex_loads.get(load_ind).getAppliedToBodyName() == 'calcn_l':
    #         limb_labels['left'].append(ex_loads.get(load_ind).get_force_identifier())
    # right_grf = [np.vstack([grf_data.getDependentColumn(col + 'x').to_numpy() for col in limb_labels['right']]).sum(axis=0),
    #              np.vstack([grf_data.getDependentColumn(col + 'y').to_numpy() for col in limb_labels['right']]).sum(axis=0),
    #              np.vstack([grf_data.getDependentColumn(col + 'z').to_numpy() for col in limb_labels['right']]).sum(axis=0)]
    # left_grf = [np.vstack([grf_data.getDependentColumn(col + 'x').to_numpy() for col in limb_labels['left']]).sum(axis=0),
    #             np.vstack([grf_data.getDependentColumn(col + 'y').to_numpy() for col in limb_labels['left']]).sum(axis=0),
    #             np.vstack([grf_data.getDependentColumn(col + 'z').to_numpy() for col in limb_labels['left']]).sum(axis=0)]
    #
    # # Resample data to match solution time
    # new_time = np.linspace(grf_time[0], grf_time[-1], solution_data.getNumTimes())
    # left_grf_i = []
    # right_grf_i = []
    # for ii in range(3):
    #     interpolator_l = interp1d(grf_time, left_grf[ii], kind='linear')
    #     interpolator_r = interp1d(grf_time, right_grf[ii], kind='linear')
    #     left_grf_i.append(interpolator_l(new_time))
    #     right_grf_i.append(interpolator_r(new_time))
    #
    # # Extract sphere forces from solution data states
    #
    # # Get the names of left and right spheres to extract forces from
    # # TODO: ensure this works for 2d and 3d models
    # left_spheres = [osim_model.getForceSet().get(ff).getName() for ff in range(osim_model.getForceSet().getSize()) \
    #                 if '_L' in osim_model.getForceSet().get(ff).getName() and osim_model.getForceSet().get(ff).getName().startswith('Contact')]
    # right_spheres = [osim_model.getForceSet().get(ff).getName() for ff in range(osim_model.getForceSet().getSize()) \
    #                 if '_R' in osim_model.getForceSet().get(ff).getName() and osim_model.getForceSet().get(ff).getName().startswith('Contact')]
    #
    # # Set matching arrays to store estimated GRFs in
    # left_grf_est = [np.zeros(solution_data.getNumTimes()), np.zeros(solution_data.getNumTimes()), np.zeros(solution_data.getNumTimes())]
    # right_grf_est = [np.zeros(solution_data.getNumTimes()), np.zeros(solution_data.getNumTimes()), np.zeros(solution_data.getNumTimes())]
    #
    # # Get values and speeds tables
    # values_table = solution_data.exportToValuesTable()
    # speeds_table = solution_data.exportToSpeedsTable()
    #
    # # Loop through time instances
    # for time_ind in range(solution_data.getNumTimes()):
    #     # Loop through value and speed states
    #     for value_state in values_table.getColumnLabels():
    #         osim_model.setStateVariableValue(model_state, value_state, values_table.getDependentColumn(value_state).to_numpy()[time_ind])
    #     for speed_state in speeds_table.getColumnLabels():
    #         osim_model.setStateVariableValue(model_state, speed_state, speeds_table.getDependentColumn(speed_state).to_numpy()[time_ind])
    #     # Realize model to velocity stage
    #     osim_model.realizeVelocity(model_state)
    #     # Extract sphere forces in XYZ axes and allocate estimates to array
    #     # Index 1 of sphere force output is Vec3 of forces
    #     for axis_ind in range(3):
    #         left_grf_est[axis_ind][time_ind] = np.sum([osim.SmoothSphereHalfSpaceForce().safeDownCast(
    #             osim_model.getForceSet().get(sphere)).getSphereForce(model_state).get(1).get(
    #             axis_ind) for sphere in left_spheres])
    #         right_grf_est[axis_ind][time_ind] = np.sum([osim.SmoothSphereHalfSpaceForce().safeDownCast(
    #             osim_model.getForceSet().get(sphere)).getSphereForce(model_state).get(1).get(
    #             axis_ind) for sphere in right_spheres])
    #
    # # # Review experimental GRF vs. estimates
    # # # Showing vertical force here
    # # plt.plot(left_grf_i[1], ls='-', c='b')
    # # plt.plot(right_grf_i[1], ls='-', c='g')
    # # plt.plot(left_grf_est[1], ls=':', c='b')
    # # plt.plot(right_grf_est[1], ls=':', c='g')
    #
    # # Repeat process but half the stiffness of the spheres in the model
    #
    # # Half the sphere stiffness
    # for sphere in left_spheres:
    #     osim.SmoothSphereHalfSpaceForce().safeDownCast(
    #         osim_model.getForceSet().get(sphere)).set_stiffness(osim.SmoothSphereHalfSpaceForce().safeDownCast(
    #         osim_model.getForceSet().get(sphere)).get_stiffness() / 2)
    # for sphere in right_spheres:
    #     osim.SmoothSphereHalfSpaceForce().safeDownCast(
    #         osim_model.getForceSet().get(sphere)).set_stiffness(osim.SmoothSphereHalfSpaceForce().safeDownCast(
    #         osim_model.getForceSet().get(sphere)).get_stiffness() / 2)
    #
    # # Finalize model connections and re-initialise state
    # osim_model.finalizeConnections()
    # model_state = osim_model.initSystem()
    #
    # # Set matching arrays to store estimated GRFs in
    # left_grf_est_2 = [np.zeros(solution_data.getNumTimes()), np.zeros(solution_data.getNumTimes()), np.zeros(solution_data.getNumTimes())]
    # right_grf_est_2 = [np.zeros(solution_data.getNumTimes()), np.zeros(solution_data.getNumTimes()), np.zeros(solution_data.getNumTimes())]
    #
    # # Loop through time instances
    # for time_ind in range(solution_data.getNumTimes()):
    #     # Loop through value and speed states
    #     for value_state in values_table.getColumnLabels():
    #         osim_model.setStateVariableValue(model_state, value_state, values_table.getDependentColumn(value_state).to_numpy()[time_ind])
    #     for speed_state in speeds_table.getColumnLabels():
    #         osim_model.setStateVariableValue(model_state, speed_state, speeds_table.getDependentColumn(speed_state).to_numpy()[time_ind])
    #     # Realize model to velocity stage
    #     osim_model.realizeVelocity(model_state)
    #     # Extract sphere forces in XYZ axes and allocate estimates to array
    #     # Index 1 of sphere force output is Vec3 of forces
    #     for axis_ind in range(3):
    #         left_grf_est_2[axis_ind][time_ind] = np.sum([osim.SmoothSphereHalfSpaceForce().safeDownCast(
    #             osim_model.getForceSet().get(sphere)).getSphereForce(model_state).get(1).get(
    #             axis_ind) for sphere in left_spheres])
    #         right_grf_est_2[axis_ind][time_ind] = np.sum([osim.SmoothSphereHalfSpaceForce().safeDownCast(
    #             osim_model.getForceSet().get(sphere)).getSphereForce(model_state).get(1).get(
    #             axis_ind) for sphere in right_spheres])
    #
    # # Review experimental GRF vs. estimates
    # # Showing vertical force here
    # plt.plot(left_grf_i[1], ls='-', c='b')
    # plt.plot(right_grf_i[1], ls='-', c='g')
    # # plt.plot(left_grf_est[1] / 100, ls=':', c='b')
    # # plt.plot(right_grf_est[1] / 100, ls=':', c='g')
    # plt.plot(left_grf_est_2[1], ls='--', c='b')
    # plt.plot(right_grf_est_2[1], ls='--', c='g')
    #
    # # Print out this test model now that it's reached something reasonable
    # osim_model.printToXML(os.path.join('..', 'data', participant_id, 'scaling',
    #                                    f'{participant_id}_scaledModelAdjusted_{model_variant}_sphereProperties.osim'))
    #
    # # est_contact = osim.TimeSeriesTable(os.path.join('..', 'SUBJ01_2d_ForceReporter_forces.sto'))
    # # left_vert_labels = [col for col in est_contact.getColumnLabels() if '_L' in col and col.endswith('Sphere.force.Y')]
    # # vert_force = np.vstack([est_contact.getDependentColumn(col).to_numpy() for col in left_vert_labels]).mean(axis=0)
    # # plt.plot(vert_force)


# =========================================================================
# Process data
# =========================================================================

if __name__ == '__main__':

    # Run scaling function
    # -------------------------------------------------------------------------
    if runScaling:
        for participant in participant_list:
            run_scaling(participant)

    # Run IK function
    # -------------------------------------------------------------------------
    if runIK:
        for participant in participant_list:
            run_ik(participant)

    # Run experimental data tracking
    # -------------------------------------------------------------------------
    if runTracking:
        print('TODO: add in tracking function process...')

    # Run model position adjustment in tracking data
    # -------------------------------------------------------------------------
    if runContactAdjustment:
        print('TODO: add in adjustment function process...')

    # Print confirmation
    # -------------------------------------------------------------------------
    print('Data processed for all participants...')

    # Exit terminal to avoid any funny business
    # -------------------------------------------------------------------------
    os._exit(00)

# %% ---------- end of process_data.py ---------- %% #
