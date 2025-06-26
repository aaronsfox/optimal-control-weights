# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This code is a secondary step in processing the walking data from the
    SchreiberMoissenet2019 dataset through some basic steps of scaling and generating
    kinematics and kinetics from experimental data.

    TODO:
        > Add metabolics to muscles during scaling
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
import btk
import pandas as pd
import random
import pickle

# =========================================================================
# Modify to run different parts of code
# =========================================================================

# Set which processing steps to run
runScaling = False
runIK = False
runTracking = False
runContactAdjustment = False

# =========================================================================
# Set-up
# =========================================================================

# General settings
# -------------------------------------------------------------------------

# Get participant list from data folder
participant_list = [ii for ii in os.listdir(os.path.join('..', 'data')) if os.path.isdir(os.path.join('..', 'data', ii))]

# Read in participant anthropometrics
anthropometrics = pd.read_csv(os.path.join('..', 'data', 'anthropometrics.csv'))

# Set the list of speed conditions to extract
# Modify this if you want to extract different walking speeds
conditionList = [
    # 'C1',   # 0-0.4 m/s
    # 'C2',   # 0.4-0.8 m/s
    # 'C3',   # 0.8-1.2 m/s
    'C4',   # self-selected spontaneuous speed
    # 'C5',   # self-selected fast speed
    ]

# Create the two rotations needed for data
rot1 = osim.Rotation(np.deg2rad(-90), osim.Vec3(1,0,0))
rot2 = osim.Rotation(np.deg2rad(180), osim.Vec3(0,1,0))

# Read in trial data file
trials = pd.read_csv(os.path.join('..', 'data', 'trials.csv'))

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', 'model', 'Geometry'))

# Create dictionaries for tools to avoid over-writing
scaleTool = {participantId: osim.ScaleTool() for participantId in participant_list}
scaleTool_2d = {participantId: osim.ScaleTool() for participantId in participant_list}
ikTool = {participantId: {condition: osim.InverseKinematicsTool() for condition in conditionList} for participantId in participant_list}

# Set kinematics filter frequency
kinematicFiltFreq = 6

# Create dictionary specifying condition colouring
# Set both HEX and RGB colours (https://www.rapidtables.com/convert/color/hex-to-rgb.html)
# RGB are only used as osim Vec3 objects so they can be set that way here
condition_col = {
    '3D': {'HEX': '#4885ed', 'RGB': osim.Vec3(0.2823529411764706, 0.5215686274509804, 0.9294117647058824)},  # 3D = blue,
    '2D': {'HEX': '#ffa600', 'RGB': osim.Vec3(1, 0.6509803921568628, 0)}  # outside shoe = gold
}

# OpenSim settings
# -------------------------------------------------------------------------

# Create measurement set for scaling

# Set the parameters for the measurement sets
measurementSetParams = {
    # Torso
    'torso_depth': {'markerPairs': [['CV7', 'SJN'], ['TV10', 'SXS'], ], 'bodyScale': ['torso'], 'axes': 'X'},
    'torso_height': {'markerPairs': [['CV7', 'TV10'], ['SJN', 'SXS'], ], 'bodyScale': ['torso'], 'axes': 'Y'},
    'torso_width': {'markerPairs': [['R_SAE', 'L_SAE'],], 'bodyScale': ['torso'], 'axes': 'Z'},
    # Pelvis
    'pelvis_depth': {'markerPairs': [['R_IAS', 'R_IPS'], ['L_IAS', 'L_IPS'], ], 'bodyScale': ['pelvis'], 'axes': 'X'},
    'pelvis_height': {'markerPairs': [['R_IAS', 'R_FTC'], ['L_IAS', 'L_FTC'], ], 'bodyScale': ['pelvis'], 'axes': 'Y'},
    'pelvis_width': {'markerPairs': [['R_IAS', 'L_IAS'], ['R_IPS', 'L_IPS'], ], 'bodyScale': ['pelvis'], 'axes': 'Z'},
    # Right thigh
    'r_thigh_length': {'markerPairs': [['R_FTC', 'R_FLE'], ], 'bodyScale': ['femur_r'], 'axes': 'Y'},
    'r_thigh_breadth': {'markerPairs': [['R_FLE', 'R_FME'], ], 'bodyScale': ['femur_r'], 'axes': 'XZ'},
    # Right patella
    'r_patella': {'markerPairs': [['R_FTC', 'R_FLE'], ], 'bodyScale': ['patella_r'], 'axes': 'XYZ'},
    # Right shank
    'r_tibia_length': {'markerPairs': [['R_TTC', 'R_TAM'], ], 'bodyScale': ['tibia_r', 'talus_r'], 'axes': 'Y'},
    'r_tibia_breadth': {'markerPairs': [['R_TAM', 'R_FAL'], ], 'bodyScale': ['tibia_r', 'talus_r'], 'axes': 'XZ'},
    # Right foot
    'r_foot_length': {'markerPairs': [['R_FCC', 'R_FM2'], ], 'bodyScale': ['calcn_r', 'toes_r'], 'axes': 'X'},
    'r_foot_width': {'markerPairs': [['R_FM1', 'R_FM5'], ], 'bodyScale': ['calcn_r', 'toes_r'], 'axes': 'YZ'},
    # Left thigh
    'l_thigh_length': {'markerPairs': [['L_FTC', 'L_FLE'], ], 'bodyScale': ['femur_l'], 'axes': 'Y'},
    'l_thigh_breadth': {'markerPairs': [['L_FLE', 'L_FME'], ], 'bodyScale': ['femur_l'], 'axes': 'XZ'},
    # Left patella
    'l_patella': {'markerPairs': [['L_FTC', 'L_FLE'], ], 'bodyScale': ['patella_l'], 'axes': 'XYZ'},
    # Left shank
    'l_tibia_length': {'markerPairs': [['L_TTC', 'L_TAM'], ], 'bodyScale': ['tibia_l', 'talus_l'], 'axes': 'Y'},
    'l_tibia_breadth': {'markerPairs': [['L_TAM', 'L_FAL'], ], 'bodyScale': ['tibia_l', 'talus_l'], 'axes': 'XZ'},
    # Left foot
    'l_foot_length': {'markerPairs': [['L_FCC', 'L_FM2'], ], 'bodyScale': ['calcn_l', 'toes_l'], 'axes': 'X'},
    'l_foot_width': {'markerPairs': [['L_FM1', 'L_FM5'], ], 'bodyScale': ['calcn_l', 'toes_l'], 'axes': 'YZ'},
    # Right upper arm
    'r_upperarm_length': {'markerPairs': [['R_SAA', 'R_UOA'], ], 'bodyScale': ['humerus_r'], 'axes': 'Y'},
    'r_upperarm_breadth': {'markerPairs': [['R_HLE', 'R_HME'], ], 'bodyScale': ['humerus_r'], 'axes': 'XZ'},
    # Right forearm
    'r_forearm_length': {'markerPairs': [['R_HLE', 'R_RSP'], ['R_HME', 'R_UHE'], ], 'bodyScale': ['radius_r', 'ulna_r'], 'axes': 'Y'},
    'r_forearm_breadth': {'markerPairs': [['R_RSP', 'R_UHE'], ], 'bodyScale': ['radius_r', 'ulna_r'], 'axes': 'XZ'},
    # Right hand
    'r_hand_length': {'markerPairs': [['R_RSP', 'R_HM2'], ['R_UHE', 'R_HM5'], ], 'bodyScale': ['hand_r'], 'axes': 'Y'},
    'r_hand_breadth': {'markerPairs': [['R_HM2', 'R_HM5'], ], 'bodyScale': ['hand_r'], 'axes': 'XZ'},
    # Left upper arm
    'l_upperarm_length': {'markerPairs': [['L_SAA', 'L_UOA'], ], 'bodyScale': ['humerus_l'], 'axes': 'Y'},
    'l_upperarm_breadth': {'markerPairs': [['L_HLE', 'L_HME'], ], 'bodyScale': ['humerus_l'], 'axes': 'XZ'},
    # Left forearm
    'l_forearm_length': {'markerPairs': [['L_HLE', 'L_RSP'], ['L_HME', 'L_UHE'], ], 'bodyScale': ['radius_l', 'ulna_l'], 'axes': 'Y'},
    'l_forearm_breadth': {'markerPairs': [['L_RSP', 'L_UHE'], ], 'bodyScale': ['radius_l', 'ulna_l'], 'axes': 'XZ'},
    # Left hand
    'l_hand_length': {'markerPairs': [['L_RSP', 'L_HM2'], ['L_UHE', 'L_HM5'], ], 'bodyScale': ['hand_l'], 'axes': 'Y'},
    'l_hand_breadth': {'markerPairs': [['L_HM2', 'L_HM5'], ], 'bodyScale': ['hand_l'], 'axes': 'XZ'},
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
    'SJN': {'weight': 5.0}, 'SXS': {'weight': 5.0}, 'CV7': {'weight': 5.0}, 'TV10': {'weight': 5.0},
    'R_SAE': {'weight': 0.0}, 'L_SAE': {'weight': 0.0},
    # Pelvis
    'L_IAS': {'weight': 10.0}, 'L_IPS': {'weight': 10.0}, 'R_IAS': {'weight': 5.0}, 'R_IPS': {'weight': 5.0},
    # Right thigh
    'R_FTC': {'weight': 2.5}, 'R_FLE': {'weight': 5.0}, 'R_FME': {'weight': 5.0},
    # Right shank
    'R_TTC': {'weight': 2.5}, 'R_FAX': {'weight': 2.5}, 'R_TAM': {'weight': 5.0}, 'R_FAL': {'weight': 5.0},
    # Right foot
    'R_FCC': {'weight': 5.0}, 'R_FM1': {'weight': 2.5}, 'R_FM2': {'weight': 2.5}, 'R_FM5': {'weight': 2.5},
    # Left thigh
    'L_FTC': {'weight': 2.5}, 'L_FLE': {'weight': 5.0}, 'L_FME': {'weight': 5.0},
    # Left shank
    'L_TTC': {'weight': 2.5}, 'L_FAX': {'weight': 2.5}, 'L_TAM': {'weight': 5.0}, 'L_FAL': {'weight': 5.0},
    # Left foot
    'L_FCC': {'weight': 5.0}, 'L_FM1': {'weight': 2.5}, 'L_FM2': {'weight': 2.5}, 'L_FM5': {'weight': 2.5},
    # Right upper arm
    'R_SAA': {'weight': 0.0}, 'R_HLE': {'weight': 1.0}, 'R_HME': {'weight': 1.0},
    # Right forearm
    'R_UOA': {'weight': 1.0}, 'R_RSP': {'weight': 1.0}, 'R_UHE': {'weight': 1.0},
    # Right hand
    'R_HM2': {'weight': 1.0}, 'R_HM5': {'weight': 0.0},
    # Left upper arm
    'L_SAA': {'weight': 0.0}, 'L_HLE': {'weight': 1.0}, 'L_HME': {'weight': 1.0},
    # Left forearm
    'L_UOA': {'weight': 1.0}, 'L_RSP': {'weight': 1.0}, 'L_UHE': {'weight': 1.0},
    # Left hand
    'L_HM2': {'weight': 1.0}, 'L_HM5': {'weight': 0.0},
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

# Create the IK task set for tracking
""" NOTE: Defaulting to all 1's (except for non-tracking markers) as subsequent steps will adjust kinematics a little """

# Set the parameters for the IK task sets
ikTaskSetParams = {
    # Torso
    'SJN': {'weight': 1}, 'SXS': {'weight': 1}, 'CV7': {'weight': 1}, 'TV10': {'weight': 1}, 'R_SAE': {'weight': 0}, 'L_SAE': {'weight': 0},
    # Pelvis
    'L_IAS': {'weight': 1}, 'L_IPS': {'weight': 1}, 'R_IAS': {'weight': 1}, 'R_IPS': {'weight': 1},
    # Right thigh
    'R_FTC': {'weight': 1}, 'R_FLE': {'weight': 1}, 'R_FME': {'weight': 1},
    # Right shank
    'R_TTC': {'weight': 1}, 'R_FAX': {'weight': 1}, 'R_TAM': {'weight': 1}, 'R_FAL': {'weight': 1},
    # Right foot
    'R_FCC': {'weight': 1}, 'R_FM1': {'weight': 1}, 'R_FM2': {'weight': 1}, 'R_FM5': {'weight': 1},
    # Left thigh
    'L_FTC': {'weight': 1}, 'L_FLE': {'weight': 1}, 'L_FME': {'weight': 1},
    # Left shank
    'L_TTC': {'weight': 1}, 'L_FAX': {'weight': 1}, 'L_TAM': {'weight': 1}, 'L_FAL': {'weight': 1},
    # Left foot
    'L_FCC': {'weight': 1}, 'L_FM1': {'weight': 1}, 'L_FM2': {'weight': 1}, 'L_FM5': {'weight': 1},
    # Right upper arm
    'R_SAA': {'weight': 0}, 'R_HLE': {'weight': 1}, 'R_HME': {'weight': 1},
    # Right forearm
    'R_UOA': {'weight': 1}, 'R_RSP': {'weight': 1}, 'R_UHE': {'weight': 1},
    # Right hand
    'R_HM2': {'weight': 0}, 'R_HM5': {'weight': 0},
    # Left upper arm
    'L_SAA': {'weight': 0}, 'L_HLE': {'weight': 1}, 'L_HME': {'weight': 1},
    # Left forearm
    'L_UOA': {'weight': 1}, 'L_RSP': {'weight': 1}, 'L_UHE': {'weight': 1},
    # Left hand
    'L_HM2': {'weight': 0}, 'L_HM5': {'weight': 0},
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

# Set actuator forces to assist tracking simulations
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
    'shoulder_flexion_r': {'actuatorType': 'torque', 'optForce': 100.0},
    'shoulder_adduction_r': {'actuatorType': 'torque', 'optForce': 100.0},
    'shoulder_rotation_r': {'actuatorType': 'torque', 'optForce': 100.0},
    'elbow_flexion_r': {'actuatorType': 'torque', 'optForce': 50.0},
    'shoulder_flexion_l': {'actuatorType': 'torque', 'optForce': 100.0},
    'shoulder_adduction_l': {'actuatorType': 'torque', 'optForce': 100.0},
    'shoulder_rotation_l': {'actuatorType': 'torque', 'optForce': 100.0},
    'elbow_flexion_l': {'actuatorType': 'torque', 'optForce': 50.0},
    }

# Set tracking weights
globalMarkerTrackingWeight = 10.0
globalControlEffortWeight = 1e-3

# Set mesh interval
meshInterval = 50

# =========================================================================
# Define functions
# =========================================================================

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
    staticTRC = glob.glob(os.path.join('..', 'data', participant_id, 'static', '*_ST.trc'))[0]

    # Set-up and run the scale tool for the 3D model
    # -------------------------------------------------------------------------

    # Set participant mass
    scaleTool[participant_id].setSubjectMass(
        anthropometrics[anthropometrics['subjectID'] == int(participant_id)]['mass'].values[0]
    )

    # Set generic model file
    # scaleTool[participant_id].getGenericModelMaker().setModelFileName(os.path.join('..', 'model',
    #                                                                                'Uhlrich2022_SchreiberMoissenet2019.osim'))
    scaleTool[participant_id].getGenericModelMaker().setModelFileName(os.path.join('..', 'model',
                                                                                   'Miller2021_SchreiberMoissenet2019.osim'))

    # Set measurement set in model scaler
    scaleTool[participant_id].getModelScaler().setMeasurementSet(scaleMeasurementSet)

    # Set scale tasks in tool
    for ii in range(scaleTaskSet.getSize()):
        scaleTool[participant_id].getMarkerPlacer().getIKTaskSet().cloneAndAppend(scaleTaskSet.get(ii))

    # Set marker file
    scaleTool[participant_id].getMarkerPlacer().setMarkerFileName(staticTRC)
    scaleTool[participant_id].getModelScaler().setMarkerFileName(staticTRC)

    # Set options
    scaleTool[participant_id].getModelScaler().setPreserveMassDist(True)
    scaleOrder = osim.ArrayStr()
    scaleOrder.set(0, 'measurements')
    scaleTool[participant_id].getModelScaler().setScalingOrder(scaleOrder)

    # Set time ranges
    timeRange = osim.ArrayDouble()
    timeRange.set(0, 0)  # initial time
    timeRange.set(1, 1)  # final time
    scaleTool[participant_id].getMarkerPlacer().setTimeRange(timeRange)
    scaleTool[participant_id].getModelScaler().setTimeRange(timeRange)

    # Set output files
    scaleTool[participant_id].getModelScaler().setOutputModelFileName(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModel.osim'))
    scaleTool[participant_id].getModelScaler().setOutputScaleFileName(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaleSet.xml'))

    # Set marker adjustment parameters
    scaleTool[participant_id].getMarkerPlacer().setOutputMotionFileName(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_staticMotion.mot'))
    scaleTool[participant_id].getMarkerPlacer().setOutputModelFileName(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModelAdjusted.osim'))

    # Save and run scale tool
    scaleTool[participant_id].printToXML(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaleSetup.xml'))
    scaleTool[participant_id].run()

    # Set-up and run the scale tool for the 2D model (no marker adjustment)
    # -------------------------------------------------------------------------

    # Set participant mass
    scaleTool_2d[participant_id].setSubjectMass(
        anthropometrics[anthropometrics['subjectID'] == int(participant_id)]['mass'].values[0]
    )

    # Turn off marker placer
    scaleTool_2d[participant_id].getMarkerPlacer().setApply(False)

    # Set generic model file
    # scaleTool_2d[participant_id].getGenericModelMaker().setModelFileName(os.path.join('..', 'model',
    #                                                                                   'Uhlrich2022_SchreiberMoissenet2019_2d.osim'))
    scaleTool_2d[participant_id].getGenericModelMaker().setModelFileName(os.path.join('..', 'model',
                                                                                      'Denton2023_SchreiberMoissenet2019.osim'))

    # Set measurement set in model scaler
    scaleTool_2d[participant_id].getModelScaler().setMeasurementSet(scaleMeasurementSet)

    # Set marker file
    scaleTool_2d[participant_id].getMarkerPlacer().setMarkerFileName(staticTRC)
    scaleTool_2d[participant_id].getModelScaler().setMarkerFileName(staticTRC)

    # Set options
    scaleTool_2d[participant_id].getModelScaler().setPreserveMassDist(True)
    scaleOrder = osim.ArrayStr()
    scaleOrder.set(0, 'measurements')
    scaleTool_2d[participant_id].getModelScaler().setScalingOrder(scaleOrder)

    # Set time ranges
    timeRange = osim.ArrayDouble()
    timeRange.set(0, 0)  # initial time
    timeRange.set(1, 1)  # final time
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
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModelAdjusted.osim'))
    scaledModel_2d = osim.Model(
        os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModel_2d.osim'))

    # Set model name
    scaledModel.setName(participant_id)
    scaledModel_2d.setName(participant_id+'_2d')

    # Scale model muscle forces according to height-mass relationship

    # Get generic model mass and set generic height for 3D model (this is the same as 2D model)
    genModel = osim.Model(os.path.join('..', 'model', 'Miller2021_SchreiberMoissenet2019.osim'))
    genModel_2d = osim.Model(os.path.join('..', 'model', 'Denton2023_SchreiberMoissenet2019.osim'))
    genModelMass = np.sum([genModel.getBodySet().get(bodyInd).getMass() for bodyInd in range(genModel.getBodySet().getSize())])
    genModelHeight = 1.70

    # Get scaled model height (use mass from earlier)
    heightM = anthropometrics.loc[anthropometrics['subjectID'] == int(participant_id),]['height'].values[0]
    massKg = anthropometrics[anthropometrics['subjectID'] == int(participant_id)]['mass'].values[0]

    # Get muscle volume totals based on mass and heights with linear equation
    genericMuscVol = 47.05 * genModelMass * genModelHeight + 1289.6
    scaledMuscVol = 47.05 * massKg * heightM + 1289.6

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

    # # Add marker locations underneath foot markers at floor level based on static motion
    # # These may be useful later in determining foot-ground contact points
    #
    # # Clear all markers from 2D model as these won't be used with IK
    # scaledModel_2d.updMarkerSet().clearAndDestroy()
    #
    # # Set the list of markers to project to the floot
    # floorMarkers = ['R_FM1', 'R_FM2', 'R_FM5', 'R_FCC',
    #                 'L_FM1', 'L_FM2', 'L_FM5', 'L_FCC']
    #
    # # Load the static motion
    # staticMotion = osim.TimeSeriesTable(
    #     os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_staticMotion.mot'))
    #
    # # Initialise the model state
    # modelState = scaledModel.initSystem()
    #
    # # Set model states from static motion
    # for stateName in staticMotion.getColumnLabels():
    #     scaledModel.setStateVariableValue(modelState, stateName, staticMotion.getDependentColumn(stateName).to_numpy()[0])
    #
    # # Realize model to position stage
    # scaledModel.realizePosition(modelState)
    #
    # # Loop through floor markers
    # # Identify position in ground at floor level and translate back to associated body
    # # Append markers to both 3D and 2D model
    # for marker in floorMarkers:
    #     # Get marker location in ground
    #     markerInGround = scaledModel.updMarkerSet().get(marker).getLocationInGround(modelState)
    #     # Set the y-axis to be zero for the marker in ground position (i.e. floor level)
    #     markerInGround.set(1, 0)
    #     # Translate this new position back to the markers associated body
    #     floorMarkerLoc = scaledModel.getGround().findStationLocationInAnotherFrame(
    #         modelState, markerInGround, scaledModel.updMarkerSet().get(marker).getParentFrame()
    #     )
    #     # Create new marker and append to model
    #     newMarker = osim.Marker()
    #     newMarker.setName(marker + '_ground')
    #     newMarker.setParentFrameName(scaledModel.updMarkerSet().get(marker).getParentFrameName())
    #     newMarker.set_location(floorMarkerLoc)
    #     scaledModel.getMarkerSet().cloneAndAppend(newMarker)
    #     scaledModel_2d.getMarkerSet().cloneAndAppend(newMarker)
    #
    # # Create mid-foot markers between FM and heel points
    # # Add these to both the 3d and 2d models
    # for marker in floorMarkers:
    #     # Add additional markers at the mid-foot with the FM markers
    #     # These sit at the mid distance to the heel marker along the X, Y and Z axes
    #     if '_FM' in marker:
    #         # Get marker location in ground
    #         markerInGround = scaledModel.updMarkerSet().get(marker).getLocationInGround(modelState)
    #         # Set the y-axis to be zero for the marker in ground position (i.e. floor level)
    #         markerInGround.set(1, 0)
    #         # Get the appropriate heel marker position
    #         heelMarkerInGround = scaledModel.updMarkerSet().get(marker[0] + '_FCC').getLocationInGround(modelState)
    #         # Find mid-point of the points in the ground along the x-axis and y-axis
    #         midPointX = markerInGround.get(0) - ((markerInGround.get(0) - heelMarkerInGround.get(0)) / 2)
    #         midPointY = markerInGround.get(1) - ((markerInGround.get(1) - heelMarkerInGround.get(1)) / 2)
    #         midPointZ = markerInGround.get(2) - ((markerInGround.get(2) - heelMarkerInGround.get(2)) / 2)
    #         # Get the mid-point marker location in the body frame
    #         midMarkerLoc = scaledModel.getGround().findStationLocationInAnotherFrame(
    #             modelState, osim.Vec3(midPointX, midPointY, midPointZ),
    #             scaledModel.updMarkerSet().get(marker).getParentFrame())
    #         # Create new marker and append to model
    #         newMarker = osim.Marker()
    #         newMarker.setName(marker + '_mid_ground')
    #         newMarker.setParentFrameName(scaledModel.updMarkerSet().get(marker).getParentFrameName())
    #         newMarker.set_location(midMarkerLoc)
    #         scaledModel.getMarkerSet().cloneAndAppend(newMarker)
    #         scaledModel_2d.getMarkerSet().cloneAndAppend(newMarker)

    # Adjust contact geometry positions back to original reference locations

    # 3D model
    # Load in unadjusted model and get marker positions
    unadjustedModel = osim.Model(os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModel.osim'))
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
    scaledModel.printToXML(os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModelAdjusted.osim'))
    scaledModel_2d.printToXML(os.path.join('..', 'data', participant_id, 'scaling', f'{participant_id}_scaledModelAdjusted_2d.osim'))


# Run inverse kinematics
# -------------------------------------------------------------------------
def run_ik(participant_id):

    """
    :param participant_id: participant ID to run IK for
    :return:
    """

    # Create scaling directory for files
    os.makedirs(os.path.join('..', 'data', participant_id, 'ik'), exist_ok=True)

    # Get dynamic files
    dyna_file = glob.glob(os.path.join('..', 'data', participant_id, 'dynamic', '*.c3d'))

    # Loop through trials
    for trial_file in dyna_file:

        # Load event data timing
        with open(os.path.join('..', 'data', participant_id, 'events',
                               os.path.split(trial_file)[-1].split('.')[0] + '.pkl'), 'rb') as pklFile:
            event_data = pickle.load(pklFile)

        # Run IK on trial
        # -------------------------------------------------------------------------

        # Get trial name and condition
        condition = os.path.split(trial_file)[-1].split('.')[0].split('_')[1]
        trialName = os.path.split(trial_file)[-1].split('.')[0].replace(f'{participant_id}_', '')

        # Create a folder for the condition and trial
        os.makedirs(os.path.join('..', 'data', participant_id, 'ik', trialName), exist_ok=True)

        # Set model
        ikTool[participant_id][condition].set_model_file(os.path.join('..', 'data', participant_id, 'scaling',
                                                                      f'{participant_id}_scaledModelAdjusted.osim'))

        # Set task set (consistent for all trials)
        for taskInd in range(ikTaskSet.getSize()):
            ikTool[participant_id][condition].getIKTaskSet().adoptAndAppend(ikTaskSet.get(taskInd))

        # Set to report marker locations
        ikTool[participant_id][condition].set_report_marker_locations(True)

        # Set the marker file (relative to setup file location)
        ikTool[participant_id][condition].setMarkerDataFileName(os.path.join('..', 'data', participant_id, 'dynamic',
                                                                             os.path.split(trial_file)[-1].split('.')[0] + '.trc'))

        # Set times
        ikTool[participant_id][condition].setStartTime(event_data['startTime'])
        ikTool[participant_id][condition].setEndTime(event_data['endTime'])

        # Set output filename (relative to setup file location)
        ikTool[participant_id][condition].setOutputMotionFileName(os.path.join('..', 'data', participant_id, 'ik', trialName,
                                                                               f'{trialName}_ik.mot'))

        # Save IK tool to file
        ikTool[participant_id][condition].printToXML('ikSetup.xml')

        # Bring the tool back in and run it (this seems to avoid Python kernel crashing)
        ikRun = osim.InverseKinematicsTool('ikSetup.xml')
        ikRun.run()

        # Rename supplementary marker outputs
        shutil.move('_ik_marker_errors.sto',
                    os.path.join('..', 'data', participant_id, 'ik', trialName,
                                 f'{trialName}_ikMarkerErrors.sto'))
        shutil.move('_ik_model_marker_locations.sto',
                    os.path.join('..', 'data', participant_id, 'ik', trialName,
                                 f'{trialName}_ikModelMarkerLocations.sto'))
        shutil.move('ikSetup.xml',
                    os.path.join('..', 'data', participant_id, 'ik', trialName,
                                 f'{trialName}_ikSetup.xml'))

        # Run 2D version of IK
        # -------------------------------------------------------------------------

        # The first step is to create an appropriate 2D version of the marker data
        # All z-data therefore aligns to the locked position on the scaled model

        # Read in TRC data
        trc_data = osim.TimeSeriesTableVec3(os.path.join('..', 'data', participant_id, 'dynamic',
                                                         os.path.split(trial_file)[-1].split('.')[0] + '.trc'))

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
        osim.TRCFileAdapter().write(trc_data_2d,
                                    os.path.join('..', 'data', participant_id, 'dynamic',
                                                 os.path.split(trial_file)[-1].split('.')[0] + '_2d.trc'))

        # Re-use IK tool to get 2D kinematics
        # Only some parameters need to be updated

        # Set model
        ikTool[participant_id][condition].set_model_file(os.path.join('..', 'data', participant_id, 'scaling',
                                                                      f'{participant_id}_scaledModelAdjusted_2d.osim'))

        # Set the marker file (relative to setup file location)
        ikTool[participant_id][condition].setMarkerDataFileName(os.path.join('..', 'data', participant_id, 'dynamic',
                                                                             os.path.split(trial_file)[-1].split('.')[0] + '_2d.trc'))

        # Set output filename (relative to setup file location)
        ikTool[participant_id][condition].setOutputMotionFileName(os.path.join('..', 'data', participant_id, 'ik', trialName,
                                                                               f'{trialName}_ik_2d.mot'))

        # Save IK tool to file
        ikTool[participant_id][condition].printToXML('ikSetup.xml')

        # Bring the tool back in and run it (this seems to avoid Python kernel crashing)
        ikRun = osim.InverseKinematicsTool('ikSetup.xml')
        ikRun.run()

        # Rename supplementary marker outputs
        shutil.move('_ik_marker_errors.sto',
                    os.path.join('..', 'data', participant_id, 'ik', trialName,
                                 f'{trialName}_ikMarkerErrors_2d.sto'))
        shutil.move('_ik_model_marker_locations.sto',
                    os.path.join('..', 'data', participant_id, 'ik', trialName,
                                 f'{trialName}_ikModelMarkerLocations_2d.sto'))
        shutil.move('ikSetup.xml',
                    os.path.join('..', 'data', participant_id, 'ik', trialName,
                                 f'{trialName}_ikSetup_2d.xml'))


# Run marker tracking
# -------------------------------------------------------------------------
def run_tracking(participant_id, model_variant='3d'):

    # Organise and set-up files
    # -------------------------------------------------------------------------

    # Create directory for files
    os.makedirs(os.path.join('..', 'data', participant_id, 'exp'), exist_ok=True)

    # Get dynamic files
    dyna_file = glob.glob(os.path.join('..', 'data', participant_id, 'dynamic', '*.c3d'))

    # Loop through trials
    for trial_file in dyna_file:

        # Load event data timing
        with open(os.path.join('..', 'data', participant_id, 'events',
                               os.path.split(trial_file)[-1].split('.')[0] + '.pkl'), 'rb') as pklFile:
            event_data = pickle.load(pklFile)

        # Run marker tracking on trial
        # -------------------------------------------------------------------------

        # Get trial name and condition
        condition = os.path.split(trial_file)[-1].split('.')[0].split('_')[1]
        trial_name = os.path.split(trial_file)[-1].split('.')[0].replace(f'{participant_id}_', '')

        # Create folder for condition and trial
        os.makedirs(os.path.join('..', 'data', participant_id, 'exp', trial_name), exist_ok=True)

        # Create directory to store model variant specific results
        os.makedirs(os.path.join('..', 'data', participant_id, 'exp', trial_name, model_variant), exist_ok=True)

        # Navigate to simulation folder for ease of use
        home_dir = os.getcwd()
        os.chdir(os.path.join('..', 'data', participant_id, 'exp', trial_name, model_variant))

        # Copy external loads file to simulation directory
        shutil.copyfile(os.path.join('..', '..', '..', 'dynamic', f'{participant_id}_{trial_name}_grf.mot'),
                        f'{participant_id}_{trial_name}_grf.mot')
        shutil.copyfile(os.path.join('..', '..', '..', 'dynamic', f'{participant_id}_{trial_name}_grf.xml'),
                        f'{participant_id}_{trial_name}_grf.xml')

        # Copy TRC file to simulation directory
        if model_variant == '3d':
            shutil.copyfile(os.path.join('..', '..', '..', 'dynamic', f'{participant_id}_{trial_name}.trc'),
                            f'{participant_id}_{trial_name}_{model_variant}.trc')
        elif model_variant == '2d':
            shutil.copyfile(os.path.join('..', '..', '..', 'dynamic', f'{participant_id}_{trial_name}_{model_variant}.trc'),
                            f'{participant_id}_{trial_name}_{model_variant}.trc')

        # Copy IK file to simulation directory
        if model_variant == '3d':
            shutil.copyfile(os.path.join('..', '..', '..', 'ik', trial_name, f'{trial_name}_ik.mot'),
                            f'{trial_name}_ik_{model_variant}.mot')
        elif model_variant == '2d':
            shutil.copyfile(os.path.join('..', '..', '..', 'ik', trial_name, f'{trial_name}_ik_{model_variant}.mot'),
                            f'{trial_name}_ik_{model_variant}.mot')

        # Copy model file to simulation directory
        if model_variant == '3d':
            shutil.copyfile(os.path.join('..', '..', '..', 'scaling', f'{participant_id}_scaledModelAdjusted.osim'),
                            f'{participant_id}_scaledModelAdjusted_{model_variant}.osim')
        elif model_variant == '2d':
            shutil.copyfile(os.path.join('..', '..', '..', 'scaling', f'{participant_id}_scaledModelAdjusted_{model_variant}.osim'),
                            f'{participant_id}_scaledModelAdjusted_{model_variant}.osim')

        # For 2D model the GRFs need to be updated to eliminate z-axis data
        # In lieu of accurate COP tracking in Z-direction this is normalised to mid-point of malleoli markers
        if model_variant == '2d':
            # Load in GRF data
            grf_data = osim.TimeSeriesTable(f'{participant_id}_{trial_name}_grf.mot')
            ex_loads = osim.ExternalLoads(f'{participant_id}_{trial_name}_grf.xml', True)
            # Load in marker data for Z-COP reference
            marker_data = osim.TimeSeriesTableVec3(f'{participant_id}_{trial_name}_{model_variant}.trc').flatten()
            # Get average position of ankle markers
            r_midpoint = (marker_data.getDependentColumn('R_FAL_3').to_numpy().mean() + marker_data.getDependentColumn('R_TAM_3').to_numpy().mean()) / 2
            l_midpoint = (marker_data.getDependentColumn('L_FAL_3').to_numpy().mean() + marker_data.getDependentColumn('L_TAM_3').to_numpy().mean()) / 2
            # Figure out which column to apply right and left mid-point to based on external loads
            # Divide by 1000 here to convert to m from mm
            cop_vals = {}
            if ex_loads.get(0).getAppliedToBodyName() == 'calcn_l':
                cop_vals[ex_loads.get(0).get_point_identifier() + 'z'] = l_midpoint / 1000
                cop_vals[ex_loads.get(1).get_point_identifier() + 'z'] = r_midpoint / 1000
            elif ex_loads.get(0).getAppliedToBodyName() == 'calcn_r':
                cop_vals[ex_loads.get(0).get_point_identifier() + 'z'] = r_midpoint / 1000
                cop_vals[ex_loads.get(1).get_point_identifier() + 'z'] = l_midpoint / 1000
            # Get GRF data column labels
            grf_columns = list(grf_data.getColumnLabels())
            # Loop through row indices and reset values
            for row_ind in range(grf_data.getNumRows()):
                # First check if all values are zero, as then the row can be skipped
                if not all(grf_data.getRowAtIndex(row_ind).to_numpy() == 0):
                    # Create the array of zeros
                    replace_data = np.zeros(len(grf_columns))
                    # Loop through columns
                    # fz and mz values aren't checked for in this process as they will be left as zero
                    for col in grf_columns:
                        # First check if it is a replacement COP value
                        if col in cop_vals.keys():
                            replace_data[grf_columns.index(col)] = cop_vals[col]
                        # Check for x or y data that needs to be retained
                        elif col.endswith('x') or col.endswith('y'):
                            replace_data[grf_columns.index(col)] = grf_data.getDependentColumn(col).to_numpy()[row_ind]
                    # Set the row in the grf data to the replacement values
                    grf_data.setRowAtIndex(row_ind, osim.RowVector().createFromMat(replace_data))
            # Write the new grf data to file
            osim.STOFileAdapter().write(grf_data, f'{participant_id}_{trial_name}_grf.mot')

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
            osim_model.updForceSet().remove(remove-counter)
            counter += 1

        # Finalise model connections
        osim_model.finalizeConnections()

        # Print model to file
        osim_model.printToXML(f'{participant_id}_{trial_name}_exp-tracking_{model_variant}.osim')

        # Adjust kinematic data from IK for initial guess
        # -------------------------------------------------------------------------

        # Load in the kinematic data
        kinematics_data = osim.Storage(f'{trial_name}_ik_{model_variant}.mot')

        # Create a copy of the kinematics data to alter the column labels in
        states_data = osim.Storage(f'{trial_name}_ik_{model_variant}.mot')

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
        states_data.printToXML(f'{trial_name}_coordinates_{model_variant}.sto')

        # Set up tracking simulation
        # -------------------------------------------------------------------------

        # Create tracking tool
        track = osim.MocoTrack()
        track.setName(f'{participant_id}_{trial_name}_exp-tracking_{model_variant}')

        # Create model processor
        track_model_proc = osim.ModelProcessor(f'{participant_id}_{trial_name}_exp-tracking_{model_variant}.osim')

        # Append external loads
        track_model_proc.append(osim.ModOpAddExternalLoads(f'{participant_id}_{trial_name}_grf.xml'))

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
        track.setMarkersReferenceFromTRC(f'{participant_id}_{trial_name}_{model_variant}.trc')

        # Set to ignore unused columns
        track.set_allow_unused_references(True)

        # Set global tracking weight
        track.set_markers_global_tracking_weight(globalMarkerTrackingWeight)

        # Set marker weights
        tracking_weights = osim.MocoWeightSet()
        for marker in ikTaskSetParams:
            if ikTaskSetParams[marker]['weight'] != 0:
                tracking_weights.cloneAndAppend(osim.MocoWeight(marker, ikTaskSetParams[marker]['weight']))
        track.set_markers_weight_set(tracking_weights)

        # Set the timings
        track.set_initial_time(osim.TimeSeriesTable(f'{trial_name}_coordinates_{model_variant}.sto').getIndependentColumn()[0])
        track.set_final_time(osim.TimeSeriesTable(f'{trial_name}_coordinates_{model_variant}.sto').getIndependentColumn()[-1])

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
        ik_states = osim.TimeSeriesTable(f'{trial_name}_coordinates_{model_variant}.sto')
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
        solution.write(f'{participant_id}_{trial_name}_exp-tracking-solution_{model_variant}.sto')

        # Remove initial tracked states and markers file
        os.remove(f'{participant_id}_{trial_name}_exp-tracking_{model_variant}_tracked_markers.sto')

        # Return to home directory
        os.chdir(home_dir)

        # Create a folder of data to use with muscle redundancy solver
        # -------------------------------------------------------------------------

        #  Create the folder to store data in
        os.makedirs(os.path.join('..', 'data', participant_id, 'mrs', trial_name, model_variant), exist_ok=True)

        # Copy the IK data across
        shutil.copyfile(os.path.join('..', 'data', participant_id, 'exp', trial_name, model_variant, f'{trial_name}_ik_{model_variant}.mot'),
                        os.path.join('..', 'data', participant_id, 'mrs', trial_name, model_variant, f'{trial_name}_ik_{model_variant}.mot'))

        # Convert tracking solution controls to ID like file
        tracking_controls = osim.MocoTrajectory(os.path.join('..', 'data', participant_id, 'exp', trial_name, model_variant,
                                                             f'{participant_id}_{trial_name}_exp-tracking-solution_{model_variant}.sto')
                                                ).exportToControlsTable()

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
            elif any([ii in col for ii in ['pelvis_list', 'pelvis_tilt', 'pelvisrotation']]):
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
                                    os.path.join('..', 'data', participant_id, 'mrs', trial_name, model_variant,
                                                 f'{trial_name}_id_{model_variant}.sto'))


# Adjust vertical position of model in kinematics based on static trial data
# Position is adjusted based on desired sphere penetration in static trial motion
# -------------------------------------------------------------------------
def adjust_model_position(participant_id, model_variant='3d', sphere_penetration_per = 0.25):

    # Create folder for model to be stored
    os.makedirs(os.path.join('..', 'data', participant_id, 'model'), exist_ok=True)

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
                                                      f'{participant_id}_staticMotion.mot'))

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

    # Identify tracking files
    tracking_trials = [ff for ff in os.listdir(os.path.join('..', 'data', participant_id, 'exp'))
                       if os.path.isdir(os.path.join(os.path.join('..', 'data', participant_id, 'exp'), ff))]

    # Loop through trials to apply offset
    for trial in tracking_trials:

        # Load in solution trajectory
        solution_data = osim.MocoTrajectory(os.path.join('..', 'data', participant_id, 'exp', trial, model_variant,
                                                         f'{participant_id}_{trial}_exp-tracking-solution_{model_variant}.sto'))

        # Adjust vertical pelvis position by average offset
        pelvis_ty_state = osim_model.getCoordinateSet().get('pelvis_ty').getAbsolutePathString()+'/value'
        solution_data.setState(pelvis_ty_state, solution_data.getState(pelvis_ty_state).to_numpy() + offset_avg)

        # Write adjusted data to file
        solution_data.write(os.path.join('..', 'data', participant_id, 'exp', trial, model_variant,
                                         f'{participant_id}_{trial}_exp-tracking-solution-adjusted_{model_variant}.sto'))

    # Finalise processing by saving model to file
    # -------------------------------------------------------------------------

    # Write new model to file for use in subsequent processes
    # This doesn't have any changes applied but is useful to finalise here
    osim_model.finalizeConnections()
    osim_model.printToXML(os.path.join('..', 'data', participant_id, 'model', f'{participant_id}_model_{model_variant}.osim'))


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
