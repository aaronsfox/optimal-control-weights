# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This code is a preliminary step in extracting just the relevant data needed
    from the SchreiberMoissenet2019 dataset. There are a number of C3D files in
    each participants folder, and this code grabs the appropriate files for the
    desired speeds. Participant data is checked for trials that meet the conditions
    for analysis (see README for details). The c3d files are converted to OpenSim
    friendly formats, those being TRC and MOT file types. Each participant needs
    to have their data within a 'raw' folder and then in a folder named with the
    participant code (e.g. '2014001') for this script to work (see README in
    folder for details).

    An important variable in the set-up of this code is the n_participants variable.
    This dictates the number of participants randomly selected to extract data for.
    The total number of participants will be this value * 2, as it selects an even
    number of male vs. female participants. We chose to have a total of 10 participants,
    as this was reasonable from a computational perspective for this work, and 10 is
    also a nice round number...
    
    TODO:
        > Could extract EMG data from C3D file for subsequent use/validation in pipelines?
        > Given EMG data is from right limb, perhaps get trials that are similar in left/right foot patterns?

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
from tqdm import tqdm

# =========================================================================
# Set-up
# =========================================================================

# Set number of participants to extract data for
# Note that you end up with double this as we want 5 male and 5 female participants
n_participants = 5

# Set the participant list based on the codes in the raw folder
# See README.MD in dataset folder for details on setting up raw data
participantList = [ii for ii in os.listdir(os.path.join('..', 'data', 'raw')) if os.path.isdir(os.path.join('..', 'data', 'raw', ii))]

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

# =========================================================================
# Define functions
# =========================================================================

# Extract participants data
# -------------------------------------------------------------------------
def extract_participant(participant_id):

    """
    :param participant_id (string): id of the participant to extract data for
    :return:
    """

    # Create folders for the participant

    # Starting directory
    os.makedirs(os.path.join('..', 'data', participant_id), exist_ok=True)

    # Static files directory
    os.makedirs(os.path.join('..', 'data', participant_id, 'static'), exist_ok=True)

    # Dynamic trials file directory
    os.makedirs(os.path.join('..', 'data', participant_id, 'dynamic'), exist_ok=True)

    # Event data directory
    os.makedirs(os.path.join('..', 'data', participant_id, 'events'), exist_ok=True)

    # Get the static file

    # Get the path to the static file
    staticFile = glob.glob(os.path.join('..', 'data', 'raw', participant_id, '*_ST.c3d*'))[0]

    # Copy c3d across to static directory
    shutil.copyfile(staticFile, os.path.join('..', 'data', participant_id, 'static', os.path.split(staticFile)[-1]))

    # Construct opensim 3d object
    c3dFile = osim.C3DFileAdapter()
    c3dFile.setLocationForForceExpression(osim.C3DFileAdapter.ForceLocation_CenterOfPressure)

    # Read in the static trial
    staticC3D = c3dFile.read(staticFile)

    # Get markers table
    staticMarkers = c3dFile.getMarkersTable(staticC3D)

    # Rotate marker data
    for iRow in range(staticMarkers.getNumRows()):
        # Apply the two rotations
        staticMarkers.setRowAtIndex(iRow, rot1.multiply(staticMarkers.getRowAtIndex(iRow)))

    # There's a potentially annoying bug to deal with in scaling when only one row of marker data exists
    # Add a pseudo row of identical marker data to avoid this
    staticMarkers.appendRow(1, staticMarkers.getRow(0))

    # Write static markers to TRC file
    osim.TRCFileAdapter().write(staticMarkers, os.path.join('..', 'data', participant_id, 'static',
                                                            os.path.split(staticFile)[-1].split('.')[0] + '.trc'))

    # Get the dynamic files

    # Identify dynamic files to extract from those selected
    extractFiles = [os.path.join('..', 'data', 'raw', participant_id,
                                 f'{participant_id}_{conditionTrial[condition]}.c3d') for condition in conditionList]

    # Loop through files to copy and convert
    for dynaFile in extractFiles:

        # Copy c3d across to dynamic directory
        shutil.copyfile(dynaFile, os.path.join('..', 'data', participant_id, 'dynamic', os.path.split(dynaFile)[-1]))

        # Construct opensim 3d object
        c3dFile = osim.C3DFileAdapter()
        c3dFile.setLocationForForceExpression(osim.C3DFileAdapter.ForceLocation_CenterOfPressure)

        # Read in the dynamic trial
        dynaC3D = c3dFile.read(dynaFile)

        # Get markers table
        dynaMarkers = c3dFile.getMarkersTable(dynaC3D)

        # Rotate marker data
        for iRow in range(dynaMarkers.getNumRows()):
            # Apply the two rotations
            dynaMarkers.setRowAtIndex(iRow, rot1.multiply(dynaMarkers.getRowAtIndex(iRow)))

        # Check if moving in negative x-direction and data needs to be flipped for consistency
        # Check by examining average difference in pelvis marker (negativ difference needs to be flipped)
        if np.diff(dynaMarkers.flatten().getDependentColumn('L_IAS_1').to_numpy()).mean() < 0:
            flipDirection = True
        else:
            flipDirection = False

        # Rotate 180 degrees abot y-axis to flip direction if necessary
        if flipDirection:
            # Rotate marker data
            for iRow in range(dynaMarkers.getNumRows()):
                # Apply the two rotations
                dynaMarkers.setRowAtIndex(iRow, rot2.multiply(dynaMarkers.getRowAtIndex(iRow)))

        # Write dynamic markers to TRC file
        osim.TRCFileAdapter().write(dynaMarkers, os.path.join('..', 'data', participant_id, 'dynamic',
                                                              os.path.split(dynaFile)[-1].split('.')[0] + '.trc'))

        # Get forces table
        dynaForces = c3dFile.getForcesTable(dynaC3D)

        # Rotate forces data
        for iRow in range(dynaForces.getNumRows()):
            dynaForces.setRowAtIndex(iRow, rot1.multiply(dynaForces.getRowAtIndex(iRow)))

        # Rotate 180 degrees abot y-axis to flip direction if necessary
        if flipDirection:
            # Rotate forces data
            for iRow in range(dynaForces.getNumRows()):
                dynaForces.setRowAtIndex(iRow, rot2.multiply(dynaForces.getRowAtIndex(iRow)))

        # Flatten forces data
        forcesFlat = dynaForces.flatten()

        # Convert to numpy array
        # Pre-allocate numpy array based on data size
        dataArray = np.zeros((forcesFlat.getNumRows(),
                              forcesFlat.getNumColumns()))

        # Extract data
        for forceInd in range(forcesFlat.getNumColumns()):
            dataArray[:, forceInd] = forcesFlat.getDependentColumn(forcesFlat.getColumnLabels()[forceInd]).to_numpy()

        # Replace nan's for COP and moment data with zeros
        np.nan_to_num(dataArray, copy=False, nan=0.0)

        # Convert force point data from mm to m
        for forceName in list(forcesFlat.getColumnLabels()):
            if forceName.startswith('p') or forceName.startswith('m'):
                # Get force index
                forceInd = list(forcesFlat.getColumnLabels()).index(forceName)
                # Convert to m units in data array
                dataArray[:, forceInd] = dataArray[:, forceInd] / 1000

        # Build the new time series table
        forcesStorage = osim.Storage()

        # Get the time data
        time = forcesFlat.getIndependentColumn()

        # Create maps to replace text from force labels with
        # Force plate and type identifiers
        forceType = {}
        for ii in range(1, int(len(forcesFlat.getColumnLabels()) / 9) + 1):
            forceType[f'f{ii}'] = f'ground_force_{ii}_v'
            forceType[f'p{ii}'] = f'ground_force_{ii}_p'
            forceType[f'm{ii}'] = f'ground_force_{ii}_m'
        # Axis identifiers
        forceAxis = {'1': 'x',
                     '2': 'y',
                     '3': 'z'}

        # Set labels in table
        newLabels = osim.ArrayStr()
        newLabels.append('time')
        for forceLabel in forcesFlat.getColumnLabels():
            # Split the label to get parts
            labelSplit = forceLabel.split('_')
            # Create new label
            forceLabel = f'{forceType[labelSplit[0]]}{forceAxis[labelSplit[1]]}'
            # Append to labels vector
            newLabels.append(forceLabel)
        forcesStorage.setColumnLabels(newLabels)

        # Add data
        for iRow in range(dataArray.shape[0]):
            row = osim.ArrayDouble()
            for iCol in range(dataArray.shape[1]):
                row.append(dataArray[iRow, iCol])
            # Add data to storage
            forcesStorage.append(time[iRow], row)

        # Set name for storage object
        forcesStorage.setName(os.path.split(dynaFile)[-1].split('.')[0] + '_grf')

        # Write to file
        forcesStorage.printResult(forcesStorage,
                                  os.path.split(dynaFile)[-1].split('.')[0] + '_grf',
                                  os.path.join('..', 'data', participant, 'dynamic'),
                                  0.001, '.mot')

        # Create external loads files
        # Note that external loads are applied to feet based on timing of force plate contact and event name

        # Create the external loads
        forceXML = osim.ExternalLoads()

        # Convert forces to time-series table for easier use
        forcesTable = osim.TimeSeriesTable(os.path.join('..', 'data', participant,
                                                        'dynamic', os.path.split(dynaFile)[-1].split('.')[0] + '_grf.mot'))

        # Extract the vertical force data from the two plates
        vForce1 = forcesTable.getDependentColumn('ground_force_1_vy').to_numpy()
        vForce2 = forcesTable.getDependentColumn('ground_force_2_vy').to_numpy()

        # Identify contact indices and times on plates based on force threshold
        forceThreshold = 20

        # Get the c3d data with events using btk
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(dynaFile)
        reader.Update()
        c3dData = reader.GetOutput()

        # Get events
        eventData = {'eventName': [], 'eventFrame': [], 'eventTime': []}
        moreEvents = True
        eventInd = 0
        while moreEvents:
            try:
                event = c3dData.GetEvent(eventInd)
                eventData['eventName'].append(event.GetContext() + ' ' + event.GetLabel())
                eventData['eventFrame'].append(event.GetFrame())
                eventData['eventTime'].append(event.GetTime())
                eventInd += 1
            except:
                moreEvents = False
        events = pd.DataFrame.from_dict(eventData)
        events.sort_values(by='eventFrame', inplace=True)
        events.reset_index(drop=True, inplace=True)

        # Identify limb contact for force plate 1
        v1_ind = np.argmax(vForce1 > forceThreshold)
        v1_t = forcesTable.getIndependentColumn()[v1_ind]

        # Get the strike foot based on time difference between foot strike and event
        strikeFoot = events['eventName'][np.argmin(np.array(np.abs(events['eventTime'] - v1_t)))].split(' ')[0]

        # Create external loads for plate 1
        grf1 = osim.ExternalForce()
        grf1.setName('grf1')
        if strikeFoot == 'Left':
            grf1.setAppliedToBodyName('calcn_l')
        elif strikeFoot == 'Right':
            grf1.setAppliedToBodyName('calcn_r')
        else:
            raise ValueError('Left or right foot not identified as strike foot...')
        grf1.setForceExpressedInBodyName('ground')
        grf1.setPointExpressedInBodyName('ground')
        grf1.setForceIdentifier('ground_force_1_v')
        grf1.setPointIdentifier('ground_force_1_p')
        grf1.setTorqueIdentifier('ground_force_1_m')
        forceXML.cloneAndAppend(grf1)

        # Identify limb contact for force plate 2
        v2_ind = np.argmax(vForce2 > forceThreshold)
        v2_t = forcesTable.getIndependentColumn()[v2_ind]

        # Get the strike foot based on time difference between foot strike and event
        strikeFoot = events['eventName'][np.argmin(np.array(np.abs(events['eventTime'] - v2_t)))].split(' ')[0]

        # Create external loads for plate 2
        grf2 = osim.ExternalForce()
        grf2.setName('grf2')
        if strikeFoot == 'Left':
            grf2.setAppliedToBodyName('calcn_l')
        elif strikeFoot == 'Right':
            grf2.setAppliedToBodyName('calcn_r')
        else:
            raise ValueError('Left or right foot not identified as strike foot...')
        grf2.setForceExpressedInBodyName('ground')
        grf2.setPointExpressedInBodyName('ground')
        grf2.setForceIdentifier('ground_force_2_v')
        grf2.setPointIdentifier('ground_force_2_p')
        grf2.setTorqueIdentifier('ground_force_2_m')
        forceXML.cloneAndAppend(grf2)

        # Set GRF datafile in external loads
        forceXML.setDataFileName(os.path.split(dynaFile)[-1].split('.')[0] + '_grf.mot')

        # Write to file
        forceXML.printToXML(os.path.join('..', 'data', participant_id, 'dynamic',
                                         os.path.split(dynaFile)[-1].split('.')[0] + '_grf.xml'))

        # Extract trial event timings

        # Get first and second foot strike timings
        firstFootStrike = np.min((v1_t, v2_t))
        secondFootStrike = np.max((v1_t, v2_t))

        # Find the toe off after the first foot strike as the start time for simulations
        # Find the subsequent toe off to indicate the corresponding end time for simulations
        offEvents = [events.iloc[ii]['eventTime'] for ii in range(len(events)) if
                     'Off' in events.iloc[ii]['eventName'] and events.iloc[ii]['eventTime'] > firstFootStrike]
        offEvents.sort()
        startTime = offEvents[0]
        endTime = offEvents[1]

        # Store timings in a dictionary and save
        eventData = {'startTime': startTime, 'endTime': endTime}
        with open(os.path.join('..', 'data', participant_id, 'events',
                               os.path.split(dynaFile)[-1].split('.')[0] + '.pkl'), 'wb') as pklFile:
            pickle.dump(eventData, pklFile, protocol=pickle.HIGHEST_PROTOCOL)


# =========================================================================
# Extract participant data
# =========================================================================

if __name__ == '__main__':

    # Identify selection of valid participants
    # -------------------------------------------------------------------------

    # Set list to include valid participants
    valid_participants = []

    # Loop through participants
    for participant in participantList:

        # Determine if participant has the necessary good trials for the defined speed

        # Create a dictionary to store boolean status for participant conditions
        # Defaults to False and will be changed if appropriate
        conditionStatus = {condition: False for condition in conditionList}
        conditionTrial = {condition: None for condition in conditionList}

        # An initial check is required to see if the force plate info is there for the participant
        if int(participant) in trials['subjectId'].to_list():

            # Check for trials from condition
            for condition in conditionList:

                # Get condition labels from dataframe
                conditionLabels = [label for label in trials.columns if label.startswith(condition + '_')]

                # Extract the trial FP contacts for current condition
                fpContacts = [trials.loc[(trials['subjectId'] == int(participant))][label].values[0] for label in conditionLabels]

                # Find where trial has the two FP contacts (i.e. 12 or 21)
                validTrials = [contact == 12 or contact == 21 for contact in fpContacts]

                # Check if condition is useable
                if np.sum(validTrials) > 0:

                    # Convert condition status if appropriate
                    conditionStatus[condition] = True

                    # Make the random selection of the trial to use from the viable trials
                    random.seed(int(participant))
                    trialSelect = random.choice(list(np.where(validTrials)[0]))
                    conditionTrial[condition] = conditionLabels[trialSelect]

            # Create variable whether to process participant based on all conditions being valid
            # Add to valid list if this is the case
            if all(conditionStatus[condition] for condition in conditionList):
                valid_participants.append(participant)

    # Randomly select the participants to extract
    # -------------------------------------------------------------------------

    # Get the valid male and female participants
    valid_participants_m = anthropometrics.loc[(anthropometrics['gender'] == 'M') &
                                               (anthropometrics['subjectID'].astype(str).isin(valid_participants))]['subjectID'].astype(str).to_list()
    valid_participants_f = anthropometrics.loc[(anthropometrics['gender'] == 'W') &
                                               (anthropometrics['subjectID'].astype(str).isin(valid_participants))]['subjectID'].astype(str).to_list()

    # Set the seed so that participant selection is consistent for male participants
    random.seed(12345)
    select_participants_m = random.sample(valid_participants_m, n_participants)

    # Set the seed so that participant selection is consistent for female participants
    random.seed(54321)
    select_participants_f = random.sample(valid_participants_f, n_participants)

    # Set final list of selected participants
    select_participants = select_participants_m + select_participants_f

    # Extract data for selected participants
    # -------------------------------------------------------------------------
    for participant in tqdm(select_participants):
            extract_participant(participant)

    # Print confirmation
    # -------------------------------------------------------------------------
    print('Data extracted for all participants...')

    # Exit terminal to avoid any funny business
    # -------------------------------------------------------------------------
    os._exit(00)

# %% ---------- end of extract_data.py ---------- %% #
