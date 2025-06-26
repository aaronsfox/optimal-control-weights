# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This code is a preliminary step in converting the C3D files from the Crienkinge2023
    dataset to OpenSim useable formats.

    Each participants data was set-up in 'dynamic' and 'static' folders, and the files
    labelled as 'walk.c3d' and 'static.c3d', respectively.

    TODO:
        > Add detailed notes on participant selection
        > Consider EMG data?

"""

# =========================================================================
# Import packages
# =========================================================================

import opensim as osim
import os
import numpy as np
import btk
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt

# =========================================================================
# Set-up
# =========================================================================

# Set participant list
participant_list = ['SUBJ01']

# Create the two rotations needed for data
rot1 = osim.Rotation(np.deg2rad(-90), osim.Vec3(1,0,0))
rot2 = osim.Rotation(np.deg2rad(180), osim.Vec3(0,1,0))

# Set markers to keep
marker_list = ['LFHD', 'RFHD', 'LBHD', 'RBHD',
               'C7', 'T10', 'CLAV', 'STRN',
               'LSHO', 'LELB', 'LWRA', 'LWRB', 'LFIN',
               'RSHO', 'RELB', 'RWRA', 'RWRB', 'RFIN',
               'LASI', 'RASI', 'SACR',
               'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE',
               'RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 'RTOE']

# Set dictionary of joint angles from C3D and corresponding model coordinates
joint_angles = {'LHipAngles_1': 'hip_flexion_l',  # 'LHipAngles_2': 'hip_adduction_l', 'LHipAngles_3': 'hip_rotation_l',
                'RHipAngles_1': 'hip_flexion_r',  # 'RHipAngles_2': 'hip_adduction_r', 'RHipAngles_3': 'hip_rotation_r',
                'LKneeAngles_1': 'knee_angle_l',
                'RKneeAngles_1': 'knee_angle_r',
                'LAnkleAngles_1': 'ankle_angle_l',
                'RAnkleAngles_1': 'ankle_angle_r',
                'RThoraxAngles_1': 'lumbar_ext'}
invert_angle = {'hip_flexion_l': False,  # 'hip_adduction_l': False, 'hip_rotation_l': False,
                'hip_flexion_r': False,  # 'hip_adduction_r': False, 'hip_rotation_r': False,
                'knee_angle_l': True,
                'knee_angle_r': True,
                'ankle_angle_l': False,
                'ankle_angle_r': False,
                'lumbar_ext': True}

# Set vertical force threshold for force plate data
vertForceThreshold = 20

# Set force data filter frequency
forceFiltFreq = 20

# =========================================================================
# Define functions
# =========================================================================

# Convert participants data
# -------------------------------------------------------------------------
def convert_participant_data(participant_id):

    """
    :param participant_id (string): id of the participant to extract data for
    :return:
    """

    # Static trial
    # -------------------------------------------------------------------------

    # Get the path to the static file
    static_file = os.path.join('..', 'data', participant_id, 'static', 'static.c3d')

    # Construct opensim c3d object
    c3dFile = osim.C3DFileAdapter()
    c3dFile.setLocationForForceExpression(osim.C3DFileAdapter.ForceLocation_CenterOfPressure)

    # Read in the static trial
    staticC3D = c3dFile.read(static_file)

    # Get markers table
    staticMarkers = c3dFile.getMarkersTable(staticC3D)

    # Other data seems to come across from C3D file, so remove any columns that aren't in marker list
    for col in staticMarkers.getColumnLabels():
        if col not in marker_list:
            staticMarkers.removeColumn(col)

    # Rotate marker data
    for iRow in range(staticMarkers.getNumRows()):
        # Apply the rotations
        staticMarkers.setRowAtIndex(iRow, rot1.multiply(staticMarkers.getRowAtIndex(iRow)))

    # Write static markers to TRC file
    osim.TRCFileAdapter().write(staticMarkers, os.path.join('..', 'data', participant_id, 'static',
                                                            os.path.split(static_file)[-1].split('.')[0] + '.trc'))

    # Dynamic file
    # -------------------------------------------------------------------------

    # Get the path to the dynamic file
    dynamic_file = os.path.join('..', 'data', participant_id, 'dynamic', 'walk.c3d')

    # Get the c3d data with events using btk
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(dynamic_file)
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

    # Get the first frame number to adjust event timings to align with OpenSim data
    first_frame = c3dData.GetFirstFrame()
    sample_rate = c3dData.GetPointFrequency()
    subtract_time = (first_frame - 1) / sample_rate

    # Subtract frames and time from event data
    events['eventFrame'] = events['eventFrame'] - first_frame
    events['eventTime'] = events['eventTime'] - subtract_time

    # Construct opensim c3d object
    c3dFile = osim.C3DFileAdapter()
    c3dFile.setLocationForForceExpression(osim.C3DFileAdapter.ForceLocation_CenterOfPressure)

    # Read in the dynamic trial
    dynaC3D = c3dFile.read(dynamic_file)

    # Get markers table and flattened version for joint angles
    dynaMarkers = c3dFile.getMarkersTable(dynaC3D)
    dynaFlat = dynaMarkers.clone().flatten()

    # Other data seems to come across from C3D file, so remove any columns that aren't in marker list
    for col in dynaMarkers.getColumnLabels():
        if col not in marker_list:
            dynaMarkers.removeColumn(col)

    # Rotate marker data
    for iRow in range(dynaMarkers.getNumRows()):
        # Apply the rotation
        dynaMarkers.setRowAtIndex(iRow, rot1.multiply(dynaMarkers.getRowAtIndex(iRow)))

    # Check if moving in negative x-direction and data needs to be flipped for consistency
    # Check by examining average difference in pelvis marker (negativ difference needs to be flipped)
    if np.diff(dynaMarkers.flatten().getDependentColumn('RASI_1').to_numpy()).mean() < 0:
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
                                                          os.path.split(dynamic_file)[-1].split('.')[0] + '.trc'))

    # Get the desired joint angles to keep
    for col in dynaFlat.getColumnLabels():
        if col not in joint_angles.keys():
            dynaFlat.removeColumn(col)

    # Rename column labels
    new_labels = [joint_angles[col] for col in dynaFlat.getColumnLabels()]
    dynaFlat.setColumnLabels(new_labels)

    # Build new table to have clean converted radians data
    angles_table = osim.TimeSeriesTable()

    # Add data
    for row_ind in range(dynaFlat.getNumRows()):
        angles_table.appendRow(dynaFlat.getIndependentColumn()[row_ind],
                               osim.RowVector().createFromMat(
                                   np.array([
                                       np.deg2rad(dynaFlat.getDependentColumn(col).to_numpy())[row_ind] * -1 if invert_angle[col] \
                                       else np.deg2rad(dynaFlat.getDependentColumn(col).to_numpy())[row_ind] for col in new_labels
                                   ])))

    # Set column labels
    angles_table.setColumnLabels(new_labels)

    # Write plug in gait joint angles to file
    osim.STOFileAdapter().write(angles_table, os.path.join('..', 'data', participant_id, 'dynamic',
                                                           os.path.split(dynamic_file)[-1].split('.')[0] + '_PiG-angles.mot'))
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

    # Get isolated force labels
    gen_force_labels = [col.rsplit('_', 1)[0] for col in forcesFlat.getColumnLabels() if col.startswith('f') and col.endswith('_2')]
    gen_force_ind = [int(ff[-1]) for ff in gen_force_labels]

    # Create filter to apply to force data
    # Get the sampling rate
    fs = 1 / np.diff(np.array(forcesFlat.getIndependentColumn())).mean()
    # Define low-pass Butterworth filter
    nyq = 0.5 * fs
    normCutoff = forceFiltFreq / nyq
    b, a = butter(4, normCutoff, btype='low', analog=False)

    # Loop through forces and clean up non-contact data. Filter forces here as well
    for force_ind in gen_force_ind:
        # Get vertical force index
        vert_ind = list(forcesFlat.getColumnLabels()).index(f'f{force_ind}_2')
        # Get indices in array that match this force index
        all_plate_ind = [list(forcesFlat.getColumnLabels()).index(col) for col in forcesFlat.getColumnLabels() if int(col[1]) == force_ind]
        # Find where vertical force is above threshold
        in_plate_contact = dataArray[:, vert_ind] > vertForceThreshold
        # Find on and off periods
        contact_periods = []
        start = None
        min_frames = 100
        for ii, val in enumerate(in_plate_contact):
            if val:
                if start is None:
                    start = ii  # Start of a new True period
            else:
                if start is not None:
                    if ii - start >= min_frames:
                        contact_periods.append((start, ii - 1))
                    start = None  # Reset start
        # Handle case where the list ends on a True run
        if start is not None and len(in_plate_contact) - start >= min_frames:
            contact_periods.append((start, len(in_plate_contact) - 1))
        # Create boolean mask for plate contact
        plate_mask = np.zeros_like(dataArray[:, vert_ind], dtype=bool)
        for start, end in contact_periods:
            plate_mask[start:end + 1] = True  # `end + 1` because end is inclusive
        # Set all non-contact force plate data to zero and filter
        for pp in all_plate_ind:
            dataArray[~plate_mask,pp] = 0.0
            for period in contact_periods:
                dataArray[period[0]:period[1]+1, pp] = filtfilt(b, a, dataArray[period[0]:period[1]+1, pp])

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
    forcesStorage.setName(os.path.split(dynamic_file)[-1].split('.')[0] + '_grf')

    # Write to file
    forcesStorage.printResult(forcesStorage,
                              os.path.split(dynamic_file)[-1].split('.')[0] + '_grf',
                              os.path.join('..', 'data', participant_id, 'dynamic'),
                              0.001, '.mot')

    # Bring back force data for help with external loads processing
    force_table = osim.TimeSeriesTable(os.path.join('..', 'data', participant_id, 'dynamic','walk_grf.mot'))

    # Create external loads files
    # Note that external loads are applied to feet based on timing of force plate contact and event name

    # Create the external loads
    forceXML = osim.ExternalLoads()

    # Extract the vertical force data from the four plates
    vert_force = {f'force_{ii+1}': force_table.getDependentColumn(f'ground_force_{ii+1}_vy').to_numpy() for ii in range(4)}

    # Identify limb contact for each force plate
    for force_number in vert_force.keys():
        # First check if any contact with this plate is made
        if any(vert_force[force_number] > 50):
            # Get number indicator from label
            force_ind = int(force_number.split('_')[1])
            # Identify contact index and time
            contact_ind = np.argmax(vert_force[force_number] > vertForceThreshold)
            contact_time = force_table.getIndependentColumn()[contact_ind]
            # Get the strike foot based on time difference between foot strike and event
            strike_foot = events['eventName'][np.argmin(np.array(np.abs(events['eventTime'] - contact_time)))].split(' ')[0]
            # Create external load for current plate
            grf = osim.ExternalForce()
            grf.setName(f'grf{force_ind}')
            if strike_foot == 'Left':
                grf.setAppliedToBodyName('calcn_l')
            elif strike_foot == 'Right':
                grf.setAppliedToBodyName('calcn_r')
            else:
                raise ValueError('Left or right foot not identified as strike foot...')
            grf.setForceExpressedInBodyName('ground')
            grf.setPointExpressedInBodyName('ground')
            grf.setForceIdentifier(f'ground_force_{force_ind}_v')
            grf.setPointIdentifier(f'ground_force_{force_ind}_p')
            grf.setTorqueIdentifier(f'ground_force_{force_ind}_m')
            forceXML.cloneAndAppend(grf)

    # Set GRF datafile in external loads
    forceXML.setDataFileName(os.path.split(dynamic_file)[-1].split('.')[0] + '_grf.mot')

    # Write to file
    forceXML.printToXML(os.path.join('..', 'data', participant_id, 'dynamic',
                                     os.path.split(dynamic_file)[-1].split('.')[0] + '_grf.xml'))

    # Extract trial event timings
    # -------------------------------------------------------------------------

    # Get the timing of foot contact on the middle plate as this is the start we want
    start_ind = np.argmax(vert_force['force_2'] > vertForceThreshold)
    start_time = force_table.getIndependentColumn()[start_ind]
    # Get the strike foot based on time difference between foot strike and event
    strike_foot = events['eventName'][np.argmin(np.array(np.abs(events['eventTime'] - start_time)))].split(' ')[0]

    # Find the subsequent foot strike for the side identified on plate 2

    # Find the toe off after the first foot strike as the start time for simulations
    # Find the subsequent toe off to indicate the corresponding end time for simulations
    next_foot_events = [events.iloc[ii]['eventTime'] for ii in range(len(events)) if
                        f'{strike_foot} Foot Strike' in events.iloc[ii]['eventName'] and events.iloc[ii]['eventTime'] > (start_time + 0.25)]
    next_foot_events.sort()
    end_time = next_foot_events[0]

    # Bring marker and force data in as new tables to trim to gait cycle time frame
    final_marker_data = osim.TimeSeriesTableVec3(os.path.join('..', 'data', participant_id, 'dynamic',
                                                              os.path.split(dynamic_file)[-1].split('.')[0] + '.trc'))
    final_force_data = osim.TimeSeriesTable(os.path.join('..', 'data', participant_id, 'dynamic',
                                                         os.path.split(dynamic_file)[-1].split('.')[0] + '_grf.mot'))

    # Trim to identified times
    final_marker_data.trim(start_time, end_time)
    final_force_data.trim(start_time, end_time)

    # Write to file
    osim.TRCFileAdapter().write(final_marker_data, os.path.join('..', 'data', participant_id, 'dynamic',
                                                                os.path.split(dynamic_file)[-1].split('.')[0] + '.trc'))
    osim.STOFileAdapter().write(final_force_data, os.path.join('..', 'data', participant_id, 'dynamic',
                                                               os.path.split(dynamic_file)[-1].split('.')[0] + '_grf.mot'))


# =========================================================================
# Extract participant data
# =========================================================================

if __name__ == '__main__':

    # COnvert data for selected participants
    # -------------------------------------------------------------------------
    for participant in tqdm(participant_list):
            convert_participant_data(participant)

    # Print confirmation
    # -------------------------------------------------------------------------
    print('Data extracted for all participants...')

    # Exit terminal to avoid any funny business
    # -------------------------------------------------------------------------
    os._exit(00)

# %% ---------- end of extract_data.py ---------- %% #
