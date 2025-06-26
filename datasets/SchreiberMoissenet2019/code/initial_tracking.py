# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    NOTES:
        > This just keeps becoming more of a disaster...

    TODO:
        > add script notes here...note that all steps (including MRS) need to have been run for this
        > consider mesh refinement approach?

"""

# =========================================================================
# Import packages
# =========================================================================

import opensim as osim
import os
import shutil
import numpy as np
import re
import argparse

# =========================================================================
# Flags for running analysis
# =========================================================================

# TODO: change this to have all participants in this script?

# Set participant ID to run
participant = '2014014'

# Set the trial name
trial = 'C4_04'

# Set the model variant
modelVariant = '2d'

# # Parse out input arguments
# TODO: probably not needed in this script
# # -------------------------------------------------------------------------
# parser = argparse.ArgumentParser()
#
# # Participant
# parser.add_argument('-p', '--participant', action = 'store', type = str,
#                     help = 'Enter the participant ID (an integer in this dataset)')
#
# # Tracking weight
# parser.add_argument('-w', '--weight', action = 'store', type = float,
#                     help = 'Enter weight for tracking goal in optimal control problem')
#
# # Trial name
# parser.add_argument('-t', '--trial', action = 'store', type = str,
#                     help = 'Enter trial name for simulation which must match a trial for the participant (e.g. C4_01)')
#
# # Parse out the arguments
# args = parser.parse_args()
# participant = args.participant
# trackingWeight = args.weight
# trialName = args.trial

# =========================================================================
# Set-up
# =========================================================================

# Set the participant list based on the codes in the raw folder
# See README.MD in dataset folder for details on setting up raw data
participant_list = [ii for ii in os.listdir(os.path.join('..', 'data')) if os.path.isdir(os.path.join('..', 'data', ii))]

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', 'model', 'Geometry'))

# Set the weights for the terms in the objective function
# These weights are mostly derived from Denton & Umberger (2023), see: https://doi.org/10.1002/cnm.3777
# The decision process behind these settings was to give a reasonable balance of minimising
# effort and matching reference data well. The weight on the term that minimises the derivatives
# of auxiliary variables is set as low as possible while still resulting in smooth activations.
globalControlEffortWeight = 1.0e-1  # much lower than Denton & Umberger
globalStateTrackingWeight = 1.0e0
globalContactTrackingWeight = 5.0e-1  # TODO: does this then also need to be reduced?
globalAuxDerivWeight = 1.0e-3

# Set intervals for mesh refinement
meshIntervals = [5, 12, 25]

# Set actuator forces to append to upper body coordinates
actForces = {
    'lumbar_ext': {'actuatorType': 'torque', 'optForce': 100.0},
    'lumbar_bend': {'actuatorType': 'torque', 'optForce': 100.0},
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

# Set list of muscles to consider elastic tendons for
elasticTendons = {
    '2d': ['gastroc_r', 'gastroc_l', 'soleus_r', 'soleus_l'],
    '3d': ['gaslat_r', 'gaslat_l', 'gasmed_r', 'gasmed_l', 'soleus_r', 'soleus_l']
}

# Create overall realistic bounds for joint coordinates
coord_bounds = {'2d': {
    'pelvis_tx': [],
    'pelvis_ty': [],
    'pelvis_tilt': [np.deg2rad(-25),np.deg2rad(25)],
    'hip_flexion_r': [np.deg2rad(-20),np.deg2rad(60)],
    'knee_angle_r': [np.deg2rad(-90),np.deg2rad(0)],
    'ankle_angle_r': [np.deg2rad(-35),np.deg2rad(20)],
    'mtp_angle_r': [np.deg2rad(-25),np.deg2rad(75)],
    'hip_flexion_l': [np.deg2rad(-20),np.deg2rad(60)],
    'knee_angle_l': [np.deg2rad(-90),np.deg2rad(0)],
    'ankle_angle_l': [np.deg2rad(-35),np.deg2rad(20)],
    'mtp_angle_l': [np.deg2rad(-25),np.deg2rad(75)],
    'lumbar_ext': [np.deg2rad(-40),np.deg2rad(5)],
    'shoulder_flexion_r': [np.deg2rad(-30),np.deg2rad(30)],
    'elbow_flexion_r': [np.deg2rad(0),np.deg2rad(90)],
    'shoulder_flexion_l': [np.deg2rad(-30),np.deg2rad(30)],
    'elbow_flexion_l': [np.deg2rad(0),np.deg2rad(90)]
    }
}

# TODO: any other settings?

# =========================================================================
# Define functions
# =========================================================================

# Function to run initial tracking simulation
# -------------------------------------------------------------------------
def run_initial_sim(participant_id, trial_name, model_variant='3d'):

    # Organise and set-up files
    # -------------------------------------------------------------------------

    # Create over-arching folder for data
    os.makedirs(os.path.join('..', 'data', participant_id, 'initial'), exist_ok=True)
    os.makedirs(os.path.join('..', 'data', participant_id, 'initial', trial_name, model_variant), exist_ok=True)

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..', 'data', participant_id, 'initial', trial_name, model_variant))

    # Copy external loads file to simulation directory
    # Note these come from the experimental tracking folder as they have been edited for 2d model
    shutil.copyfile(os.path.join('..', '..', '..', 'exp', trial_name, model_variant, f'{participant_id}_{trial_name}_grf.mot'),
                    f'{participant_id}_{trial_name}_grf.mot')
    shutil.copyfile(os.path.join('..', '..', '..', 'exp', trial_name, model_variant, f'{participant_id}_{trial_name}_grf.xml'),
                    f'{participant_id}_{trial_name}_grf.xml')

    # Copy experimental tracking kinematics to simulation directory
    # Note that this uses the version that has been adjusted based on contact sphere penetration
    shutil.copyfile(os.path.join('..', '..', '..', 'exp', trial_name, model_variant,
                                 f'{participant_id}_{trial_name}_exp-tracking-solution-adjusted_{model_variant}.sto'),
                    f'{participant_id}_{trial_name}_exp-tracking-solution-adjusted_{model_variant}.sto')

    # Copy model file to simulation directory
    # Note that this uses the model with optimised parameters from muscle redundancy solver
    shutil.copyfile(os.path.join('..', '..', '..', 'model', f'{participant_id}_model_{model_variant}_optim.osim'),
                    f'{participant_id}_model_{model_variant}_optim.osim')

    # Set-up model for simulation
    # -------------------------------------------------------------------------

    # TODO: is there any adjustment needed to model as in Moco examples? Active force width etc.?
    # TODO: consider reserve actuators?

    # Simplify model to begin with
    model_proc = osim.ModelProcessor(f'{participant_id}_model_{model_variant}_optim.osim')
    model_proc.append(osim.ModOpIgnoreTendonCompliance())
    model_proc.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    model_proc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

    # Load in the model
    osim_model = osim.Model(f'{participant_id}_model_{model_variant}_optim.osim')
    osim_model = model_proc.process()

    # Unlock mtp and subtalar joints for tracking simulations
    osim_model.getJointSet().get('mtp_r').get_coordinates(0).set_locked(False)
    osim_model.getJointSet().get('mtp_l').get_coordinates(0).set_locked(False)
    if model_variant == '3d':
        osim_model.getJointSet().get('subtalar_r').get_coordinates(0).set_locked(False)
        osim_model.getJointSet().get('subtalar_l').get_coordinates(0).set_locked(False)

    # Add upper body coordinate actuators to model
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

    # Set rigid tendons in the model
    # TODO: use if tendon compliance re-enabled earlier
    # for muscle_ind in range(osim_model.getMuscles().getSize()):
    #     if osim_model.getMuscles().get(muscle_ind).getName() not in elasticTendons[model_variant]:
    #         # Set to rigid tendon
    #         osim_model.getMuscles().get(muscle_ind).set_ignore_tendon_compliance(True)

    # Finalise model connections
    osim_model.finalizeConnections()

    # Print model to file
    osim_model.printToXML(f'{participant_id}_{trial_name}_initial-model_{model_variant}.osim')

    # Set up tracking simulation
    # -------------------------------------------------------------------------

    # Create tracking tool
    track = osim.MocoTrack()
    track.setName(f'{participant_id}_{trial_name}_initial_{model_variant}')

    # Set model in tool
    track_model_proc = osim.ModelProcessor(f'{participant_id}_{trial_name}_initial-model_{model_variant}.osim')
    track.setModel(track_model_proc)

    # Set the timings
    # To be safe just use start and end of tracking solution
    track.set_initial_time(
        osim.TimeSeriesTable(f'{participant_id}_{trial_name}_exp-tracking-solution-adjusted_{model_variant}.sto').getIndependentColumn()[0])
    track.set_final_time(
        osim.TimeSeriesTable(f'{participant_id}_{trial_name}_exp-tracking-solution-adjusted_{model_variant}.sto').getIndependentColumn()[-1])

    # Set states tracking parameters
    # -------------------------------------------------------------------------

    # Load in the solution file and export to states table
    states_data = osim.MocoTrajectory(
        f'{participant_id}_{trial_name}_exp-tracking-solution-adjusted_{model_variant}.sto').exportToStatesTable()

    # Add in zero data for the originally locked coordinates as these won't be in original solution
    # Create the array to add in
    zero_data = np.zeros(states_data.getNumRows())
    # Add in the data for coordinate values and speeds
    states_data.appendColumn(osim_model.getCoordinateSet().get('mtp_angle_r').getAbsolutePathString() + '/value',
                             osim.Vector().createFromMat(zero_data))
    states_data.appendColumn(osim_model.getCoordinateSet().get('mtp_angle_r').getAbsolutePathString() + '/speed',
                             osim.Vector().createFromMat(zero_data))
    states_data.appendColumn(osim_model.getCoordinateSet().get('mtp_angle_l').getAbsolutePathString() + '/value',
                             osim.Vector().createFromMat(zero_data))
    states_data.appendColumn(osim_model.getCoordinateSet().get('mtp_angle_l').getAbsolutePathString() + '/speed',
                             osim.Vector().createFromMat(zero_data))
    if model_variant == '3d':
        states_data.appendColumn(osim_model.getCoordinateSet().get('subtalar_angle_r').getAbsolutePathString() + '/value',
                                 osim.Vector().createFromMat(zero_data))
        states_data.appendColumn(osim_model.getCoordinateSet().get('subtalar_angle_r').getAbsolutePathString() + '/speed',
                                 osim.Vector().createFromMat(zero_data))
        states_data.appendColumn(osim_model.getCoordinateSet().get('subtalar_angle_l').getAbsolutePathString() + '/value',
                                 osim.Vector().createFromMat(zero_data))
        states_data.appendColumn(osim_model.getCoordinateSet().get('subtalar_angle_l').getAbsolutePathString() + '/speed',
                                 osim.Vector().createFromMat(zero_data))

    # Write tracking states data to file
    osim.STOFileAdapter().write(states_data, f'{participant_id}_{trial_name}_tracking-states_{model_variant}.sto')

    # Set the reference file
    table_proc = osim.TableProcessor(f'{participant_id}_{trial_name}_tracking-states_{model_variant}.sto')
    track.setStatesReference(table_proc)

    # Set to ignore unused columns
    track.set_allow_unused_references(True)

    # Set global tracking weight
    track.set_states_global_tracking_weight(globalStateTrackingWeight)

    # Turn off tracking position derivaties (i.e. speeds) as these are provided in states
    track.set_track_reference_position_derivatives(False)

    # Set tracked states to guess
    track.set_apply_tracked_states_to_guess(True)

    # Initialise to a Moco study and problem to finalise
    # -------------------------------------------------------------------------

    # Get study and problem
    study = track.initialize()
    problem = study.updProblem()

    # Add initial activation goal
    # -------------------------------------------------------------------------

    # TODO: problem seems to struggle with this active?

    # # For all muscles with activation dynamics, the initial activation and initial excitation should be the same.
    # # Without this goal, muscle activation may undesirably start at its maximum possible value as only excitation
    # # is penalised
    # initial_act = osim.MocoInitialActivationGoal('initial_activation')
    # problem.addGoal(initial_act)

    # Get and modify control effort goal
    # -------------------------------------------------------------------------

    # Get a reference to the MocoControlCost goal and set parameters
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'))
    effort.setWeight(globalControlEffortWeight)
    effort.setExponent(2)

    # Get and modify states tracking goal
    # -------------------------------------------------------------------------

    # Get a reference to the states tracking goal
    tracking = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'))

    # Here we scale individual tracking weights with the data range
    # The states tracking goal has a function to do this, but it doesn't work with the
    # purely zero data added into the states for mtp_angle or subtalar_angle. Therefore
    # the process is replicated here and these zero joints (along with pelvis_tx) have
    # a zero weight set to them.

    # TODO: this may over-weight something like pelvis_ty given small range...maybe consider absolute values too?

    # Set individual state weights based on the data range
    for coord_ind in range(osim_model.updCoordinateSet().getSize()):
        # Get the name and absolute path of the coordinate
        coord_name = osim_model.updCoordinateSet().get(coord_ind).getName()
        coord_path = osim_model.updCoordinateSet().get(coord_ind).getAbsolutePathString()
        # Check if coordinate is one we want to track
        if coord_name not in ['pelvis_tx', 'mtp_angle_r', 'mtp_angle_l', 'subtalar_angle_r', 'subtalar_angle_l']:
            # Identify the range for the value and speed from the tracking data
            value_range = np.ptp(states_data.getDependentColumn(f'{coord_path}/value').to_numpy())
            speed_range = np.ptp(states_data.getDependentColumn(f'{coord_path}/speed').to_numpy())
            # Set weight value (this is the same process used in MocoStateTrackingGoal.cpp)
            value_weight = 1.0 / value_range
            speed_weight = 1.0 / speed_range
            # Set weight for states
            tracking.setWeightForState(f'{coord_path}/value', value_weight)
            tracking.setWeightForState(f'{coord_path}/speed', speed_weight)
            # tracking.setWeightForState(f'{coord_path}/value', 1.0)  # TODO: standard or normalised?
            # tracking.setWeightForState(f'{coord_path}/speed', 1.0)  # TODO: standard or normalised?
        # Otherwise set weight for state to zero
        else:
            tracking.setWeightForState(f'{coord_path}/value', 0.0)
            tracking.setWeightForState(f'{coord_path}/speed', 0.0)

    # In lieu of tracking pelvis_tx set an average speed goal
    # -------------------------------------------------------------------------

    # TODO: could this still be here while tracking pelvis_tx?
    # Maybe not, seems to struggle with both but individually produce similar solutions...

    # Create the average speed goal
    speed_goal = osim.MocoAverageSpeedGoal('speed')

    # Determine average speed from the states data
    avg_speed = states_data.getDependentColumn(
        osim_model.getCoordinateSet().get('pelvis_tx').getAbsolutePathString() + '/speed').to_numpy().mean()

    # Set speed in goal
    speed_goal.set_desired_average_speed(avg_speed)

    # Add to problem
    problem.addGoal(speed_goal)

    # Add contact tracking goal
    # -------------------------------------------------------------------------

    # Set right and left contact sphere groups
    forcesLeftFoot = osim.StdVectorString()
    forcesRightFoot = osim.StdVectorString()
    if model_variant == '3d':
        for forceInd in range(osim_model.updForceSet().getSize()):
            if osim_model.getForceSet().get(forceInd).getAbsolutePathString().lower().startswith('/forceset/contact'):
                if osim_model.getForceSet().get(forceInd).getName().endswith('_l'):
                    forcesLeftFoot.append(osim_model.updForceSet().get(forceInd).getAbsolutePathString())
                elif osim_model.getForceSet().get(forceInd).getName().endswith('_r'):
                    forcesRightFoot.append(osim_model.updForceSet().get(forceInd).getAbsolutePathString())
    if model_variant == '2d':
        for forceInd in range(osim_model.updForceSet().getSize()):
            if osim_model.getForceSet().get(forceInd).getAbsolutePathString().lower().startswith('/forceset/contact'):
                if osim_model.getForceSet().get(forceInd).getAbsolutePathString().lower()[-3:-1] == '_l':
                    forcesLeftFoot.append(osim_model.updForceSet().get(forceInd).getAbsolutePathString())
                elif osim_model.getForceSet().get(forceInd).getAbsolutePathString().lower()[-3:-1] == '_r':
                    forcesRightFoot.append(osim_model.updForceSet().get(forceInd).getAbsolutePathString())

    # Read in external loads to identify tracking groups
    ex_loads = osim.ExternalLoads(f'{participant_id}_{trial_name}_grf.xml', True)

    # Set alternate frames for tracking GRF bodies
    altFramesRightFoot = osim.StdVectorString()
    altFramesRightFoot.append('/bodyset/toes_r')
    altFramesLeftFoot = osim.StdVectorString()
    altFramesLeftFoot.append('/bodyset/toes_l')

    # Set the left and right contact tracking groups
    if ex_loads.get(0).getAppliedToBodyName() == 'calcn_l':
        trackLeftGRF = osim.MocoContactTrackingGoalGroup(forcesLeftFoot, ex_loads.get(0).getName(), altFramesLeftFoot)
        trackRightGRF = osim.MocoContactTrackingGoalGroup(forcesRightFoot, ex_loads.get(1).getName(), altFramesRightFoot)
    else:
        trackLeftGRF = osim.MocoContactTrackingGoalGroup(forcesLeftFoot, ex_loads.get(1).getName(), altFramesLeftFoot)
        trackRightGRF = osim.MocoContactTrackingGoalGroup(forcesRightFoot, ex_loads.get(0).getName(), altFramesRightFoot)

    # Create plane contact tracking goal for 2D data
    if model_variant == '2d':
        contactTracking = osim.MocoContactTrackingGoal('contact', globalContactTrackingWeight)
        contactTracking.setExternalLoadsFile(f'{participant_id}_{trial_name}_grf.xml')
        contactTracking.addContactGroup(trackLeftGRF)
        contactTracking.addContactGroup(trackRightGRF)
        # contactTracking.setNormalizeTrackingError(True)  # TODO: determine if this is desirable? won't work with zeroed data...
        contactTracking.setProjection('plane')
        contactTracking.setProjectionVector(osim.Vec3(0, 0, 1))
        problem.addGoal(contactTracking)
    # TODO: otherwise...create 3d goal
    elif model_variant == '3d':
        print('TODO: add contact tracking here...')

    # Add symmetry goal to permit simulating one step
    # -------------------------------------------------------------------------

    # TODO: not sure whether I'm applying this correctly...even fails with just knee, and seems to be enforcing more consistent symmetry?
    # Seems to work with some individual pairs added when muscle model is simplified and no initial activation goal...
    # Seems to cause huge problems when left and right are included together?
    # Even just using knee_r and ankle_l thought the simulation seems to lose it's mind...
    # Could be something to do with bounds? What if initial bounds are taken out with just knee)r and ankle_r again?
    # Seems like initial bounds are needed...
    # Seemed to work before, but now it doesn't...

    # Create symmetry goal
    symmetry = osim.MocoPeriodicityGoal('symmetry')

    # Test some single pair
    # And corresponding pairs that have worked singularly
    # for coord in ['knee_angle_r', 'ankle_angle_r', 'knee_angle_l', 'ankle_angle_l']:
    for coord in ['knee_angle_r', 'ankle_angle_r']:
        coord_path = osim_model.getCoordinateSet().get(coord).getAbsolutePathString()
        coord_path_opp = osim_model.getCoordinateSet().get(coord[:-2]+'_l').getAbsolutePathString()
        symmetry.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/value',
                                                           coord_path_opp + '/value'))
    symmetry.addStatePair(osim.MocoPeriodicityGoalPair('/jointset/ground_pelvis/pelvis_tx/speed'))

    # TODO: ensure this is set-up to work with both 2d and 3d models --- there will need to be some negated pairings for 3d model...

    # # Symmetric coordinate values and speeds (except for pelvis_tx)
    # for coord_ind in range(osim_model.getNumCoordinates()):
    #     # Get coordinate name
    #     coord = osim_model.getCoordinateSet().get(coord_ind).getName()
    #     # Avoid setting pelvis translation periodicity here
    #     if coord not in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
    #         # Check for right side
    #         if coord.endswith('_r'):
    #             # Add symmetry with left side pair for value and speed
    #             coord_path = osim_model.getCoordinateSet().get(coord).getAbsolutePathString()
    #             coord_path_opp = osim_model.getCoordinateSet().get(coord[:-2]+'_l').getAbsolutePathString()
    #             symmetry.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/value',
    #                                                                coord_path_opp + '/value'))
    #             symmetry.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/speed',
    #                                                                coord_path_opp + '/speed'))
    #         # Check for left side
    #         elif coord.endswith('_l'):
    #             # Add symmetry with right side pair for value and speed
    #             coord_path = osim_model.getCoordinateSet().get(coord).getAbsolutePathString()
    #             coord_path_opp = osim_model.getCoordinateSet().get(coord[:-2] + '_r').getAbsolutePathString()
    #             symmetry.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/value',
    #                                                                coord_path_opp + '/value'))
    #             symmetry.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/speed',
    #                                                                coord_path_opp + '/speed'))
    #         # Otherwise add matching coordinate value and speed pairs
    #         # TODO: in 3d version some will need to be negated
    #         else:
    #             coord_path = osim_model.getCoordinateSet().get(coord).getAbsolutePathString()
    #             symmetry.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/value'))
    #             symmetry.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/speed'))
    #
    # # Pair pelvis translations with themselves
    # symmetry.addStatePair(
    #     osim.MocoPeriodicityGoalPair(osim_model.getCoordinateSet().get('pelvis_ty').getAbsolutePathString() + '/value'))
    # symmetry.addStatePair(
    #     osim.MocoPeriodicityGoalPair(osim_model.getCoordinateSet().get('pelvis_ty').getAbsolutePathString() + '/speed'))
    # if model_variant == '3d':
    #     symmetry.addStatePair(
    #         osim.MocoPeriodicityGoalPair(osim_model.getCoordinateSet().get('pelvis_tz').getAbsolutePathString() + '/value'))
    #     symmetry.addStatePair(
    #         osim.MocoPeriodicityGoalPair(osim_model.getCoordinateSet().get('pelvis_tz').getAbsolutePathString() + '/speed'))

    # # Symmetric muscle activations and tendon forces
    # # TODO: ensure this is set-up to work with both 2d and 3d models
    # for musc_ind in range(osim_model.getMuscles().getSize()):
    #     # Get muscle name
    #     musc = osim_model.getMuscles().get(musc_ind).getName()
    #     # Check for right side
    #     if musc.endswith('_r'):
    #         # Add symmetry with left side pair for activation and normalised tendon force
    #         musc_path = osim_model.getMuscles().get(musc).getAbsolutePathString()
    #         musc_path_opp = osim_model.getMuscles().get(musc[:-2]+'_l').getAbsolutePathString()
    #         symmetry.addStatePair(osim.MocoPeriodicityGoalPair(musc_path + '/activation',
    #                                                                musc_path_opp + '/activation'))
    #         # Tendon force if using elastic tendon
    #         if musc in elasticTendons[model_variant]:
    #             symmetry.addStatePair(osim.MocoPeriodicityGoalPair(musc_path + '/normalized_tendon_force',
    #                                                                musc_path_opp + '/normalized_tendon_force'))
    #     # Check for left side
    #     elif musc.endswith('_l'):
    #         # Add symmetry with right side pair for activation and normalised tendon force
    #         musc_path = osim_model.getMuscles().get(musc).getAbsolutePathString()
    #         musc_path_opp = osim_model.getMuscles().get(musc[:-2] + '_r').getAbsolutePathString()
    #         symmetry.addStatePair(osim.MocoPeriodicityGoalPair(musc_path + '/activation',
    #                                                            musc_path_opp + '/activation'))
    #         # Tendon force if using elastic tendon
    #         if musc in elasticTendons[model_variant]:
    #             symmetry.addStatePair(osim.MocoPeriodicityGoalPair(musc_path + '/normalized_tendon_force',
    #                                                                musc_path_opp + '/normalized_tendon_force'))

    # # Symmetric controls
    # # TODO: ensure this is set-up to work with both 2d and 3d models ---
    # control_names = problem.createRep().createControlInfoNames()
    # for control in control_names:
    #     # Check for right side muscle control
    #     if control.endswith('_r'):
    #         # Pair with left side control
    #         symmetry.addControlPair(osim.MocoPeriodicityGoalPair(control, control[:-2]+'_l'))
    #     # Check for left side muscle contrp;
    #     elif control.endswith('_l'):
    #         # Pair with right side control
    #         symmetry.addControlPair(osim.MocoPeriodicityGoalPair(control, control[:-2] + '_r'))
    #     # Check for right side torque actuator
    #     if control.endswith('_r_torque'):
    #         # Pair with left side torque actuator
    #         symmetry.addControlPair(osim.MocoPeriodicityGoalPair(control, control[:-9]+'_l_torque'))
    #     # Check for left side torque actuator
    #     if control.endswith('_l_torque'):
    #         # Pair with right side torque actuator
    #         symmetry.addControlPair(osim.MocoPeriodicityGoalPair(control, control[:-9] + '_r_torque'))
    #     else:
    #         # Pair with self
    #         # TODO: negated controls for some?
    #         symmetry.addControlPair(osim.MocoPeriodicityGoalPair(control))

    # Add to problem
    problem.addGoal(symmetry)

    # Set bounds in problem
    # -------------------------------------------------------------------------

    # TODO: new approach to bounds seems to be messing up problem solution
    # TODO: moreso seems initial activation goal seems to make it harder to solve. There seems to be high activation initially without this though...

    # # Muscle bounds
    # problem.setStateInfoPattern('/forceset/.*/normalized_tendon_force', [0, 1.5], [], [])  # TODO: reactivate if using elastic tendons
    problem.setStateInfoPattern('/forceset/.*/activation', [0.01, 2.0], [], [])  # allow muscles to over-activate rather than increasing force

    # Set bounds on joint coordinate values using the states tracking data as a reference
    # Given pelvis_tx is untracked and controlled via average speed, only an initial bound is set for this
    # Only total bounds are set for MTP and subtalar angles, where reasonable values are used
    # For other coordinates, initial bounds are set to be within 20% of the range for their tracking values
    # Final bounds are not set as they are more so dictated by the dynamics of the problem (see: https://simtk.org/plugins/phpBB/viewtopicPhpbb.php?f=1815&t=19781&p=0&start=0&view=&sid=97392e0361ee4ab8a89d4d6cd7134afe)
    # Maximum and minimum bounds are set to be +/- 20% of the range for the coordinate

    # Create a dictionary to store initial bounds for coordinates
    initial_coord_bounds = {}

    # Loop through coordinates
    for coord_ind in range(osim_model.updCoordinateSet().getSize()):
        # Get the name and absolute path of the coordinate
        coord_name = osim_model.updCoordinateSet().get(coord_ind).getName()
        coord_path = osim_model.updCoordinateSet().get(coord_ind).getAbsolutePathString()
        # First check for pelvis_tx and pelvis_tz
        if coord_name == 'pelvis_tx' or coord_name == 'pelvis_tz':
            # Set the initial value to match that of the states data
            initial_coord_bounds[coord_name] = states_data.getDependentColumn(f'{coord_path}/value').to_numpy()[0]
        # Next check for pelvis_ty
        elif coord_name == 'pelvis_ty':
            # Allow some small variation outside of the initial value
            initial_val = states_data.getDependentColumn(f'{coord_path}/value').to_numpy()[0]
            val_range = np.ptp(states_data.getDependentColumn(f'{coord_path}/value').to_numpy())
            initial_coord_bounds[coord_name] = [initial_val - (val_range * 0.10), initial_val + 0.1]
            # Allow some small variation outside of the range of values
            min_val = states_data.getDependentColumn(f'{coord_path}/value').to_numpy().min()
            max_val = states_data.getDependentColumn(f'{coord_path}/value').to_numpy().min()
            coord_bounds[model_variant][coord_name] = [min_val - (val_range * 0.20), max_val + 0.1]
        # Next check for coordinates outside of subtalar or mtp angles
        elif coord_name not in ['mtp_angle_r', 'mtp_angle_l', 'subtalar_angle_r', 'subtalar_angle_l']:
            # Allow some small variation outside of the initial value relative to value range
            initial_val = states_data.getDependentColumn(f'{coord_path}/value').to_numpy()[0]
            val_range = np.ptp(states_data.getDependentColumn(f'{coord_path}/value').to_numpy())
            initial_coord_bounds[coord_name] = [initial_val - (val_range * 0.20), initial_val + (val_range * 0.20)]
        # Lastly set empty values for any remaining desired coordinates
        elif coord_name in ['mtp_angle_r', 'mtp_angle_l', 'subtalar_angle_r', 'subtalar_angle_l']:
            initial_coord_bounds[coord_name] = []

    # Check that any initial bounds do not exceed total coordinate bounds
    for coord_name in coord_bounds[model_variant].keys():
        if len(coord_bounds[model_variant][coord_name]) > 0 and len(initial_coord_bounds[coord_name]) > 0:
            # Replace lower initial value if less than total bounds
            if initial_coord_bounds[coord_name][0] < coord_bounds[model_variant][coord_name][0]:
                initial_coord_bounds[coord_name][0] = coord_bounds[model_variant][coord_name][0]
            # Replace upper initial value if greater than total bounds
            if initial_coord_bounds[coord_name][1] > coord_bounds[model_variant][coord_name][1]:
                initial_coord_bounds[coord_name][1] = coord_bounds[model_variant][coord_name][1]

    # Set coordinate value bounds in problem
    for coord_name in coord_bounds[model_variant].keys():
        # Get coordinate path
        coord_path = osim_model.updCoordinateSet().get(coord_name).getAbsolutePathString()
        # Set in problem
        problem.setStateInfo(f'{coord_path}/value',
                             coord_bounds[model_variant][coord_name],
                             initial_coord_bounds[coord_name],
                             [])

    # Solve problem via mesh refinement approach
    # -------------------------------------------------------------------------

    # Loop through mesh intervals
    for mesh_int in meshIntervals:

        # Define and configure the solver
        solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

        # Set solver options
        solver.set_optim_max_iterations(2000)
        solver.set_num_mesh_intervals(mesh_int)
        solver.set_optim_constraint_tolerance(1e-3)
        solver.set_optim_convergence_tolerance(1e-3)
        solver.set_minimize_implicit_auxiliary_derivatives(True)
        solver.set_implicit_auxiliary_derivatives_weight(globalAuxDerivWeight)

        # Reset problem to check any issues
        solver.resetProblem(problem)

        # Solve the problem
        solution = study.solve()

    # # Option to visualise solution
    # study.visualize(solution)

    # Save files and finalize
    # -------------------------------------------------------------------------

    # TODO: other outputs like metabolcs and muscle forces...

    # Write solution to file
    # TODO: different names?
    if solution.isSealed():
        solution.unseal()
    solution.write(f'{participant_id}_{trial_name}_reference-solution_{model_variant}_meshInterval-{meshInterval}.sto')

    # Extract ground reaction forces

    # Set contact sphere names
    contactSpheres_r = osim.StdVectorString()
    contactSpheres_l = osim.StdVectorString()
    for ii in range(osim_model.getForceSet().getSize()):
        if osim_model.getForceSet().get(ii).getName().lower().startswith('contact'):
            if model_variant == '3d':
                if osim_model.getForceSet().get(ii).getName().endswith('_r'):
                    contactSpheres_r.append(osim_model.getForceSet().get(ii).getAbsolutePathString())
                elif osim_model.getForceSet().get(ii).getName().endswith('_l'):
                    contactSpheres_l.append(osim_model.getForceSet().get(ii).getAbsolutePathString())
            elif model_variant == '2d':
                if 'Ground_R' in osim_model.getForceSet().get(ii).getName():
                    contactSpheres_r.append(osim_model.getForceSet().get(ii).getAbsolutePathString())
                elif 'Ground_L' in osim_model.getForceSet().get(ii).getName():
                    contactSpheres_l.append(osim_model.getForceSet().get(ii).getAbsolutePathString())

    # Create external loads table
    forces_table = osim.createExternalLoadsTableForGait(osim_model, solution,
                                                        contactSpheres_r, contactSpheres_l)

    # Write to file
    osim.STOFileAdapter().write(forces_table, f'{participant_id}_{trial_name}_reference-solution_{model_variant}_grf.sto')

    # Remove initial tracked states and markers file
    os.remove(f'{participant_id}_{trial_name}_initial_{model_variant}_tracked_states.sto')

    # Return to home directory
    os.chdir(home_dir)

# =========================================================================
# Start code here ...
# =========================================================================

if __name__ == '__main__':
    print('run main code here')

# %% ---------- end of template.py ---------- %% #
