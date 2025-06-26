# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    TODO: add script notes here...
        > Clean up script
        > Contact parameter optimisation?
        > Track speed goal instead of pelvis_tx?
        > Include muscles for better initial guess later on?
            >> Optimise muscle parameters at scaling prior to doing this

        > Normalised state (or non-normalised) and contact tracking weighted goals WITH bounds seems to work ok

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
# Flags for running analyses
# =========================================================================

# Set participant ID to run
participant = '2014014'

# Set the trial name
trial = 'C4_04'

# # Parse out input arguments
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

# Set weights for torque-driven optimisations
globalCoordinateTrackingWeight = 1e-1  # seems to work well for torque tracking at 0.1
# globalContactTrackingWeight = globalCoordinateTrackingWeight * 1e3
# globalContactTrackingWeight = 0.5
globalContactTrackingWeight = [1e3, 1e3, 1e2]
# globalContactTrackingWeight = [5e-1, 5e-1, 2.5e-1]
globalControlEffortGoal = 1e-3

#Set a scaling factor for tracking the speeds
speedsTrackingScale = 0.01

# Set mesh interval
meshInterval = 25  #TODO: testing coarsely... fix to 25...or 50? mesh refinement approach

# Set kinematics filter frequency
kinematicFiltFreq = 6  # TODO: is this needed?

# Create a dictionary of the coordinate tasks for tracking simulations
coordTasks = {'pelvis_tx': 2.5e1, 'pelvis_ty': 1.0e2, 'pelvis_tz': 2.5e1,
              'pelvis_tilt': 7.5e2, 'pelvis_list': 2.5e2, 'pelvis_rotation': 5.0e1,
              'hip_flexion_r': 7.5e1, 'hip_adduction_r': 5.0e1, 'hip_rotation_r': 1.0e1,
              'knee_angle_r': 1.0e2, 'ankle_angle_r': 1.0e2,
              'subtalar_angle_r': 1.0e0, 'mtp_angle_r': 1.0e-1,
              'hip_flexion_l': 7.5e1, 'hip_adduction_l': 5.0e1, 'hip_rotation_l': 1.0e1,
              'knee_angle_l': 1.0e2, 'ankle_angle_l': 1.0e2,
              'subtalar_angle_l': 1.0e0, 'mtp_angle_l': 1.0e-1,
              'lumbar_ext': 7.5e1, 'lumbar_bend': 5.0e1, 'lumbar_rota': 2.5e1,
              'shoulder_flexion_r': 5.0e0, 'shoulder_adduction_r': 5.0e0, 'shoulder_rotation_r': 5.0e0,
              'elbow_flexion_r': 5.0e0,
              'shoulder_flexion_l': 5.0e0, 'shoulder_adduction_l': 5.0e0, 'shoulder_rotation_l': 5.0e0,
              'elbow_flexion_l': 5.0e0,
              }

# Set actuator forces to drive tracking simulations
actForces = {
    # TODO: are residuals needed for this to solve nicely?
    'pelvis_tx': {'actuatorType': 'residual', 'optForce': 2.5e1},  # need to be higher --- testing
    'pelvis_ty': {'actuatorType': 'residual', 'optForce': 2.5e1},  # need to be higher --- testing
    'pelvis_tz': {'actuatorType': 'residual', 'optForce': 2.5e1},  # need to be higher --- testing
    'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 2.5e1},  # need to be higher --- testing
    'pelvis_list': {'actuatorType': 'residual', 'optForce': 2.5e1},  # need to be higher --- testing
    'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 2.5e1},  # need to be higher --- testing
    'hip_flexion_r': {'actuatorType': 'torque', 'optForce': 300.0},
    'hip_adduction_r': {'actuatorType': 'torque', 'optForce': 200.0},
    'hip_rotation_r': {'actuatorType': 'torque', 'optForce': 100.0},
    'knee_angle_r': {'actuatorType': 'torque', 'optForce': 300.0},
    'ankle_angle_r': {'actuatorType': 'torque', 'optForce': 200.0},
    'subtalar_angle_r': {'actuatorType': 'torque', 'optForce': 100.0},
    'mtp_angle_r': {'actuatorType': 'torque', 'optForce': 50.0},
    'hip_flexion_l': {'actuatorType': 'torque', 'optForce': 300.0},
    'hip_adduction_l': {'actuatorType': 'torque', 'optForce': 200.0},
    'hip_rotation_l': {'actuatorType': 'torque', 'optForce': 100.0},
    'knee_angle_l': {'actuatorType': 'torque', 'optForce': 300.0},
    'ankle_angle_l': {'actuatorType': 'torque', 'optForce': 200.0},
    'subtalar_angle_l': {'actuatorType': 'torque', 'optForce': 100.0},
    'mtp_angle_l': {'actuatorType': 'torque', 'optForce': 50.0},
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

# Set actuator forces to assist inverse simulations
actForces_inverse = {
    # TODO: are residuals needed for this to solve nicely?
    'pelvis_tx': {'actuatorType': 'residual', 'optForce': 2.5e0},  # need to be higher --- testing
    'pelvis_ty': {'actuatorType': 'residual', 'optForce': 2.5e0},  # need to be higher --- testing
    'pelvis_tz': {'actuatorType': 'residual', 'optForce': 2.5e0},  # need to be higher --- testing
    'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 2.5e0},  # need to be higher --- testing
    'pelvis_list': {'actuatorType': 'residual', 'optForce': 2.5e0},  # need to be higher --- testing
    'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 2.5e0},  # need to be higher --- testing
    'hip_flexion_r': {'actuatorType': 'reserve', 'optForce': 1.0},
    'hip_adduction_r': {'actuatorType': 'reserve', 'optForce': 1.0},
    'hip_rotation_r': {'actuatorType': 'reserve', 'optForce': 1.0},
    'knee_angle_r': {'actuatorType': 'reserve', 'optForce': 1.0},
    'ankle_angle_r': {'actuatorType': 'reserve', 'optForce': 1.0},
    'subtalar_angle_r': {'actuatorType': 'reserve', 'optForce': 1.0},
    'mtp_angle_r': {'actuatorType': 'reserve', 'optForce': 1.0},
    'hip_flexion_l': {'actuatorType': 'reserve', 'optForce': 1.0},
    'hip_adduction_l': {'actuatorType': 'reserve', 'optForce': 1.0},
    'hip_rotation_l': {'actuatorType': 'reserve', 'optForce': 1.0},
    'knee_angle_l': {'actuatorType': 'reserve', 'optForce': 1.0},
    'ankle_angle_l': {'actuatorType': 'reserve', 'optForce': 1.0},
    'subtalar_angle_l': {'actuatorType': 'reserve', 'optForce': 1.0},
    'mtp_angle_l': {'actuatorType': 'reserve', 'optForce': 1.0},
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

# # Set markers to look up for ground contact constraints
# contactMarkers = [
#     'R_FM1_ground', 'R_FM2_ground', 'R_FM5_ground', 'R_FCC_ground',
#     'R_FM1_mid_ground', 'R_FM5_mid_ground',
#     # 'R_FM2_mid_ground',
#     'L_FM1_ground', 'L_FM2_ground', 'L_FM5_ground', 'L_FCC_ground',
#     'L_FM1_mid_ground', 'L_FM5_mid_ground'
#     # 'L_FM2_mid_ground',
# ]

# # Contact sphere settings
# stiffness = 3067776
# dissipation = 2.0
# staticFriction = 0.8
# dynamicFriction = 0.8
# viscousFriction = 0.5
# transitionVelocity = 0.2
#
# # Set contact sphere radii
# contactHeelRadius = 0.030
# contactMidfootRadius = 0.025
# contactToeRadius = 0.020

# =========================================================================
# Define functions
# =========================================================================

# Function to run torque-driven inverse simulation
# -------------------------------------------------------------------------
def run_inverse_sim(participant_id, trial_name, model_file_name, ik_file_name,
                    model_variant='3d'):

    """
    :param participant_id:
    :param trial_name:
    :param model_file_name:
    :param ik_file_name:
    :param model_variant:
    :return:
    """

    # Organise and set-up files
    # -------------------------------------------------------------------------

    # Create over-arching folder for data
    os.makedirs(os.path.join('..', 'data', participant_id, 'inverse'), exist_ok=True)
    os.makedirs(os.path.join('..', 'data', participant_id, 'inverse', trial_name), exist_ok=True)

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..', 'data', participant_id, 'inverse', trial_name))

    # Copy external loads file to simulation directory
    shutil.copyfile(os.path.join('..', '..', 'dynamic', f'{participant_id}_{trial_name}_grf.mot'),
                    f'{participant_id}_{trial_name}_grf.mot')
    shutil.copyfile(os.path.join('..', '..', 'dynamic', f'{participant_id}_{trial_name}_grf.xml'),
                    f'{participant_id}_{trial_name}_grf.xml')

    # Copy IK file to simulation directory
    shutil.copyfile(os.path.join('..', '..', 'ik', trial_name, ik_file_name), ik_file_name)

    # Copy model file to simulation directory
    shutil.copyfile(os.path.join('..', '..', 'scaling', model_file_name), model_file_name)

    # Set-up model for simulation
    # -------------------------------------------------------------------------

    # Use a processor to remove muscles from model
    model_proc = osim.ModelProcessor(model_file_name)
    # model_proc.append(osim.ModOpRemoveMuscles())

    # TODO: consider appropriate muscle parameters
    model_proc.append(osim.ModOpIgnoreTendonCompliance())  # TODO: seems to cause significant problems with feasibility of problem when on...
    # model_proc.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    # model_proc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

    # Process to edit model further
    osim_model = model_proc.process()

    # Unlock mtp and subtalar joints for tracking simulations
    osim_model.getJointSet().get('mtp_r').get_coordinates(0).set_locked(False)
    osim_model.getJointSet().get('mtp_l').get_coordinates(0).set_locked(False)
    if model_variant == '3d':
        osim_model.getJointSet().get('subtalar_r').get_coordinates(0).set_locked(False)
        osim_model.getJointSet().get('subtalar_l').get_coordinates(0).set_locked(False)

    # Add coordinate actuators to model
    # Depending on if model is 2d or 3d variant not all actuators will need to be created
    model_coordinates = [osim_model.getCoordinateSet().get(ii).getName() for ii in range(osim_model.getNumCoordinates())]
    # Loop through coordinates
    for coordinate in actForces_inverse:
        # Check if in model coordinates
        if coordinate in model_coordinates:
            # Create actuator
            actu = osim.CoordinateActuator()
            # Set name
            actu.setName(f'{coordinate}_{actForces_inverse[coordinate]["actuatorType"]}')
            # Set coordinate
            actu.setCoordinate(osim_model.updCoordinateSet().get(coordinate))
            # Set optimal force
            actu.setOptimalForce(actForces_inverse[coordinate]['optForce'])
            # Set min and max control
            actu.setMinControl(np.inf * -1)
            actu.setMaxControl(np.inf * 1)
            # Append to model force set
            osim_model.updForceSet().cloneAndAppend(actu)

    # TODO: Turn off contact geometry forces if necessary
    for forceInd in range(osim_model.getForceSet().getSize()):
        if osim_model.updForceSet().get(forceInd).getName().lower().startswith('contact'):
            # Switch off the contact force
            osim_model.updForceSet().get(forceInd).set_appliesForce(False)

    # Finalise model connections
    osim_model.finalizeConnections()

    # Print model to file
    # TODO: separate name for contact vs. no contact models?
    osim_model.printToXML(f'{participant_id}_{trial_name}_inverseModel_{model_variant}.osim')

    # Adjust kinematic data for tracking simulation
    # -------------------------------------------------------------------------

    # Load in the kinematic data
    kinematics_data = osim.Storage(ik_file_name)

    # Create a copy of the kinematics data to alter the column labels in
    states_data = osim.Storage(ik_file_name)

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

    # # Bring back in to remove the pelvis_ty coordinate for appropriate contact tracking
    # new_states = osim.TimeSeriesTable(f'{trial_name}_coordinates_{model_variant}.sto')
    # new_states.removeColumn('/jointset/ground_pelvis/pelvis_ty/value')
    # osim.STOFileAdapter.write(new_states, f'{trial_name}_coordinates_{model_variant}_no-pelvis-ty.sto')

    # Set up inverse simulation
    # -------------------------------------------------------------------------

    # Create inverse tool
    inverse = osim.MocoInverse()
    inverse.setName(f'{participant_id}_{trial_name}_inverse_{model_variant}')

    # Create model processor
    model_proc = osim.ModelProcessor(f'{participant_id}_{trial_name}_inverseModel_{model_variant}.osim')

    # Append external loads
    # TODO: contact tracking with adjusted parameters instead of external loads? Doesn't really work
    model_proc.append(osim.ModOpAddExternalLoads(f'{participant_id}_{trial_name}_grf.xml'))

    # Set model in tool
    inverse.setModel(model_proc)

    # Set the coordinates reference file
    table_proc = osim.TableProcessor(f'{trial_name}_coordinates_{model_variant}.sto')
    table_proc.append(osim.TabOpLowPassFilter(kinematicFiltFreq))  # TODO: necessary? probably for inverse sim...
    inverse.setKinematics(table_proc)

    # By default, Moco gives an error if the kinematics contains extra columns.
    # Here, we tell Moco to allow (and ignore) those extra columns.
    inverse.set_kinematics_allow_extra_columns(True)

    # Set the timings
    # To be safe just use start and end of coordinates file
    inverse.set_initial_time(osim.TimeSeriesTable(f'{trial_name}_coordinates_{model_variant}.sto').getIndependentColumn()[0])
    inverse.set_final_time(osim.TimeSeriesTable(f'{trial_name}_coordinates_{model_variant}.sto').getIndependentColumn()[-1])
    inverse.set_mesh_interval(0.01)

    # Initialise to a Moco study and problem to finalise
    # -------------------------------------------------------------------------

    # Get study and problem
    study = inverse.initialize()
    problem = study.updProblem()

    # # Get a reference to the problem model
    # problem_model = problem.updModel()

    # Remove the pelvis_ty prescribed motion from the model
    # TODO: doesn't seem like you can have a coordinate removed from function set
    # for f_ind in range(problem_model.getComponent('/position_motion').get_functions().getSize()):
    #     if problem_model.getComponent('/position_motion').get_functions().get(f_ind).getName() == '/jointset/ground_pelvis/pelvis_ty':
    #         problem_model.getComponent('/position_motion').get_functions().remove(f_ind)
    #         break

    # Update control effort goal

    # Get a reference to the MocoControlCost goal and set parameters
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('excitation_effort'))
    effort.setWeight(0.1)  # TODO: manually set here - should be in options
    effort.setExponent(2)

    # Update individual weights in control effort goal
    # Put higher weight on residual use to heavily penalise
    effort.setWeightForControlPattern('/forceset/.*_residual', 10.0)
    # Put a higher weights on reserves to penalise their use
    effort.setWeightForControlPattern('/forceset/.*_reserve', 10.0)
    # Put standard weight on upper body torque actuators
    effort.setWeightForControlPattern('/forceset/.*_torque', 1.0)

    # # Create a states tracking goal to manage pelvis_ty tracking
    #
    # # Create the goal
    # tracking = osim.MocoStateTrackingGoal('state_tracking', 0.1)
    #
    # # Set the coordinate references
    # # Use same as inverse
    # tracking.setReference(table_proc)
    #
    # # Allow extra columns
    # tracking.setAllowUnusedReferences(True)
    #
    # # Set weights so that alone pelvis_ty are tracked in this goal
    # for coord_ind in range(osim_model.updCoordinateSet().getSize()):
    #     # Get name and absolute path to coordinate
    #     coord_name = osim_model.updCoordinateSet().get(coord_ind).getName()
    #     coord_path = osim_model.updCoordinateSet().get(coord_ind).getAbsolutePathString()
    #     if coord_name == 'pelvis_ty':
    #         tracking.setWeightForState(f'{coord_path}/value', 1.0)
    #     else:
    #         tracking.setWeightForState(f'{coord_path}/value', 0.0)
    #
    # # Add to problem
    # problem.addGoal(tracking)

    # # Optimise contact sphere parameters
    # # -------------------------------------------------------------------------
    # # TODO: just testing on stiffness here
    #
    # # Get contact parameter sphere names
    # osim_model.initSystem()
    # contact_spheres = []
    # for forceInd in range(osim_model.updForceSet().getSize()):
    #     if 'Sphere' in osim_model.getForceSet().get(forceInd).getConcreteClassName():
    #         contact_spheres.append(osim_model.updForceSet().get(forceInd).getAbsolutePathString())
    #
    # # Pair up matching spheres from left and right foot to ensure symmetry
    # # TODO: will this work for 3D model? Should make contact sphere naming conventions the same during scaling to avoid this issue
    # r_spheres = {}
    # l_spheres = {}
    # for sphere in contact_spheres:
    #     match = re.search(r'_(R|L)(\d+)$', sphere)
    #     if match:
    #         side, number = match.groups()
    #         if side == 'R':
    #             r_spheres[number] = sphere
    #         else:
    #             l_spheres[number] = sphere
    # paired_spheres = [(r_spheres[n], l_spheres[n]) for n in r_spheres if n in l_spheres]
    #
    # # Create the parameters to optimise for each sphere pair
    # sphere_params = {}
    # for param_ind in range(len(paired_spheres)):
    #     # Create the parameter
    #     param_name = f'sphere_{param_ind + 1}'
    #     sphere_params[param_name] = osim.MocoParameter()
    #     sphere_params[param_name].setName(param_name)
    #     # Append the component paths
    #     sphere_params[param_name].appendComponentPath(paired_spheres[param_ind][0])
    #     sphere_params[param_name].appendComponentPath(paired_spheres[param_ind][1])
    #     # Set the property name to optimise
    #     # TODO: repeat for different properties
    #     sphere_params[param_name].setPropertyName('stiffness')
    #     # Set bounds
    #     # Currently set to 20% either side of current value
    #     # Get the current parameter value
    #     # TODO: would need to change for different parameters
    #     param_val = osim.SmoothSphereHalfSpaceForce.safeDownCast(
    #         osim_model.getForceSet().get(paired_spheres[param_ind][0].split('/')[-1])).get_stiffness()
    #     # Set bounds
    #     sphere_params[param_name].setBounds(osim.MocoBounds(param_val * 0.8, param_val * 1.2))
    #     # Add to problem
    #     problem.addParameter(sphere_params[param_name])

    # Set bounds in problem
    # -------------------------------------------------------------------------

    # # Set bounds on joint coordinates using the tracking data as a reference
    # # Pelvis translations are ignored as these typically need to be modified more
    # # Initial and final bounds are set to be within 20% of the range their tracking values
    # # Max and minimum bounds are set to be +/- 20% of the range for the coordinate
    # for state_name in list(ik_tracking.getColumnLabels()):
    #
    #     # Check for joint coordinate state
    #     if state_name.endswith('/value'):  # and not any(coord in stateName for coord in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']):
    #
    #         # Calculate the range, initial and final values for coordinate
    #         valRange = np.ptp(ik_tracking.getDependentColumn(state_name).to_numpy())
    #         initialVal = ik_tracking.getDependentColumn(state_name).to_numpy()[0]
    #         finalVal = ik_tracking.getDependentColumn(state_name).to_numpy()[-1]
    #         maxVal = ik_tracking.getDependentColumn(state_name).to_numpy().max()
    #         minVal = ik_tracking.getDependentColumn(state_name).to_numpy().min()
    #
    #         # Check for pelvis_tx first to make it at least start at the same point
    #         # TODO: could consider speed goal instead of tx?
    #         # TODO: pelvis constraints were initially much tighter, but expanded out for 2D problem as a potential solution
    #         # TODO: is this necessary for all states?
    #         if 'pelvis_tx' in state_name:
    #             # Constrain initial value, leave final value open
    #             problem.setStateInfo(state_name,
    #                                  [minVal - (valRange * 0.2), maxVal + 0.1],  # maxVal + (valRange * 0.2)],
    #                                  initialVal,
    #                                  [finalVal - (valRange * 0.1), finalVal + 0.1],  # finalVal + (valRange * 0.1)]
    #                                  )
    #         # Check for other pelvis translational coordinates and set tighter bounds in problem
    #         elif any(coord in state_name for coord in ['pelvis_ty', 'pelvis_tz']):
    #             problem.setStateInfo(state_name,
    #                                  [minVal - (valRange * 0.2), maxVal + 0.1],  # maxVal + (valRange * 0.2)],
    #                                  [initialVal - (valRange * 0.1), initialVal + 0.1],  # initialVal + (valRange * 0.1)],
    #                                  [finalVal - (valRange * 0.2), finalVal + 0.1],  # finalVal + (valRange * 0.1)]
    #                                  )
    #         # Otherwise set the slightly looser bounds
    #         # elif not any(coord in state_name for coord in ['mtp_angle_r', 'mtp_angle_l']):
    #         else:
    #             problem.setStateInfo(state_name,
    #                                  [minVal - (valRange * 0.2), maxVal + (valRange * 0.2)],
    #                                  [initialVal - (valRange * 0.2), initialVal + (valRange * 0.2)],
    #                                  [finalVal - (valRange * 0.2), finalVal + (valRange * 0.2)])

    # # Muscle bounds
    # problem.setStateInfoPattern('/forceset/.*/normalized_tendon_force', [0, 1.5], [], [])

    # Define and configure the solver
    # -------------------------------------------------------------------------
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

    # Set solver options
    solver.set_optim_max_iterations(1000)
    solver.set_num_mesh_intervals(meshInterval)
    solver.set_optim_constraint_tolerance(1e-3)
    solver.set_optim_convergence_tolerance(1e-3)
    # NOTE: I don't think this helps...maybe weight needs to be quite small...?
    # solver.set_multibody_dynamics_mode('implicit')
    # solver.set_minimize_implicit_multibody_accelerations(True)  # smoothness criterion
    # solver.set_implicit_multibody_accelerations_weight(1e-2)
    # solver.set_parameters_require_initsystem(True)  # TODO: seemingly needs to be true for contact sphere stiffness
    # solver.set_minimize_implicit_auxiliary_derivatives(True)
    # solver.set_implicit_auxiliary_derivatives_weight(0.001)  # TODO: set manually here --- should be in options

    # Reset problem to avoid any issues
    solver.resetProblem(problem)

    # # Switching to implicit mode causes a problem with the guess not having accelerations
    # # These therefore need to be created in the guess
    # guess = solver.getGuess()
    # guess.generateAccelerationsFromSpeeds()
    # solver.setGuess(guess)

    # TODO: only if necessary for contact tracking...?

    # # With parameters there are some difficulties with the guess not including the parameters, so some tweaking is required
    #
    # # Get the initial created guess from the tracked states
    # initial_guess = solver.getGuess()
    #
    # # Push up pelvis_ty in starting guess to avoid contact spheres starting too far in ground
    # # Set the pelvis_ty starting value to a small increment above where it is in coordinates data
    # initial_guess.setState('/jointset/ground_pelvis/pelvis_ty/value',
    #                        initial_guess.getState('/jointset/ground_pelvis/pelvis_ty/value').to_numpy() + 0.05)

    # # Create a guess that will now include the parameters to be optimised
    # created_guess = solver.createGuess()
    #
    # # Set the guess in solver
    # solver.setGuess(created_guess)

    # # Create guess to manipulate
    # guess = solver.createGuess()
    #
    # # Load in inverse solution
    # inverse_solution = osim.MocoTrajectory(f'{participant_id}_{trial_name}_inverse-solution_{model_variant}.sto')
    #
    # # Resample guess to inverse solution time-steps
    # guess.resampleWithNumTimes(inverse_solution.getNumTimes())
    #
    # # Fill states from inverse solution
    # for state_name in guess.getStateNames():
    #     guess.setState(state_name, inverse_solution.getState(state_name).to_numpy())
    #
    # # Fill controls from inverse solution
    # for control_name in guess.getControlNames():
    #     guess.setControl(control_name, inverse_solution.getControl(control_name).to_numpy())
    #
    # # Set guess in solver
    # solver.setGuess(guess)

    # # Reset problem to finalise
    # solver.resetProblem(problem)

    # Solve the problem
    # -------------------------------------------------------------------------
    solution = study.solve()
    # solution.write('test-solution.sto')

    """
        > Some quite strange muscle activations in solution --- maybe need to optimise parameters?
    """

    # Save files and finalize
    # -------------------------------------------------------------------------

    # Write solution to file
    # TODO: different names?
    if solution.isSealed():
        solution.unseal()
    solution.write(f'{participant_id}_{trial_name}_inverse-solution_{model_variant}.sto')

    # Return to home directory
    os.chdir(home_dir)


# Function to run torque-driven tracking simulation
# -------------------------------------------------------------------------
def run_reference_sim(participant_id, trial_name, model_file_name, ik_file_name,
                      model_variant='3d'):

    """
    :param participant_id:
    :param trial_name:
    :param model_file_name:
    :param ik_file_name:
    :param model_variant:
    :return:
    """

    # Organise and set-up files
    # -------------------------------------------------------------------------

    # Create over-arching folder for data
    os.makedirs(os.path.join('..', 'data', participant_id, 'reference'), exist_ok=True)
    os.makedirs(os.path.join('..', 'data', participant_id, 'reference', trial_name), exist_ok=True)

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..', 'data', participant_id, 'reference', trial_name))

    # Copy external loads file to simulation directory
    shutil.copyfile(os.path.join('..', '..', 'dynamic', f'{participant_id}_{trial_name}_grf.mot'),
                    f'{participant_id}_{trial_name}_grf.mot')
    shutil.copyfile(os.path.join('..', '..', 'dynamic', f'{participant_id}_{trial_name}_grf.xml'),
                    f'{participant_id}_{trial_name}_grf.xml')

    # Copy IK file to simulation directory
    shutil.copyfile(os.path.join('..', '..', 'ik', trial_name, ik_file_name), ik_file_name)

    # # Copy inverse solution to simulation directory
    # shutil.copyfile(os.path.join('..', '..', 'inverse', trial_name, f'{participant_id}_{trial_name}_inverse-solution_{model_variant}.sto'),
    #                 f'{participant_id}_{trial_name}_inverse-solution_{model_variant}.sto')

    # Copy model file to simulation directory
    shutil.copyfile(os.path.join('..', '..', 'scaling', model_file_name), model_file_name)

    # Set-up model for simulation
    # -------------------------------------------------------------------------

    # Use a processor to remove muscles from model
    model_proc = osim.ModelProcessor(model_file_name)
    model_proc.append(osim.ModOpRemoveMuscles())
    osim_model = model_proc.process()

    # Unlock mtp and subtalar joints for tracking simulations
    osim_model.getJointSet().get('mtp_r').get_coordinates(0).set_locked(False)
    osim_model.getJointSet().get('mtp_l').get_coordinates(0).set_locked(False)
    if model_variant == '3d':
        osim_model.getJointSet().get('subtalar_r').get_coordinates(0).set_locked(False)
        osim_model.getJointSet().get('subtalar_l').get_coordinates(0).set_locked(False)

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

    # # TODO: Turn off contact geometry forces if necessary
    # for forceInd in range(osim_model.getForceSet().getSize()):
    #     if osim_model.updForceSet().get(forceInd).getName().lower().startswith('contact'):
    #         # Switch off the contact force
    #         osim_model.updForceSet().get(forceInd).set_appliesForce(False)

    # # Add contact geometry to the model
    #
    # # Create the half space for the floor
    # contact_floor = osim.ContactHalfSpace(osim.Vec3(0, 0, 0),
    #                                       osim.Vec3(0, 0, -1.5707963267948966),
    #                                       osim_model.getGround(),
    #                                       'floor')
    #
    # # Connect to model
    # osim_model.addContactGeometry(contact_floor)
    #
    # # Create and add the contact spheres
    # for marker in contactMarkers:
    #
    #     # Set radius specific to marker location
    #     if '_FCC' in marker:
    #         contact_radius = contactHeelRadius
    #     elif '_mid' in marker:
    #         contact_radius = contactMidfootRadius
    #     else:
    #         contact_radius = contactToeRadius
    #
    #     # Get location and frame of marker
    #     location = osim_model.updMarkerSet().get(marker).get_location()
    #     frame = osim.PhysicalFrame.safeDownCast(
    #         osim_model.updBodySet().get(osim_model.updMarkerSet().get(marker).getParentFrameName().split('/')[-1]))
    #
    #     # Update location so that bottom of sphere aligns with bottom of foot
    #     updated_location = osim.Vec3(location.get(0), location.get(1)+(contact_radius/2), location.get(2))
    #
    #     # Create contact geometry
    #     contact_sphere = osim.ContactSphere(contact_radius, updated_location, frame, f'{marker}_contactSphere')
    #
    #     # Connect to model
    #     osim_model.addContactGeometry(contact_sphere)
    #
    #     # Create the sphere force to go with the contact spheres
    #     sphere_force = osim.SmoothSphereHalfSpaceForce(f'{marker}_contactForce',
    #                                                    contact_sphere, contact_floor)
    #
    #     # Set sphere force parameters
    #     sphere_force.set_stiffness(stiffness)
    #     sphere_force.set_dissipation(dissipation)
    #     sphere_force.set_static_friction(staticFriction)
    #     sphere_force.set_dynamic_friction(dynamicFriction)
    #     sphere_force.set_viscous_friction(viscousFriction)
    #     sphere_force.set_transition_velocity(transitionVelocity)
    #     sphere_force.connectSocket_half_space(contact_floor)
    #     sphere_force.connectSocket_sphere(contact_sphere)
    #
    #     # Add the force to model
    #     osim_model.addForce(sphere_force)

    # Finalise model connections
    osim_model.finalizeConnections()

    # Print model to file
    # TODO: separate name for contact vs. no contact models?
    osim_model.printToXML(f'{participant_id}_{trial_name}_referenceModel_{model_variant}.osim')

    # Adjust kinematic data for tracking simulation
    # -------------------------------------------------------------------------

    # Load in the kinematic data
    kinematics_data = osim.Storage(ik_file_name)

    # Create a copy of the kinematics data to alter the column labels in
    states_data = osim.Storage(ik_file_name)

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
    track.setName(f'{participant_id}_{trial_name}_reference_{model_variant}')

    # Create model processor
    track_model_proc = osim.ModelProcessor(f'{participant_id}_{trial_name}_referenceModel_{model_variant}.osim')

    # TODO: Append external loads if necessary - need a 2D version with zeroed horizontal shear forces?
    # track_model_proc.append(osim.ModOpAddExternalLoads(f'{participant_id}_{trial_name}_grf.xml'))

    # Set model in tool
    track.setModel(track_model_proc)

    # Set the coordinates reference file
    table_proc = osim.TableProcessor(f'{trial_name}_coordinates_{model_variant}.sto')
    table_proc.append(osim.TabOpLowPassFilter(kinematicFiltFreq))
    track.setStatesReference(table_proc)

    # Set to ignore unused columns
    track.set_allow_unused_references(True)

    # Set global tracking weight
    track.set_states_global_tracking_weight(globalCoordinateTrackingWeight)

    # Track positive derivaties (i.e. speeds)
    track.set_track_reference_position_derivatives(True)

    # Set tracked states to guess
    track.set_apply_tracked_states_to_guess(True)

    # Set the timings
    # To be safe just use start and end of coordinates file
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
    effort.setWeight(globalControlEffortGoal)
    effort.setExponent(2)

    # TODO: this is already taken care of with optimal force values (low vs. high)
    # # Update individual weights in control effort goal
    # # Put higher weight on residual use
    # effort.setWeightForControlPattern('/forceset/.*_residual', 5.0)
    # # Put heavy weight on the reserve actuators
    # effort.setWeightForControlPattern('/forceset/.*_torque', 1.0)

    # # Create control tracking goal
    #
    # # Create the goal and set weight
    # control_tracking = osim.MocoControlTrackingGoal('control_tracking', 10)  # TODO: set weight appropriately
    #
    # # Set controls to track from inverse solution
    # controls_ref = osim.TableProcessor(
    #     osim.MocoTrajectory(f'{participant_id}_{trial_name}_inverse-solution_{model_variant}.sto').exportToControlsTable())
    # control_tracking.setReference(controls_ref)
    #
    # # Set zero weighting to residuals tracking
    # control_tracking.setWeightForControl('/forceset/pelvis_tx_residual', 0)
    # control_tracking.setWeightForControl('/forceset/pelvis_ty_residual', 0)
    # control_tracking.setWeightForControl('/forceset/pelvis_tz_residual', 0)
    # control_tracking.setWeightForControl('/forceset/pelvis_tilt_residual', 0)
    # control_tracking.setWeightForControl('/forceset/pelvis_list_residual', 0)
    # control_tracking.setWeightForControl('/forceset/pelvis_rotation_residual', 0)
    #
    # # Add to problem
    # problem.addGoal(control_tracking)

    # Update states tracking goal

    # Get a reference to the states tracking goal
    tracking = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'))

    # # Scale individual tracking weights with data range
    # TODO: note that this doesn't work with purely zero mtp values as weight becomes inf
    # tracking.setScaleWeightsWithRange(True)

    # Set individual state weights manually
    for coord_ind in range(osim_model.updCoordinateSet().getSize()):

        # Get name and absolute path to coordinate
        coord_name = osim_model.updCoordinateSet().get(coord_ind).getName()
        coord_path = osim_model.updCoordinateSet().get(coord_ind).getAbsolutePathString()

        # If a task weight is provided, add it in
        if coord_name in list(coordTasks.keys()):
            # Set the weight for state value
            tracking.setWeightForState(f'{coord_path}/value', coordTasks[coord_name])
            # Set the weight for state speed
            tracking.setWeightForState(f'{coord_path}/speed', coordTasks[coord_name] * speedsTrackingScale)

    # # Set some weights to simply be zero
    # tracking.setWeightForState('/jointset/mtp_r/mtp_angle_r/value', 0)
    # tracking.setWeightForState('/jointset/mtp_r/mtp_angle_r/speed', 0)
    # tracking.setWeightForState('/jointset/mtp_l/mtp_angle_l/value', 0)
    # tracking.setWeightForState('/jointset/mtp_l/mtp_angle_l/speed', 0)
    # tracking.setWeightForState('/jointset/ground_pelvis/pelvis_ty/value', 0)
    # tracking.setWeightForState('/jointset/ground_pelvis/pelvis_ty/speed', 0)

    # Add contact tracking goal

    # TODO: update as plane for 2D variant...
    # TODO: only add contact tracking if necessary

    # Set right and left contact sphere groups
    # TODO: not sure these were getting added appropriately for 2D model - fixed but clean-up below
    forcesLeftFoot = osim.StdVectorString()
    forcesRightFoot = osim.StdVectorString()
    if model_variant == '3d':
        for forceInd in range(osim_model.updForceSet().getSize()):
            # if '_contactForce' in osim_model.getForceSet().get(forceInd).getAbsolutePathString():
            if osim_model.getForceSet().get(forceInd).getAbsolutePathString().lower().startswith('/forceset/contact'):
                if osim_model.getForceSet().get(forceInd).getName().endswith('_l'):
                    forcesLeftFoot.append(osim_model.updForceSet().get(forceInd).getAbsolutePathString())
                elif osim_model.getForceSet().get(forceInd).getName().endswith('_r'):
                    forcesRightFoot.append(osim_model.updForceSet().get(forceInd).getAbsolutePathString())
    if model_variant == '2d':
        for forceInd in range(osim_model.updForceSet().getSize()):
            # if '_contactForce' in osim_model.getForceSet().get(forceInd).getAbsolutePathString():
            if osim_model.getForceSet().get(forceInd).getAbsolutePathString().lower().startswith('/forceset/contact'):
                if osim_model.getForceSet().get(forceInd).getAbsolutePathString().lower()[-3:-1] == '_l':
                    forcesLeftFoot.append(osim_model.updForceSet().get(forceInd).getAbsolutePathString())
                elif osim_model.getForceSet().get(forceInd).getAbsolutePathString().lower()[-3:-1] == '_r':
                    forcesRightFoot.append(osim_model.updForceSet().get(forceInd).getAbsolutePathString())

    # Read in external loads to identify tracking groups
    exLoads = osim.ExternalLoads(f'{participant_id}_{trial_name}_grf.xml', True)

    # Set alternate frames for tracking GRF bodies
    altFramesRightFoot = osim.StdVectorString()
    altFramesRightFoot.append('/bodyset/toes_r')
    altFramesLeftFoot = osim.StdVectorString()
    altFramesLeftFoot.append('/bodyset/toes_l')

    # Set the left and right contact tracking groups
    if exLoads.get(0).getAppliedToBodyName() == 'calcn_l':
        trackLeftGRF = osim.MocoContactTrackingGoalGroup(forcesLeftFoot, exLoads.get(0).getName(), altFramesLeftFoot)
        trackRightGRF = osim.MocoContactTrackingGoalGroup(forcesRightFoot, exLoads.get(1).getName(), altFramesRightFoot)
    else:
        trackLeftGRF = osim.MocoContactTrackingGoalGroup(forcesLeftFoot, exLoads.get(1).getName(), altFramesLeftFoot)
        trackRightGRF = osim.MocoContactTrackingGoalGroup(forcesRightFoot, exLoads.get(0).getName(), altFramesRightFoot)

    # # Create plane contact tracking goal for 2D data
    # if model_variant == '2d':
    #     contactTracking = osim.MocoContactTrackingGoal('GRF_tracking', globalContactTrackingWeight[0])
    #     contactTracking.setExternalLoadsFile(f'{participant_id}_{trial_name}_grf.xml')
    #     contactTracking.addContactGroup(trackLeftGRF)
    #     contactTracking.addContactGroup(trackRightGRF)
    #     # contactTracking.setNormalizeTrackingError(True)
    #     contactTracking.setProjection('plane')
    #     contactTracking.setProjectionVector(osim.Vec3(0, 0, 1))
    #     problem.addGoal(contactTracking)

    # Create normalised contact tracking goal in each axis of 3D data
    contactTracking = []
    if model_variant == '3d':
        axis_list = ['X', 'Y', 'Z']
    elif model_variant == '2d':
        axis_list = ['X', 'Y']
    for ii in axis_list:
        contactTracking.append(osim.MocoContactTrackingGoal(f'GRF_tracking_{ii}', globalContactTrackingWeight[axis_list.index(ii)]))
        # contactTracking.append(osim.MocoContactTrackingGoal(f'GRF_tracking_{ii}', globalContactTrackingWeight[['X', 'Y', 'Z'].index(ii)]))
        contactTracking[-1].setExternalLoadsFile(f'{participant_id}_{trial_name}_grf.xml')
        contactTracking[-1].addContactGroup(trackLeftGRF)
        contactTracking[-1].addContactGroup(trackRightGRF)
        contactTracking[-1].setNormalizeTrackingError(True)
        contactTracking[-1].setProjection('vector')
        if ii == 'X':
            contactTracking[-1].setProjectionVector(osim.Vec3(1, 0, 0))
        elif ii == 'Y':
            contactTracking[-1].setProjectionVector(osim.Vec3(0, 1, 0))
        elif ii == 'Z':
            contactTracking[-1].setProjectionVector(osim.Vec3(0, 0, 1))
        # Add to problem
        problem.addGoal(contactTracking[-1])

    # Add speed tracking goal in lieu of tracking pelvis_tx

    # Get average speed from reference file
    ik_tracking = table_proc.process()
    # pelvis_tx = ik_tracking.getDependentColumn('/jointset/ground_pelvis/pelvis_tx/value').to_numpy()
    # time = np.array(ik_tracking.getIndependentColumn())
    # avg_speed = (np.diff(pelvis_tx) / np.diff(time)).mean()
    #
    # # Create goal
    # speedGoal = osim.MocoAverageSpeedGoal('speed')
    # speedGoal.set_desired_average_speed(avg_speed)
    #
    # # Add to problem
    # problem.addGoal(speedGoal)

    # Add constraints to the problem

    # TODO: only constrain if necessary

    # # Constrain contact markers to at minimum be at ground level
    # # Loop through markers to create path constraints
    # for markerInd in range(osim_model.getMarkerSet().getSize()):
    #     if osim_model.getMarkerSet().get(markerInd).getName().lower().startswith('contact'):
    #         # Create constraint
    #         marker = osim_model.getMarkerSet().get(markerInd).getName()
    #         markerConstraint = osim.MocoOutputConstraint()
    #         markerConstraint.setName(f'{marker}_constraint')
    #         # Set path to marker location
    #         markerConstraint.setOutputPath(f'/markerset/{marker}|location')
    #         # Set output index to y-axis
    #         markerConstraint.setOutputIndex(1)
    #         # Create and set the bounds to slightly below ground level and a reasonable height
    #         markerBounds = osim.StdVectorMocoBounds()
    #         markerBounds.append(osim.MocoBounds(-0.01, 0.25))
    #         markerConstraint.updConstraintInfo().setBounds(markerBounds)
    #         # Add to problem
    #         problem.addPathConstraint(markerConstraint)

    # Optimise contact sphere parameters
    # -------------------------------------------------------------------------
    # TODO: just testing on stiffness here --- perhaps combine this into the inverse simulation process after initial tracking?

    # # Get contact parameter sphere names
    # osim_model.initSystem()
    # contact_spheres = []
    # for forceInd in range(osim_model.updForceSet().getSize()):
    #     if 'Sphere' in osim_model.getForceSet().get(forceInd).getConcreteClassName():
    #         contact_spheres.append(osim_model.updForceSet().get(forceInd).getAbsolutePathString())
    #
    # # Pair up matching spheres from left and right foot to ensure symmetry
    # # TODO: will this work for 3D model? Should make contact sphere naming conventions the same during scaling to avoid this issue
    # r_spheres = {}
    # l_spheres = {}
    # for sphere in contact_spheres:
    #     match = re.search(r'_(R|L)(\d+)$', sphere)
    #     if match:
    #         side, number = match.groups()
    #         if side == 'R':
    #             r_spheres[number] = sphere
    #         else:
    #             l_spheres[number] = sphere
    # paired_spheres = [(r_spheres[n], l_spheres[n]) for n in r_spheres if n in l_spheres]
    #
    # # Create the parameters to optimise for each sphere pair
    # sphere_params = {}
    # for param_ind in range(len(paired_spheres)):
    #     # Create the parameter
    #     param_name = f'sphere_{param_ind+1}'
    #     sphere_params[param_name] = osim.MocoParameter()
    #     sphere_params[param_name].setName(param_name)
    #     # Append the component paths
    #     sphere_params[param_name].appendComponentPath(paired_spheres[param_ind][0])
    #     sphere_params[param_name].appendComponentPath(paired_spheres[param_ind][1])
    #     # Set the property name to optimise
    #     # TODO: repeat for different properties
    #     sphere_params[param_name].setPropertyName('stiffness')
    #     # Set bounds
    #     # Currently set to 20% either side of current value
    #     # Get the current parameter value
    #     # TODO: would need to change for different parameters
    #     param_val = osim.SmoothSphereHalfSpaceForce.safeDownCast(
    #         osim_model.getForceSet().get(paired_spheres[param_ind][0].split('/')[-1])).get_stiffness()
    #     # Set bounds
    #     sphere_params[param_name].setBounds(osim.MocoBounds(param_val * 0.8, param_val * 1.2))
    #     # Add to problem
    #     problem.addParameter(sphere_params[param_name])

    # Set bounds in problem
    # -------------------------------------------------------------------------

    # Set bounds on joint coordinates using the tracking data as a reference
    # Pelvis translations are ignored as these typically need to be modified more
    # Initial and final bounds are set to be within 20% of the range their tracking values
    # Max and minimum bounds are set to be +/- 20% of the range for the coordinate
    for state_name in list(ik_tracking.getColumnLabels()):

        # Check for joint coordinate state
        if state_name.endswith('/value'):  # and not any(coord in stateName for coord in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']):

            # Calculate the range, initial and final values for coordinate
            valRange = np.ptp(ik_tracking.getDependentColumn(state_name).to_numpy())
            initialVal = ik_tracking.getDependentColumn(state_name).to_numpy()[0]
            finalVal = ik_tracking.getDependentColumn(state_name).to_numpy()[-1]
            maxVal = ik_tracking.getDependentColumn(state_name).to_numpy().max()
            minVal = ik_tracking.getDependentColumn(state_name).to_numpy().min()

            # Check for pelvis_tx first to make it at least start at the same point
            # TODO: could consider speed goal instead of tx?
            # TODO: pelvis constraints were initially much tighter, but expanded out for 2D problem as a potential solution
            # TODO: is this necessary for all states?
            if 'pelvis_tx' in state_name:
                # Constrain initial value, leave final value open
                problem.setStateInfo(state_name,
                                     [minVal - (valRange * 0.2), maxVal + 0.1],  # maxVal + (valRange * 0.2)],
                                     initialVal,
                                     [finalVal - (valRange * 0.1), finalVal + 0.1],  # finalVal + (valRange * 0.1)]
                                     )
            # Check for other pelvis translational coordinates and set tighter bounds in problem
            elif any(coord in state_name for coord in ['pelvis_ty', 'pelvis_tz']):
                problem.setStateInfo(state_name,
                                     [minVal - (valRange * 0.2), maxVal + 0.1],  # maxVal + (valRange * 0.2)],
                                     [initialVal - (valRange * 0.1), initialVal + 0.1],  # initialVal + (valRange * 0.1)],
                                     [finalVal - (valRange * 0.2), finalVal + 0.1],  # finalVal + (valRange * 0.1)]
                                     )
            # Otherwise set the slightly looser bounds
            # elif not any(coord in state_name for coord in ['mtp_angle_r', 'mtp_angle_l']):
            else:
                problem.setStateInfo(state_name,
                                     [minVal - (valRange * 0.2), maxVal + (valRange * 0.2)],
                                     [initialVal - (valRange * 0.2), initialVal + (valRange * 0.2)],
                                     [finalVal - (valRange * 0.2), finalVal + (valRange * 0.2)])

    # Define and configure the solver
    # -------------------------------------------------------------------------
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

    # Set solver options
    solver.set_optim_max_iterations(1000)
    solver.set_num_mesh_intervals(meshInterval)
    solver.set_optim_constraint_tolerance(1e-3)
    solver.set_optim_convergence_tolerance(1e-3)
    # NOTE: I don't think this helps...maybe weight needs to be quite small...?
    # solver.set_multibody_dynamics_mode('implicit')
    # solver.set_minimize_implicit_multibody_accelerations(True)  # smoothness criterion
    # solver.set_implicit_multibody_accelerations_weight(1e-2)
    # solver.set_parameters_require_initsystem(True)  # TODO: what does this need to be for contact spheres? Seems like True is necessary...

    # Reset problem to avoid any issues
    solver.resetProblem(problem)

    # # Switching to implicit mode causes a problem with the guess not having accelerations
    # # These therefore need to be created in the guess
    # guess = solver.getGuess()
    # guess.generateAccelerationsFromSpeeds()
    # solver.setGuess(guess)

    # TODO: only if necessary for contact tracking...?

    # With parameters there are some difficulties with the guess not including the parameters, so some tweaking is required

    # Get the initial created guess from the tracked states
    initial_guess = solver.getGuess()

    # Push up pelvis_ty in starting guess to avoid contact spheres starting too far in ground
    # Set the pelvis_ty starting value to a small increment above where it is in coordinates data
    # TODO: does this actually cause a poor starting point for model? Definitely for 3D model...
    initial_guess.setState('/jointset/ground_pelvis/pelvis_ty/value',
                           initial_guess.getState('/jointset/ground_pelvis/pelvis_ty/value').to_numpy() + 0.05)

    # # Create a guess that will now include the parameters to be optimised
    # created_guess = solver.createGuess()
    #
    # # Ensure created guess matches sampling of initial guess
    # created_guess.resampleWithNumTimes(initial_guess.getNumTimes())
    #
    # # Fill the created guess with the data from the initial states tracked guess
    # for state_name in initial_guess.getStateNames():
    #     created_guess.setState(state_name, initial_guess.getState(state_name).to_numpy())
    # for control_name in initial_guess.getControlNames():
    #     created_guess.setControl(control_name, initial_guess.getControl(control_name).to_numpy())

    # Set the guess in solver
    # solver.setGuess(created_guess)
    solver.setGuess(initial_guess)
    # solver.setGuessFile(f'{participant_id}_{trial_name}_reference-solution_{model_variant}_meshInterval-5.sto')

    # # Create guess to manipulate
    # guess = solver.createGuess()
    #
    # # Load in inverse solution
    # inverse_solution = osim.MocoTrajectory(f'{participant_id}_{trial_name}_inverse-solution_{model_variant}.sto')
    #
    # # Resample guess to inverse solution time-steps
    # guess.resampleWithNumTimes(inverse_solution.getNumTimes())
    #
    # # Fill states from inverse solution
    # for state_name in guess.getStateNames():
    #     guess.setState(state_name, inverse_solution.getState(state_name).to_numpy())
    #
    # # Fill controls from inverse solution
    # for control_name in guess.getControlNames():
    #     guess.setControl(control_name, inverse_solution.getControl(control_name).to_numpy())
    #
    # # Set guess in solver
    # solver.setGuess(guess)

    # Reset problem to finalise
    solver.resetProblem(problem)

    # Solve the problem
    # -------------------------------------------------------------------------
    solution = study.solve()

    '''
    # testing out original contact tracking approach with coarse mesh and Miller 3D model
        # Seems to suffer from similar issue...
    # Testing out non-scaled weightings to hopefully emphasise tracking --- using weights from Denton code
        # still same issues...
    # test scaled states weight with external loads...
        # note that contact spheres weren't removed so this could be a little off
        # still works fine, which suggests that the problem is with adding the contact tracking goal - is it correctly added?
    # test out plane projection for contact tracking instead?
         # still suffering from same problem
    # test constraining pelvis_ty just to see if it works...
        # does not work either...
    # test out 2d model?
        # same issue jumping out of the ground...
    # setting pelvis_ty at zero with contact tracking
        # same issue
    # only tracking left foot to avoid initial right contact
        # still get same issue
    # removing contact tracking but keeping pelvis_ty weight
        # same issue
    # removing contact tracking and setting pelvis_ty weifght to 0
        # same issue
    # ramping up residual forces and reducing weight on these?
        # slightly same issue but seemed to fix it OK
    # perhaps need to eliminate residuals in motion initially and then contact tracking
        # parameter for initial non-contact tracking simulation
    # ran a standard simulation then added contact tracking
        # still seems to suffer from same issues - what about with high residuals on this?
            # still can't seem to track very well but doesn't bounce as much
    # contact sphere parameters or locations seem to be the main issue
    # able to get a good-ish solution with muscle-driven formulation using DentonUmberger code
        # torque-driven simulation wth contact tracking the problem?
    # testing contact tracking with AddBiomechanics ik data?
        # not helping...
    # does using torque tracking inverse solution as initial guess help?
        # no...
    # track controls from inverse solution?
        # nope
    # hugely higher state tracking while tracking pelvis_ty and controls
        # nope
    # hugely higher weight on contact tracking, slightly higher on states tracking, with control tracking?
        # nope
    # high contact tracking weight, low states tracking weight, no control tracking...
        # nope
    # is it bounds that are needed to help?
        # testing with normalised state and contact weighting with bounds (like it works in Earth/Mars tracking data)
        # seemed to help a fair bit - normalised GRF not ideal though...
            # standard contact tracking weights seem to do quite well --- states tracking is the least accurate and issue with convergence
            # BUT - overall it's actually working better
            # Test with simpler 2D problem, better contact sphere array, non-normalised weights with zero pelvis_ty & tx tracking?
                # seems to struggle a little --- switch back to normalised GRF and tracking all coordinates? No residuals included either
                    # perhaps add residuals back in? Normalised state and GRF tracking too
                        # Not good...
            # 2D normalised state and contact tracking goals + bounds?
                # for some reason this just tracks the states poorly?
            # 3D problem with same formulation?
                # there is still some bouncing, but on the whole it works a lot better than the 2D problem --- why?
                # seems to eliminate bouncing a little by reducing global state weight back to 1.0 or maybe removing the elevating initial guess?
                    # better, but something about it still seems a little off...pelvis_tx tracks a little off...as well as the toe off...
                    # GRFs aren't exactly that smooth either...contact sphere locations?
                        # what about using the specified state weights?
                            # still works pretty much the same
                            # turn off pelvis_ty and mtp?
                                # pelvis_ty not bad - mtp gets a little funky
                            # just pelvis_ty off?
                                # doesn't really make a huge difference
                    # dramatically increasing weights on certain coordinates? increase knee and ankle + arms, reduce pelvis rotations
                        # not as good
                    # non-normalised GRF? 0.5 weightings for contact tracking? kept increase knee, ankle and arm weights too
                        # does OK but doesn't converge
                    # normalised GRF and lower state weight (0.1)
                        # seems similar, a little bit better - possible best solution so far
                        # try this but with a 50 mesh interval...also constrained pelvis_tx to single initial value
                            # despite syncing pelvis_tx to initial value it still didn't seem to match?
                            # a little bit of foot slipping - contact spheres not optimised placement
                            # not as good of a solution then for GRFs
                        # does 2D work with this? (inc. no pushing up of initial guess)
                            # really struggles to track states - why is this different to 3D problem? Contact spheres?
                            # don't track pelvis_ty and take out residuals? widen out pelvis bounds too (I think pelvis bounds were main problem...)
                                # works much better, maybe residuals are needed though to smooth out/control pelvis oscillations? + track pelvis_ty?
                            # does non-normalised plane contact tracking approach work better (0.5 weight on contact tracking goal)
                                # solves - but high contact tracking value and noisy results
                            # perhaps reduce weight on pelvis ty, increase residual forces, 0.5 non-normalised contact tracking... 
                                # same issue, contact tracking not ideal - maybe MTP weights?
                            # contact tracking GRFs don't look great - perhaps contact sphere location, parameters?
                                # optimise these parameters beforehand with kinematics to get a good starting point?
                                    # perhaps combining approaches that work with parameter optimisation is ideal?
                                # or can they be optimised in an initial torque driven simulation?
                            # contact spheres weren't being added apppropriately to problem... testing with them added...
                                # much better, perhaps still not ideal - test with 0.5 plane weighting
                                    # doesn't converge and does't work as well
                            # seems like optimising parameters might be useful? stiffness and vertical location?
                                # testing out optimising stiffness parameter alongside tracking problem in 2D model
                                    # like in previous tries with this the parameters don't change -- test with inverse problem first?
                            # muscle-driven 2D doesn't really work that well...
                            # average speed goal seems to mess with convergence alot - maybe due to bounds or something?
                            # mtp setting at zero should help, but it doesn't seem to converge as well...
                            # is an absoute contact tracking weight better to place more emphasis on this?
                                #  doesn't work as well
                            # MTP joint seems a little painful - maybe lock at zero for IK and then leave it untracked? Or track low weight at zero?
                            # Are residuals needed back in?
                                # might be...higher residuals maybe needed to control model?
                                    # doesn't seem to help...
                
    
    '''

    # # Option to visualise solution
    # study.visualize(solution)

    # Save files and finalize
    # -------------------------------------------------------------------------

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
    os.remove(f'{participant_id}_{trial_name}_reference_{model_variant}_tracked_states.sto')

    # Return to home directory
    os.chdir(home_dir)





# =========================================================================
# Start code here ...
# =========================================================================

if __name__ == '__main__':

    # Run reference tracking simulation for 3D data
    # -------------------------------------------------------------------------

    # TODO: build in function
    print('run function')

# %% ---------- end of generate_reference.py ---------- %% #
