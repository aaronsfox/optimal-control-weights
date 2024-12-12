# -*- coding: utf-8 -*-
"""

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This code is a secondary step in processing the relevant data needed from the
    SchreiberMoissenet2019 dataset. After extracting the desired data, this code
    generates tracking simulations of the coordinate data from IK to produce a
    coordinates dynamically consistent with the ground reactions.

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

# =========================================================================
# Set-up
# =========================================================================

# Set the participant list based on the codes in the raw folder
# See README.MD in dataset folder for details on setting up raw data
participantList = [ii for ii in os.listdir(os.path.join('..', 'data')) if os.path.isdir(os.path.join('..', 'data', ii))]

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', 'model', 'Geometry'))

# Set weights for optimisations
globalCoordinateTrackingWeight = 1e0
globalControlEffortGoal = 1e-3

# Set mesh interval
# TODO: appropriate?
meshInterval = 50

# Set kinematics filter frequency
kinematicFiltFreq = 12

# Set actuator forces to drive simulations
actForces = {'pelvis_tx': {'actuatorType': 'residual', 'optForce': 2.5},
             'pelvis_ty': {'actuatorType': 'residual', 'optForce': 2.5},
             'pelvis_tz': {'actuatorType': 'residual', 'optForce': 2.5},
             'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 1.0},
             'pelvis_list': {'actuatorType': 'residual', 'optForce': 1.0},
             'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 1.0},
             'hip_flexion_r': {'actuatorType': 'torque', 'optForce': 300.0},
             'hip_adduction_r': {'actuatorType': 'torque', 'optForce': 200.0},
             'hip_rotation_r': {'actuatorType': 'torque', 'optForce': 100.0},
             'knee_angle_r': {'actuatorType': 'torque', 'optForce': 300.0},
             'ankle_angle_r': {'actuatorType': 'torque', 'optForce': 200.0},
             'hip_flexion_l': {'actuatorType': 'torque', 'optForce': 300.0},
             'hip_adduction_l': {'actuatorType': 'torque', 'optForce': 200.0},
             'hip_rotation_l': {'actuatorType': 'torque', 'optForce': 100.0},
             'knee_angle_l': {'actuatorType': 'torque', 'optForce': 300.0},
             'ankle_angle_l': {'actuatorType': 'torque', 'optForce': 200.0},
             'lumbar_extension': {'actuatorType': 'torque', 'optForce': 300.0},
             'lumbar_bending': {'actuatorType': 'torque', 'optForce': 200.0},
             'lumbar_rotation': {'actuatorType': 'torque', 'optForce': 100.0},
             'arm_flex_r': {'actuatorType': 'torque', 'optForce': 100.0},
             'arm_add_r': {'actuatorType': 'torque', 'optForce': 100.0},
             'arm_rot_r': {'actuatorType': 'torque', 'optForce': 100.0},
             'elbow_flex_r': {'actuatorType': 'torque', 'optForce': 50.0},
             'pro_sup_r': {'actuatorType': 'torque', 'optForce': 25.0},
             'arm_flex_l': {'actuatorType': 'torque', 'optForce': 100.0},
             'arm_add_l': {'actuatorType': 'torque', 'optForce': 100.0},
             'arm_rot_l': {'actuatorType': 'torque', 'optForce': 100.0},
             'elbow_flex_l': {'actuatorType': 'torque', 'optForce': 50.0},
             'pro_sup_l': {'actuatorType': 'torque', 'optForce': 25.0},
             }

# Set markers to look up for ground contact constraints
contactMarkers = [
    'R_FM1_ground', 'R_FM2_ground', 'R_FM5_ground', 'R_FCC_ground',
    # 'R_FM1_mid_ground', 'R_FM2_mid_ground', 'R_FM5_mid_ground',
    'L_FM1_ground', 'L_FM2_ground', 'L_FM5_ground', 'L_FCC_ground'
    # 'L_FM1_mid_ground', 'L_FM2_mid_ground', 'L_FM5_mid_ground'
]


# =========================================================================
# Loop through participants to run simulations
# =========================================================================

# Participant loop
for participant in participantList:

    # Identify the trials for the participant from the IK folder
    trialList = [ii for ii in os.listdir(
        os.path.join('..', 'data', participant, 'ik')) if os.path.isdir(os.path.join('..', 'data', participant, 'ik', ii))]

    # Create over-arching folder for data
    os.makedirs(os.path.join('..', 'data', participant, 'baseline'), exist_ok=True)

    # Loop through trials for simulations
    # -------------------------------------------------------------------------
    for trial in trialList:

        # Set-up folders and files for trial
        # -------------------------------------------------------------------------

        # Create folder for storing trial results
        os.makedirs(os.path.join('..', 'data', participant, 'baseline', trial), exist_ok=True)

        # Navigate to simulation folder for ease of use
        homeDir = os.getcwd()
        os.chdir(os.path.join('..', 'data', participant, 'baseline', trial))

        # Copy external loads file to simulation directory
        shutil.copyfile(os.path.join('..', '..', 'dynamic', f'{participant}_{trial}_grf.mot'),
                        f'{participant}_{trial}_grf.mot')
        shutil.copyfile(os.path.join('..', '..', 'dynamic', f'{participant}_{trial}_grf.xml'),
                        f'{participant}_{trial}_grf.xml')

        # Copy IK file to simulation directory
        shutil.copyfile(os.path.join('..', '..', 'ik', trial, f'{trial}_ik.mot'),
                        f'{trial}_ik.mot')

        # Adjust kinematic data for tracking simulation
        # -------------------------------------------------------------------------

        # Load in the kinematic data
        kinematicsStorage = osim.Storage(f'{trial}_ik.mot')

        # Create a copy of the kinematics data to alter the column labels in
        statesStorage = osim.Storage(f'{trial}_ik.mot')

        # Filter both storage objects
        # Note that this resamples time stamps so eliminates the need to do so
        kinematicsStorage.lowpassFIR(4, kinematicFiltFreq)
        statesStorage.lowpassFIR(4, kinematicFiltFreq)

        # Get the column headers for the storage file
        angleNames = kinematicsStorage.getColumnLabels()

        # Get the corresponding full paths from the model to rename the
        # angles in the kinematics file
        kinematicModel = osim.Model(os.path.join('..', '..', 'scaling', f'{participant}_scaledModelAdjusted.osim'))
        for angNo in range(angleNames.getSize()):
            currAngle = angleNames.get(angNo)
            if currAngle != 'time':
                # Try getting the full path to coordinate
                # This may fail due to their being marker data included in these files
                try:
                    # Loo for full coordinate path
                    fullPath = kinematicModel.updCoordinateSet().get(currAngle).getAbsolutePathString() + '/value'
                    # Set angle name appropriately using full path
                    angleNames.set(angNo, fullPath)
                except:
                    # Print out that current column isn't a coordinate
                    print(f'{currAngle} not a coordinate...skipping name conversion...')
                    # Set to the same as originaly
                    angleNames.set(angNo, currAngle)

        # Set the states storage object to have the updated column labels
        statesStorage.setColumnLabels(angleNames)

        # Convert from IK default of degrees to radians
        kinematicModel.initSystem()
        kinematicModel.getSimbodyEngine().convertDegreesToRadians(statesStorage)

        # Write the states storage object to file
        statesStorage.printToXML(f'{trial}_coordinates.sto')

        # Set up model for tracking simulation
        # -------------------------------------------------------------------------

        # Load the model
        osimModel = osim.Model(os.path.join('..', '..', 'scaling', f'{participant}_scaledModelAdjusted.osim'))

        # There are already some upper body torque actuators in the model
        # Clear these to add our own with consistent naming
        osimModel.updForceSet().clearAndDestroy()
        osimModel.finalizeConnections()

        # Construct a model processor to further edit the model
        modelProc = osim.ModelProcessor(osimModel)

        # Append external loads
        modelProc.append(osim.ModOpAddExternalLoads(f'{participant}_{trial}_grf.xml'))

        # Weld desired locked joints
        # Create vector string object
        weldVectorStr = osim.StdVectorString()
        [weldVectorStr.append(joint) for joint in ['subtalar_r', 'subtalar_l', 'mtp_r', 'mtp_l',
                                                   'radius_hand_r', 'radius_hand_l']]
        # Append to model processor
        modelProc.append(osim.ModOpReplaceJointsWithWelds(weldVectorStr))

        # Remove muscles from model
        # Probably unnecessary given force set destroyed
        modelProc.append(osim.ModOpRemoveMuscles())

        # Process model for further edits
        trackingModel = modelProc.process()

        # Add coordinate actuators to model
        for coordinate in actForces:
            # Create actuator
            actu = osim.CoordinateActuator()
            # Set name
            actu.setName(f'{coordinate}_{actForces[coordinate]["actuatorType"]}')
            # Set coordinate
            actu.setCoordinate(trackingModel.updCoordinateSet().get(coordinate))
            # Set optimal force
            actu.setOptimalForce(actForces[coordinate]['optForce'])
            # Set min and max control
            actu.setMinControl(np.inf * -1)
            actu.setMaxControl(np.inf * 1)
            # Append to model force set
            trackingModel.updForceSet().cloneAndAppend(actu)

        # Finalise model connections
        trackingModel.finalizeConnections()

        # Print model to file in tracking directory
        trackingModel.printToXML(f'{participant}_{trial}_baselineModel.osim')

        # Set up tracking simulation
        # -------------------------------------------------------------------------

        # Create tracking tool
        track = osim.MocoTrack()
        track.setName(f'{participant}_{trial}_baseline')

        # Set model
        trackModelProc = osim.ModelProcessor(f'{participant}_{trial}_baselineModel.osim')
        track.setModel(trackModelProc)

        # Set the coordinates reference file
        tableProcessor = osim.TableProcessor(f'{trial}_coordinates.sto')
        track.setStatesReference(tableProcessor)

        # Set to ignore unused columns
        track.set_allow_unused_references(True)

        # Set global tracking weight
        track.set_states_global_tracking_weight(globalCoordinateTrackingWeight)

        # Track positive derivaties (i.e. speeds)
        track.set_track_reference_position_derivatives(True)

        # Set tracked states to guess
        track.set_apply_tracked_states_to_guess(True)

        # Set the timings
        # Slightly different due to potential re-sampling of time-stamps
        track.set_initial_time(osim.TimeSeriesTable(f'{trial}_coordinates.sto').getIndependentColumn()[0])
        track.set_final_time(osim.TimeSeriesTable(f'{trial}_coordinates.sto').getIndependentColumn()[-1])

        # Initialise to a Moco study and problem to finalise
        # -------------------------------------------------------------------------

        # Get study and problem
        study = track.initialize()
        problem = study.updProblem()

        # Update control effort goal
        # -------------------------------------------------------------------------

        # Get a reference to the MocoControlCost goal and set parameters
        effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'))
        effort.setWeight(globalControlEffortGoal)
        effort.setExponent(2)

        # Update individual weights in control effort goal
        # Put higher weight on residual use
        effort.setWeightForControlPattern('/forceset/.*_residual', 5.0)
        # Put heavy weight on the reserve actuators
        effort.setWeightForControlPattern('/forceset/.*_torque', 1.0)

        # Update states tracking goal
        # -------------------------------------------------------------------------

        # Get a reference to the states tracking goal
        tracking = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'))

        # Scale individual tracking weights with data range
        tracking.setScaleWeightsWithRange(True)

        # Add constraints to the problem
        # -------------------------------------------------------------------------

        # Constrain contact markers to at minimum be at ground level

        # Loop through markers to create path constraints
        for marker in contactMarkers:
            # Create constraint
            markerConstraint = osim.MocoOutputConstraint()
            markerConstraint.setName(f'{marker}_constraint')
            # Set path to marker location
            markerConstraint.setOutputPath(f'/markerset/{marker}|location')
            # Set output index to y-axis
            markerConstraint.setOutputIndex(1)
            # Create and set the bounds to slightly below ground level and a reasonable height
            markerBounds = osim.StdVectorMocoBounds()
            markerBounds.append(osim.MocoBounds(-0.05, 0.25))
            markerConstraint.updConstraintInfo().setBounds(markerBounds)
            # Add to problem
            problem.addPathConstraint(markerConstraint)

        # Define and configure the solver
        # -------------------------------------------------------------------------
        solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

        # Set solver options
        solver.set_optim_max_iterations(1000)
        solver.set_num_mesh_intervals(meshInterval)
        solver.set_optim_constraint_tolerance(1e-2)
        solver.set_optim_convergence_tolerance(1e-3)
        # NOTE: I don't think this helps...maybe weight needs to be quite small...?
        # solver.set_multibody_dynamics_mode('implicit')
        # solver.set_minimize_implicit_multibody_accelerations(True)  # smoothness criterion
        # solver.set_implicit_multibody_accelerations_weight(1e-2)

        # Reset problem to avoid any issues
        solver.resetProblem(problem)

        # # Switching to implicit mode causes a problem with the guess not having accelerations
        # # These therefore need to be created in the guess
        # guess = solver.getGuess()
        # guess.generateAccelerationsFromSpeeds()
        # solver.setGuess(guess)

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
        solution.write(f'{participant}_{trial}_baselineSolution.sto')

        # Remove initial tracked states and markers file
        os.remove(f'{participant}_{trial}_baseline_tracked_states.sto')

        # Return to home directory
        os.chdir(homeDir)

    # Print out confirmation for participant
    # -------------------------------------------------------------------------
    print(f'Baseline simulations completed for {participant}...')

# =========================================================================
# Finalise and exit
# =========================================================================

# Exit console to avoid exit code error
os._exit(00)

# %% ----- end of genBaselineSims.py ----- %% #