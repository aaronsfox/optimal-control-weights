# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    NOTES:
        > Does this work better for the new dataset? Not really
        > Adjusting sphere penetration to zero in adjustment seems to help
            >> Seems to work with torque driven - even though contact tracking val high (~60) it looks ok...
            >> Muscle driven? Absolutely cooked - something probably wrong with muscles
        > Weightings balance perhaps needs to be better (e.g. higher for states, lower for contact)
            > Increase states and adjust individual contact tracking weights
        > Torque driven sim is starting to look better overall
            >> Add in tracking of joint moments to perhaps gte better COP/GRF tracking?
        > Revisit Modenese muscle optim pipeline to see if this gets muscles in better place?
            >> Seems to put muscles in a better place to achieve motion --- still can't seem to solve quite yet though
            >> Are there other problems like contraction velocity?
        > Having symmetry constraints on activations etc. actually makes things look quite good
            >> Add mtp back in? And maybe some tendon compliance?
            >> MTP seems to cause a little trouble
                >> Particularly when tracking this even with a low weight it seems to be an issue
                >> Could be because musles weren't calibrated with this joint unlocked?
                >> Similarly does disabling tendon dynamics also mess with optimised muscle parameters
                >> Are reserve actuators causing that issue?
                >> Nope, mostly just seems like mtp angle as an issue for whatever reason
        > Taking away reserves with MTP still locked
            >> Struggles...
        > Use Denton & Umberger modified initial guess without bounds?
            >> Probably still need bounds
            >> Sort of OK, but might be better to test with the more complex muscles (i.e. tendon, passive fores) re-enabled
            >> Enabling tendons seems to cause issues --- maybe because muscles have been optimised without them?
            >> Passive forces seems to make problem infeasible as well...
        > Should back joint be locked for 2D, or is low contro effort weight able to account for this?
            >> Low control weight seems to help with objective function reduction in early iterations

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
participant = 'SUBJ01'

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
globalControlEffortWeight = 1.0e-3  # lower than Denton & Umberger
globalStateTrackingWeight = 1.0e0  # higher than Denton & Umberger
globalContactTrackingWeight = [1.0e-1, 1.0e-1, 1.0e-1]  # Slightly different approach to Denton * Umberger
globalAuxDerivWeight = 1.0e-3

# Set intervals for mesh refinement
meshIntervals = [5, 12, 25]

# Set actuator forces to append to upper body coordinates
# TODO: are these needed in 3D model given back muscles included?
actForces = {
    'lumbar_ext': {'actuatorType': 'torque', 'optForce': 100.0},
    'lumbar_bend': {'actuatorType': 'torque', 'optForce': 100.0},
    'lumbar_rota': {'actuatorType': 'torque', 'optForce': 100.0},
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
    # 'mtp_angle_r': [np.deg2rad(-25),np.deg2rad(75)],
    'hip_flexion_l': [np.deg2rad(-20),np.deg2rad(60)],
    'knee_angle_l': [np.deg2rad(-90),np.deg2rad(0)],
    'ankle_angle_l': [np.deg2rad(-35),np.deg2rad(20)],
    # 'mtp_angle_l': [np.deg2rad(-25),np.deg2rad(75)],
    'lumbar_ext': [np.deg2rad(-40),np.deg2rad(5)],
    }
}

# TODO: any other settings?

# =========================================================================
# Define functions
# =========================================================================

# Function to run inverse simulation
def run_inverse_sim(participant_id, model_variant='3d'):

    # Organise and set-up files
    # -------------------------------------------------------------------------

    # Create over-arching folder for data
    os.makedirs(os.path.join('..', 'data', participant_id, 'inverse'), exist_ok=True)
    os.makedirs(os.path.join('..', 'data', participant_id, 'inverse', model_variant), exist_ok=True)

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..', 'data', participant_id, 'inverse', model_variant))

    # Copy external loads file to simulation directory
    # Note these come from the experimental tracking folder as they have been edited for 2d model
    shutil.copyfile(os.path.join('..', '..', 'exp', model_variant, 'walk_grf.mot'), 'walk_grf.mot')
    shutil.copyfile(os.path.join('..', '..', 'exp', model_variant, 'walk_grf.xml'), 'walk_grf.xml')

    # Copy experimental tracking kinematics to simulation directory
    # Note that this uses the version that has NOT been adjusted based on contact sphere penetration
    shutil.copyfile(os.path.join('..', '..', 'exp', model_variant, f'{participant_id}_exp-tracking-solution_{model_variant}.sto'),
                    f'{participant_id}_exp-tracking-solution_{model_variant}.sto')

    # Copy model file to simulation directory
    # Note that this uses the model following parameter optimisation
    shutil.copyfile(os.path.join('..', '..', 'model', f'{participant_id}_model_{model_variant}_optim.osim'),
                    f'{participant_id}_model_{model_variant}_optim.osim')

    # Set-up model for simulation
    # -------------------------------------------------------------------------

    # TODO: is there any adjustment needed to model as in Moco examples? Active force width etc.?
    # TODO: consider reserve actuators?

    # Simplify model to begin with
    model_proc = osim.ModelProcessor(f'{participant_id}_model_{model_variant}_optim.osim')
    # model_proc.append(osim.ModOpIgnoreTendonCompliance())  # Seems to cause significant problems with feasibility when left on
    model_proc.append(osim.ModOpIgnorePassiveFiberForcesDGF())  # Seems to cause significant problems with feasibility when left on?
    model_proc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

    # Get the model
    osim_model = model_proc.process()

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

    # # Unlock mtp and subtalar joints for tracking simulations
    # osim_model.getJointSet().get('mtp_r').get_coordinates(0).set_locked(False)
    # osim_model.getJointSet().get('mtp_l').get_coordinates(0).set_locked(False)
    # if model_variant == '3d':
    #     osim_model.getJointSet().get('subtalar_r').get_coordinates(0).set_locked(False)
    #     osim_model.getJointSet().get('subtalar_l').get_coordinates(0).set_locked(False)

    # Finalise model connections
    osim_model.finalizeConnections()

    # Print model to file
    osim_model.printToXML(f'{participant_id}_inverse-model_{model_variant}.osim')

    # Get kinematic states data for inverse simulation
    # -------------------------------------------------------------------------

    # Load in the solution file and export to states table
    states_data = osim.MocoTrajectory(f'{participant_id}_exp-tracking-solution_{model_variant}.sto').exportToStatesTable()

    # Write the states object to file
    osim.STOFileAdapter().write(states_data, f'{participant_id}_states_{model_variant}.sto')

    # Set up inverse simulation
    # -------------------------------------------------------------------------

    # Create tracking tool
    inverse = osim.MocoInverse()
    inverse.setName(f'{participant_id}_inverse_{model_variant}')

    # Set model in tool
    # TODO: testing if this is even possible with torque driven --- it is, now try with optimised muscle model...maybe, needs some fine tuning...
    inverse_model_proc = osim.ModelProcessor(f'{participant_id}_inverse-model_{model_variant}.osim')

    # Weld locked joints for marker tracking
    weld_joints = osim.StdVectorString()
    # MTP joints in both models
    weld_joints.append('mtp_r')
    weld_joints.append('mtp_l')
    # Subtalar only in 3D model
    if model_variant == '3d':
        weld_joints.append('subtalar_r')
        weld_joints.append('subtalar_l')
    # Lumbar joint in 2D model
    if model_variant == '2d':
        weld_joints.append('back')
    inverse_model_proc.append(osim.ModOpReplaceJointsWithWelds(weld_joints))

    # Add some reserve torques to support
    inverse_model_proc.append(osim.ModOpAddReserves(2.5))

    # Append external loads
    inverse_model_proc.append(osim.ModOpAddExternalLoads('walk_grf.xml'))

    # Set model in tool
    inverse.setModel(inverse_model_proc)

    # Set the coordinates reference file
    table_proc = osim.TableProcessor(f'{participant_id}_states_{model_variant}.sto')
    inverse.setKinematics(table_proc)

    # By default, Moco gives an error if the kinematics contains extra columns.
    # Here, we tell Moco to allow (and ignore) those extra columns.
    inverse.set_kinematics_allow_extra_columns(True)

    # Set the timings
    # To be safe just use start and end of tracking solution
    inverse.set_initial_time(osim.TimeSeriesTable(f'{participant_id}_states_{model_variant}.sto').getIndependentColumn()[0])
    inverse.set_final_time(osim.TimeSeriesTable(f'{participant_id}_states_{model_variant}.sto').getIndependentColumn()[-1])

    # Initialise to a Moco study and problem to finalise
    # -------------------------------------------------------------------------

    # Get study and problem
    study = inverse.initialize()
    problem = study.updProblem()

    # Update control effort goal
    # -------------------------------------------------------------------------

    # Get a reference to the MocoControlCost goal and set parameters
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('excitation_effort'))
    effort.setWeight(globalControlEffortWeight)
    effort.setExponent(2)

    # Define and configure the solver
    # -------------------------------------------------------------------------
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

    # Set solver options
    solver.set_optim_max_iterations(1000)
    solver.set_num_mesh_intervals(25)  # TODO: set appropriately
    solver.set_optim_constraint_tolerance(1e-3)
    solver.set_optim_convergence_tolerance(1e-3)

    # Reset problem to avoid any issues
    solver.resetProblem(problem)

    # Solve the problem
    # -------------------------------------------------------------------------
    solution = study.solve()

    # Save files and finalize
    # -------------------------------------------------------------------------

    # Write solution to file
    # TODO: different names?
    if solution.isSealed():
        solution.unseal()
    solution.write(f'{participant_id}_inverse-solution_{model_variant}.sto')

    # Return to home directory
    os.chdir(home_dir)





# Function to run initial tracking simulation
# -------------------------------------------------------------------------
def run_initial_sim(participant_id, model_variant='3d'):

    # Organise and set-up files
    # -------------------------------------------------------------------------

    # Create over-arching folder for data
    os.makedirs(os.path.join('..', 'data', participant_id, 'initial'), exist_ok=True)
    os.makedirs(os.path.join('..', 'data', participant_id, 'initial', model_variant), exist_ok=True)

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..', 'data', participant_id, 'initial', model_variant))

    # Copy external loads file to simulation directory
    # Note these come from the experimental tracking folder as they have been edited for 2d model
    shutil.copyfile(os.path.join('..', '..', 'exp', model_variant, 'walk_grf.mot'),'walk_grf.mot')
    shutil.copyfile(os.path.join('..', '..', 'exp', model_variant, 'walk_grf.xml'), 'walk_grf.xml')

    # Copy experimental tracking kinematics to simulation directory
    # Note that this uses the version that has been adjusted based on contact sphere penetration
    shutil.copyfile(os.path.join('..', '..', 'exp', model_variant, f'{participant_id}_exp-tracking-solution-adjusted_{model_variant}.sto'),
                    f'{participant_id}_exp-tracking-solution-adjusted_{model_variant}.sto')

    # Copy model file to simulation directory
    # Note that this uses the model following parameter optimisation
    shutil.copyfile(os.path.join('..', '..', 'model', f'{participant_id}_model_{model_variant}_optim.osim'),
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

    # Get the model
    osim_model = model_proc.process()

    # Remove passive subtalar and/or toe damping forces given joint will be welded
    # TODO: depending on what joint remains
    remove_force = []
    if model_variant == '3d':
        for forceInd in range(osim_model.getForceSet().getSize()):
            if osim_model.updForceSet().get(forceInd).getName().startswith('PassiveToeMoment') or \
                    osim_model.updForceSet().get(forceInd).getName().startswith('PassiveSubtalarDamping'):
                remove_force.append(forceInd)
    elif model_variant == '2d':
        for forceInd in range(osim_model.getForceSet().getSize()):
            if osim_model.updForceSet().get(forceInd).getName().startswith('mtp_damping'):
                remove_force.append(forceInd)
    counter = 0
    for remove in remove_force:
        osim_model.updForceSet().remove(remove - counter)
        counter += 1

    # # Unlock mtp and subtalar joints for tracking simulations
    # osim_model.getJointSet().get('mtp_r').get_coordinates(0).set_locked(False)
    # osim_model.getJointSet().get('mtp_l').get_coordinates(0).set_locked(False)
    # if model_variant == '3d':
    #     osim_model.getJointSet().get('subtalar_r').get_coordinates(0).set_locked(False)
    #     osim_model.getJointSet().get('subtalar_l').get_coordinates(0).set_locked(False)

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

    # # Set rigid tendons in the model
    # # TODO: use if tendon compliance re-enabled earlier
    # # TODO: this would come across accurately from the model optimisation process
    # for muscle_ind in range(osim_model.getMuscles().getSize()):
    #     if osim_model.getMuscles().get(muscle_ind).getName() not in elasticTendons[model_variant]:
    #         # Set to rigid tendon
    #         osim_model.getMuscles().get(muscle_ind).set_ignore_tendon_compliance(True)

    # Finalise model connections
    osim_model.finalizeConnections()

    # Print model to file
    osim_model.printToXML(f'{participant_id}_initial-model_{model_variant}.osim')

    # Set up tracking simulation
    # -------------------------------------------------------------------------

    # Create tracking tool
    track = osim.MocoTrack()
    track.setName(f'{participant_id}_initial_{model_variant}')

    # Set model in tool
    # TODO: testing if this is even possible with torque driven --- it is, now try with optimised muscle model...maybe, needs some fine tuning...
    track_model_proc = osim.ModelProcessor(f'{participant_id}_initial-model_{model_variant}.osim')
    # track_model_proc.append(osim.ModOpRemoveMuscles())

    # Weld locked joints for marker tracking
    weld_joints = osim.StdVectorString()
    # MTP joints in both models
    weld_joints.append('mtp_r')
    weld_joints.append('mtp_l')
    # Subtalar only in 3D model
    if model_variant == '3d':
        weld_joints.append('subtalar_r')
        weld_joints.append('subtalar_l')
    # # Lumbar joint in 2D model
    # if model_variant == '2d':
    #     weld_joints.append('back')
    track_model_proc.append(osim.ModOpReplaceJointsWithWelds(weld_joints))

    # # Add some reserve torques to support
    # track_model_proc.append(osim.ModOpAddReserves(1.0))  # TODO: keeping in at the moment to see if it helps

    # Set model in tool
    track.setModel(track_model_proc)

    # Set the timings
    # To be safe just use start and end of tracking solution
    track.set_initial_time(osim.TimeSeriesTable(f'{participant_id}_exp-tracking-solution-adjusted_{model_variant}.sto').getIndependentColumn()[0])
    track.set_final_time(osim.TimeSeriesTable(f'{participant_id}_exp-tracking-solution-adjusted_{model_variant}.sto').getIndependentColumn()[-1])

    # Set states tracking parameters
    # -------------------------------------------------------------------------

    # Load in the solution file and export to states table
    states_data = osim.MocoTrajectory(f'{participant_id}_exp-tracking-solution-adjusted_{model_variant}.sto').exportToStatesTable()

    # # Add in zero data for the originally locked coordinates as these won't be in original solution
    # # Create the array to add in
    # zero_data = np.zeros(states_data.getNumRows())
    # # Add in the data for coordinate values and speeds
    # states_data.appendColumn(osim_model.getCoordinateSet().get('mtp_angle_r').getAbsolutePathString() + '/value',
    #                          osim.Vector().createFromMat(zero_data))
    # states_data.appendColumn(osim_model.getCoordinateSet().get('mtp_angle_r').getAbsolutePathString() + '/speed',
    #                          osim.Vector().createFromMat(zero_data))
    # states_data.appendColumn(osim_model.getCoordinateSet().get('mtp_angle_l').getAbsolutePathString() + '/value',
    #                          osim.Vector().createFromMat(zero_data))
    # states_data.appendColumn(osim_model.getCoordinateSet().get('mtp_angle_l').getAbsolutePathString() + '/speed',
    #                          osim.Vector().createFromMat(zero_data))
    # if model_variant == '3d':
    #     states_data.appendColumn(osim_model.getCoordinateSet().get('subtalar_angle_r').getAbsolutePathString() + '/value',
    #                              osim.Vector().createFromMat(zero_data))
    #     states_data.appendColumn(osim_model.getCoordinateSet().get('subtalar_angle_r').getAbsolutePathString() + '/speed',
    #                              osim.Vector().createFromMat(zero_data))
    #     states_data.appendColumn(osim_model.getCoordinateSet().get('subtalar_angle_l').getAbsolutePathString() + '/value',
    #                              osim.Vector().createFromMat(zero_data))
    #     states_data.appendColumn(osim_model.getCoordinateSet().get('subtalar_angle_l').getAbsolutePathString() + '/speed',
    #                              osim.Vector().createFromMat(zero_data))

    # Write tracking states data to file
    osim.STOFileAdapter().write(states_data, f'{participant_id}_tracking-states_{model_variant}.sto')

    # Set the reference file
    table_proc = osim.TableProcessor(f'{participant_id}_tracking-states_{model_variant}.sto')
    track.setStatesReference(table_proc)

    # Set to ignore unused columns
    track.set_allow_unused_references(True)

    # Set global tracking weight
    track.set_states_global_tracking_weight(globalStateTrackingWeight)

    # Set to track position derivatives
    track.set_track_reference_position_derivatives(True)

    # Set tracked states to guess
    track.set_apply_tracked_states_to_guess(True)

    # Initialise to a Moco study and problem to finalise
    # -------------------------------------------------------------------------

    # Get study and problem
    study = track.initialize()
    problem = study.updProblem()

    # Add initial activation goal
    # -------------------------------------------------------------------------

    # TODO: this seems necessary for good solutions...

    # For all muscles with activation dynamics, the initial activation and initial excitation should be the same.
    # Without this goal, muscle activation may undesirably start at its maximum possible value as only excitation
    # is penalised
    initial_act = osim.MocoInitialActivationGoal('initial_activation')
    problem.addGoal(initial_act)

    # Get and modify control effort goal
    # -------------------------------------------------------------------------

    # Get a reference to the MocoControlCost goal and set parameters
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'))

    # Set parameters
    effort.setWeight(globalControlEffortWeight)
    effort.setExponent(2)

    # Update individual weights in control effort goal
    # Put low weight on torque actuator use so this isn't overly penalised
    effort.setWeightForControlPattern('/forceset/.*_torque', 0.1)

    # Get and modify states tracking goal
    # -------------------------------------------------------------------------

    # Get a reference to the states tracking goal
    tracking = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'))

    # TODO: test this approach instead of generic state weights? This had a higher global state weight of 1.0...
    tracking.setWeightForState('/jointset/ground_pelvis/pelvis_tx/value', 0)
    tracking.setWeightForState('/jointset/ground_pelvis/pelvis_tx/speed', 0)

    # Here we scale individual tracking weights with the data range
    # The states tracking goal has a function to do this, but it doesn't work with the
    # purely zero data added into the states for mtp_angle or subtalar_angle. Therefore
    # the process is replicated here and these zero joints (along with pelvis_tx) have
    # a zero weight set to them.

    # TODO: this may over-weight something like pelvis_ty given small range...maybe consider absolute values too?

    # Get model from processor to use most up to date for setting state goals
    osim_model = track_model_proc.process()
    osim_model.initSystem()

    # # Set individual state weights based on the data range
    # for coord_ind in range(osim_model.updCoordinateSet().getSize()):
    #     # Get the name and absolute path of the coordinate
    #     coord_name = osim_model.updCoordinateSet().get(coord_ind).getName()
    #     coord_path = osim_model.updCoordinateSet().get(coord_ind).getAbsolutePathString()
    #     # Check if coordinate is one we want to track
    #     if coord_name not in ['pelvis_tx', 'mtp_angle_r', 'mtp_angle_l', 'subtalar_angle_r', 'subtalar_angle_l']:
    #         # Identify the range for the value and speed from the tracking data
    #         value_range = np.ptp(states_data.getDependentColumn(f'{coord_path}/value').to_numpy())
    #         speed_range = np.ptp(states_data.getDependentColumn(f'{coord_path}/speed').to_numpy())
    #         # Set weight value (this is the same process used in MocoStateTrackingGoal.cpp)
    #         value_weight = 1.0 / value_range
    #         speed_weight = 1.0 / speed_range
    #         # Set weight for states
    #         tracking.setWeightForState(f'{coord_path}/value', value_weight)
    #         tracking.setWeightForState(f'{coord_path}/speed', speed_weight)
    #         # tracking.setWeightForState(f'{coord_path}/value', 1.0)  # TODO: standard or normalised?
    #         # tracking.setWeightForState(f'{coord_path}/speed', 1.0)  # TODO: standard or normalised?
    #     # Otherwise set no weight for state
    #     else:
    #         tracking.setWeightForState(f'{coord_path}/value', 0.0)
    #         tracking.setWeightForState(f'{coord_path}/speed', 0.0)

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

    # Set alternate frames for tracking GRF bodies
    altFramesRightFoot = osim.StdVectorString()
    altFramesRightFoot.append('/bodyset/toes_r')
    altFramesLeftFoot = osim.StdVectorString()
    altFramesLeftFoot.append('/bodyset/toes_l')

    # Read in external loads to identify tracking groups
    ex_loads = osim.ExternalLoads('walk_grf.xml', True)

    # Set the contact tracking groups based on the body in the external loads
    tracking_groups = []
    for load_ind in range(ex_loads.getSize()):
        if ex_loads.get(load_ind).getAppliedToBodyName() == 'calcn_r':
            tracking_groups.append(osim.MocoContactTrackingGoalGroup(forcesRightFoot, ex_loads.get(load_ind).getName(), altFramesRightFoot))
        elif ex_loads.get(load_ind).getAppliedToBodyName() == 'calcn_l':
            tracking_groups.append(osim.MocoContactTrackingGoalGroup(forcesLeftFoot, ex_loads.get(load_ind).getName(), altFramesLeftFoot))

    # # Create plane contact tracking goal for 2D data
    # if model_variant == '2d':
    #     # Create goal
    #     contactTracking = osim.MocoContactTrackingGoal('contact', globalContactTrackingWeight)
    #     # Set external loads
    #     contactTracking.setExternalLoadsFile('walk_grf.xml')
    #     # Add the contact tracking groups
    #     for contact_group in tracking_groups:
    #         contactTracking.addContactGroup(contact_group)
    #     # contactTracking.setNormalizeTrackingError(True)  # TODO: determine if this is desirable? won't work with zeroed data...
    #     # Set to track on 2d plane
    #     contactTracking.setProjection('plane')
    #     contactTracking.setProjectionVector(osim.Vec3(0, 0, 1))
    #     # Add to problem
    #     problem.addGoal(contactTracking)
    # # TODO: otherwise...create 3d goal
    # elif model_variant == '3d':
    #     print('TODO: add contact tracking here...')

    # Create contact tracking goal in each axis data
    contactTracking = []
    # Loop through relevant axes
    if model_variant == '3d':
        axis_list = ['X', 'Y', 'Z']
    elif model_variant == '2d':
        axis_list = ['X', 'Y']
    for ii in axis_list:
        # Create goal
        contactTracking.append(osim.MocoContactTrackingGoal(f'GRF_tracking_{ii}', globalContactTrackingWeight[axis_list.index(ii)]))
        # Add the external loads
        contactTracking[-1].setExternalLoadsFile('walk_grf.xml')
        # Add the contact tracking groups
        for contact_group in tracking_groups:
            contactTracking[-1].addContactGroup(contact_group)
        # Set to use projection vector in tracking
        contactTracking[-1].setProjection('vector')
        # Set axis vector
        if ii == 'X':
            contactTracking[-1].setProjectionVector(osim.Vec3(1, 0, 0))
        elif ii == 'Y':
            contactTracking[-1].setProjectionVector(osim.Vec3(0, 1, 0))
        elif ii == 'Z':
            contactTracking[-1].setProjectionVector(osim.Vec3(0, 0, 1))
        # Add to problem
        problem.addGoal(contactTracking[-1])

    # Add symmetry goal to permit simulating one step
    # -------------------------------------------------------------------------

    # TODO: full gait cycle now so that all joint coordinates can be periodic... just start with testing speed...

    # Create symmetry goal
    symmetry = osim.MocoPeriodicityGoal('symmetry')

    # # TODO: if removing back joint
    # states_data.removeColumn('/jointset/back/lumbar_ext/value')
    # states_data.removeColumn('/jointset/back/lumbar_ext/speed')

    # Set symmetry across all state values and speeds (except for pelvis_tx)
    for state_name in states_data.getColumnLabels():
        if 'pelvis_tx' not in state_name:
            # Set as individual periodic pair
            symmetry.addStatePair(osim.MocoPeriodicityGoalPair(state_name, state_name))

    # Add an individual periodic pair for pelvis_tx speed
    symmetry.addStatePair(osim.MocoPeriodicityGoalPair('/jointset/ground_pelvis/pelvis_tx/speed'))

    # Symmetric muscle activations and tendon forces
    # TODO: ensure this is set-up to work with both 2d and 3d models
    for musc_ind in range(osim_model.getMuscles().getSize()):
        # Get muscle name
        musc = osim_model.getMuscles().get(musc_ind).getName()
        musc_path = osim_model.getMuscles().get(musc).getAbsolutePathString()
        # Add symmetry for activation for muscle for full gait cycle
        symmetry.addStatePair(osim.MocoPeriodicityGoalPair(musc_path + '/activation'))
        # # Tendon force if using elastic tendon
        # if musc in elasticTendons[model_variant]:
        #     symmetry.addStatePair(osim.MocoPeriodicityGoalPair(musc_path + '/normalized_tendon_force'))

    # Symmetric controls
    # TODO: ensure this is set-up to work with both 2d and 3d models --- some for 3d likely need to be negated
    control_names = problem.createRep().createControlInfoNames()
    for control in control_names:
        # Ensure symmetry within control
        symmetry.addControlPair(osim.MocoPeriodicityGoalPair(control))

    # Add to problem
    problem.addGoal(symmetry)

    # Set bounds in problem
    # -------------------------------------------------------------------------

    # TODO: new approach to bounds seems to be messing up problem solution
    # TODO: moreso seems initial activation goal seems to make it harder to solve. There seems to be high activation initially without this though...

    # Muscle bounds
    # problem.setStateInfoPattern('/forceset/.*/normalized_tendon_force', [0, 1.5], [], [])  # TODO: reactivate if using elastic tendons
    problem.setStateInfoPattern('/forceset/.*/activation', [0.01, 1.5], [], [])  # allow muscles to over-activate rather than increasing force

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
            initial_coord_bounds[coord_name] = [initial_val - (val_range * 0.20), initial_val + 0.1]
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
        # solver.set_parameters_require_initsystem(True)  # seems necessary for stiffness parameter to change

        # Reset problem to check any issues
        solver.resetProblem(problem)

        # # Get initial guess to modify
        # initial_guess = solver.getGuess()
        #
        # # Set any tendon forces to be at a reasonable level in initial guess
        # for state_name in initial_guess.getStateNames():
        #     if state_name.endswith('/normalized_tendon_force'):
        #         initial_guess.setState(state_name, np.ones(initial_guess.getNumTimes())*0.2)
        #
        # # Set in solver and reset problem
        # solver.setGuess(initial_guess)
        # solver.resetProblem(problem)

        # # Load in generic guess for tracking
        # generic_guess = osim.MocoTrajectory('InitialGuessFileForTracking.sto')
        #
        # # Convert to full gait cycle
        # generic_guess_full = osim.createPeriodicTrajectory(generic_guess)
        # generic_guess_full.resampleWithNumTimes(initial_guess.getNumTimes())
        #
        # # Fill relevant states and controls from generic guess
        # # Get activations and controls for relevant muscles, plus any untracked states
        # for col in generic_guess_full.getStateNames():
        #     if col in initial_guess.getStateNames() and (col.endswith('/activation') or 'mtp_angle' in col):
        #         initial_guess.setState(col, generic_guess_full.getState(col).to_numpy())
        # for col in generic_guess_full.getControlNames():
        #     if col in initial_guess.getControlNames():
        #         initial_guess.setControl(col, generic_guess_full.getControl(col).to_numpy())
        #
        # # Write to file to check
        # initial_guess.write('initial_guess.sto')
        #
        # # Set update guess and reset problem
        # solver.setGuess(initial_guess)
        # solver.resetProblem(problem)

        # # Get initial guess with tracked states
        # initial_guess = solver.getGuess()
        #
        # # Create guess with updated parameters included
        # created_guess = solver.createGuess()
        #
        # # Resample created guess and fill with speed and value states
        # created_guess.resampleWithNumTimes(initial_guess.getNumTimes())
        # for state_name in initial_guess.exportToStatesTable().getColumnLabels():
        #     created_guess.setState(state_name, initial_guess.getState(state_name).to_numpy())
        #
        # # Set guess in solver and reset problem
        # solver.setGuess(created_guess)
        # solver.resetProblem(problem)

        # # Create the initial created guess from the tracked states
        # initial_guess = solver.getGuess()
        #
        # # Push up pelvis_ty in starting guess to avoid contact spheres starting too far in ground
        # # Set the pelvis_ty starting value to a small increment above where it is in coordinates data
        # initial_guess.setState('/jointset/ground_pelvis/pelvis_ty/value',
        #                        initial_guess.getState('/jointset/ground_pelvis/pelvis_ty/value').to_numpy() + 0.05)
        #
        # # Set guess in solver
        # solver.setGuess(initial_guess)
        #
        # # Reset problem to check any issues
        # solver.resetProblem(problem)

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
    solution.write(f'{participant_id}_initial-torque-tracking_solution_{model_variant}.sto')

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
    osim.STOFileAdapter().write(forces_table, f'{participant_id}_initial-torque-tracking_solution_grf_{model_variant}.sto')

    # Remove initial tracked states and markers file
    os.remove(f'{participant_id}_initial_{model_variant}_tracked_states.sto')

    # Return to home directory
    os.chdir(home_dir)

# =========================================================================
# Start code here ...
# =========================================================================

if __name__ == '__main__':
    print('run main code here')

# %% ---------- end of template.py ---------- %% #
