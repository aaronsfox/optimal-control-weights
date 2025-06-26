# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    TODO: add script notes here...
        > Something about model scaling messing with muscles?
        > Adding arms, experimental data causing the problem?
        > The walking data to track may just be bad...?

"""

# =========================================================================
# Import packages
# =========================================================================

import opensim as osim
import os
import shutil
import numpy as np
import argparse

# =========================================================================
# Flags for running analyses
# =========================================================================

# Set participant ID to run
participant = '2014001'

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

# Set the weights for the terms in the objective function.
# These starting points are based off Denton and Umberger, where
# the control effort and tracking weights were set through trial
# and error to give a reasonable balance between minimizing effort
# and matching the reference data well. The weight on the term that
# minimizes the derivatives of the auxiliary variables has been set
# as low as possible while still resulting in smooth muscle activations
# and tendon forces.
globalControlEffortWeight = 10.0
globalStateTrackingWeight = 1.0
globalContactTrackingWeight = 0.5
globalAuxDerivWeight = 0.001

# Set mesh interval
meshInterval = 5  #TODO: testing coarsely... fix to 25...

# Set kinematics filter frequency
kinematicFiltFreq = 6

# Set forces for actuator driven coordinates
# TODO: how to deal with vs. not needing lumbar actuators in 2d vs. 3d?
actForces = {
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

# =========================================================================
# Define functions
# =========================================================================

# Function to run full tracking simulation
# -------------------------------------------------------------------------
def run_tracking_sim(participant_id, trial_name, model_file_name, reference_solution_file,
                     model_variant='3d'):

    """
    :param participant_id:
    :param trial_name:
    :param model_file_name:
    :param reference_solution_file:
    :param model_variant:
    :return:
    """

    # =========================================================================
    # Organise and set-up files
    # =========================================================================

    # Create over-arching folder for data
    # -------------------------------------------------------------------------
    # TODO: need different folders for different weights eventually...
    os.makedirs(os.path.join('..', 'data', participant_id, 'tracking'), exist_ok=True)
    os.makedirs(os.path.join('..', 'data', participant_id, 'tracking', trial_name), exist_ok=True)

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..', 'data', participant_id, 'tracking', trial_name))

    # Copy external loads file to tracking directory
    shutil.copyfile(os.path.join('..', '..', 'dynamic', f'{participant_id}_{trial_name}_grf.mot'),
                    f'{participant_id}_{trial_name}_grf.mot')
    shutil.copyfile(os.path.join('..', '..', 'dynamic', f'{participant_id}_{trial_name}_grf.xml'),
                    f'{participant_id}_{trial_name}_grf.xml')

    # Copy reference simulation file to tracking directory
    shutil.copyfile(os.path.join('..', '..', 'reference', trial_name, reference_solution_file), reference_solution_file)

    # Copy model file to simulation directory
    shutil.copyfile(os.path.join('..', '..', 'scaling', model_file_name), model_file_name)

    # Set-up model for simulation
    # -------------------------------------------------------------------------

    # Read in model
    osim_model = osim.Model(model_file_name)

    # Add coordinate actuators to model
    # Depending on if model is 2d or 3d variant not all actuators will need to be created
    # TODO: dealing with need for lumbar actuators in one but not other
    model_coordinates = [osim_model.getCoordinateSet().get(ii).getName() for ii in range(osim_model.getNumCoordinates())]
    # Loop through coordinates
    for coordinate in actForces:
        # Check if in model coordinates
        if coordinate in model_coordinates:
            # Create actuator
            actu = osim.CoordinateActuator()
            # Set name
            actu.setName(f'{coordinate}')
            # Set coordinate
            actu.setCoordinate(osim_model.updCoordinateSet().get(coordinate))
            # Set optimal force
            actu.setOptimalForce(actForces[coordinate]['optForce'])
            # Set min and max control
            actu.setMinControl(np.inf * -1)
            actu.setMaxControl(np.inf * 1)
            # Append to model force set
            osim_model.updForceSet().cloneAndAppend(actu)

    # Finalise model connections
    osim_model.finalizeConnections()

    # Print model to file
    osim_model.printToXML(f'{participant_id}_{trial_name}_trackingModel_{model_variant}.osim')

    # Get states data for reference simulation
    # -------------------------------------------------------------------------

    # Load in the solution and export to states table data
    states_data = osim.MocoTrajectory(reference_solution_file).exportToStatesTable()

    # Write the states storage object to file
    osim.STOFileAdapter.write(states_data, f'{trial_name}_states_{model_variant}.sto')

    # =========================================================================
    # Set-up tracking simulation
    # =========================================================================

    # Create tracking tool
    track = osim.MocoTrack()
    track.setName(f'{participant_id}_{trial_name}_tracking_{model_variant}')

    # Create model processor
    # track_model_proc = osim.ModelProcessor(f'{participant_id}_{trial_name}_trackingModel_{model_variant}.osim')
    track_model_proc = osim.ModelProcessor('WalkSim_11DOF_18Mus_05_02_2018.osim')

    # Set model in tool
    track.setModel(track_model_proc)

    # Set the states reference file
    table_proc = osim.TableProcessor(f'{trial_name}_states_{model_variant}.sto')
    table_proc.append(osim.TabOpLowPassFilter(kinematicFiltFreq))
    track.setStatesReference(table_proc)

    # Set to ignore unused columns
    track.set_allow_unused_references(True)

    # Set global tracking weight
    track.set_states_global_tracking_weight(globalStateTrackingWeight)

    # Set tracked states to guess
    # TODO: more consistent guess approach?
    track.set_apply_tracked_states_to_guess(True)

    # Set the timings
    # To be safe just use start and end of states file
    track.set_initial_time(osim.TimeSeriesTable(f'{trial_name}_states_{model_variant}.sto').getIndependentColumn()[0])
    track.set_final_time(osim.TimeSeriesTable(f'{trial_name}_states_{model_variant}.sto').getIndependentColumn()[-1])

    # =========================================================================
    # Initialise to a Moco study and problem to finalise
    # =========================================================================

    # Get study and problem
    study = track.initialize()
    problem = study.updProblem()

    # Process model to help with setting up problem
    osim_model = track_model_proc.process()
    osim_model.initSystem()

    # Update control effort goal
    # -------------------------------------------------------------------------

    # Get a reference to the MocoControlCost goal and set parameters
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'))
    effort.setWeight(globalContactTrackingWeight)
    effort.setExponent(2)

    # Update states tracking goal
    # -------------------------------------------------------------------------

    # Get a reference to the states tracking goal
    tracking = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'))

    # Scale individual tracking weights with data range
    # TODO: is this causing issues with global state weight when used?
    # tracking.setScaleWeightsWithRange(True)

    # Set individual weights to avoid tracking
    # TODO: does this conflict in setting to scale weights with range?
    if model_variant == '2d':
        tracking.setWeightForState('/jointset/mtp_r/mtp_angle_r/value', 0)
        tracking.setWeightForState('/jointset/mtp_l/mtp_angle_l/value', 0)
        tracking.setWeightForState('/jointset/mtp_r/mtp_angle_r/speed', 0)
        tracking.setWeightForState('/jointset/mtp_l/mtp_angle_l/speed', 0)
        tracking.setWeightForState('/jointset/ground_pelvis/pelvis_tx/value', 0)
        tracking.setWeightForState('/jointset/ground_pelvis/pelvis_tx/speed', 0)
        tracking.setWeightForState('/jointset/ground_pelvis/pelvis_ty/value', 0)  # TODO: testing without ty tracking
        tracking.setWeightForState('/jointset/ground_pelvis/pelvis_ty/speed', 0)  # TODO: testing without ty tracking

    # Add contact tracking goal
    # -------------------------------------------------------------------------

    # TODO: update as plane for 2D variant...

    # Set right and left contact sphere groups
    forcesLeftFoot = osim.StdVectorString()
    forcesRightFoot = osim.StdVectorString()
    for forceInd in range(osim_model.updForceSet().getSize()):
        if osim_model.getForceSet().get(forceInd).getAbsolutePathString().startswith('/forceset/contact'):
            if osim_model.getForceSet().get(forceInd).getName().endswith('_l'):
                forcesLeftFoot.append(osim_model.updForceSet().get(forceInd).getAbsolutePathString())
            elif osim_model.getForceSet().get(forceInd).getName().endswith('_r'):
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

    # TODO: update as plane for 2D variant...

    contactTracking = osim.MocoContactTrackingGoal('GRF_tracking', globalContactTrackingWeight)
    contactTracking.setExternalLoadsFile(f'{participant_id}_{trial_name}_grf.xml')
    contactTracking.addContactGroup(trackLeftGRF)
    contactTracking.addContactGroup(trackRightGRF)
    contactTracking.setProjection('plane')
    contactTracking.setProjectionVector(osim.Vec3(0, 0, 1))
    problem.addGoal(contactTracking)

    # # Create normalised contact tracking goal in each axis of 3D data
    # if model_variant == '3d':
    #     contactTracking = []
    #     for ii in ['X', 'Y', 'Z']:
    #         contactTracking.append(osim.MocoContactTrackingGoal(f'GRF_tracking_{ii}', globalContactTrackingWeight))
    #         # contactTracking.append(osim.MocoContactTrackingGoal(f'GRF_tracking_{ii}', globalContactTrackingWeight[['X', 'Y', 'Z'].index(ii)]))
    #         contactTracking[-1].setExternalLoadsFile(f'{participant_id}_{trial_name}_grf.xml')
    #         contactTracking[-1].addContactGroup(trackLeftGRF)
    #         contactTracking[-1].addContactGroup(trackRightGRF)
    #         # contactTracking[-1].setNormalizeTrackingError(True)
    #         contactTracking[-1].setProjection('vector')
    #         if ii == 'X':
    #             contactTracking[-1].setProjectionVector(osim.Vec3(1, 0, 0))
    #         elif ii == 'Y':
    #             contactTracking[-1].setProjectionVector(osim.Vec3(0, 1, 0))
    #         elif ii == 'Z':
    #             contactTracking[-1].setProjectionVector(osim.Vec3(0, 0, 1))
    #         # Add to problem
    #         problem.addGoal(contactTracking[-1])

    # Add symmetry goal to permit simulating one step
    # TODO: leave out until sim gets working...
    # -------------------------------------------------------------------------

    # # Create symmetry goal
    # symmetryGoal = osim.MocoPeriodicityGoal('symmetry')
    #
    # # TODO: ensure this is set-up to work with both  2d and 3d models --- there will need to be some negated pairings for 3d model...
    #
    # # Symmetric coordinate values and speeds (except for pelvis_tx)
    # for coord_ind in range(osim_model.getNumCoordinates()):
    #     # Get coordinate name
    #     coord = osim_model.getCoordinateSet().get(coord_ind).getName()
    #     # Check for pelvis_tx
    #     if coord == 'pelvis_tx':
    #         # Only add state pair for symmetric pelvis_tx speed
    #         coord_path = osim_model.getCoordinateSet().get(coord).getAbsolutePathString()
    #         symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/speed'))
    #     # Check for right side
    #     elif coord.endswith('_r'):
    #         # Add symmetry with left side pair for value and speed
    #         coord_path = osim_model.getCoordinateSet().get(coord).getAbsolutePathString()
    #         coord_path_opp = osim_model.getCoordinateSet().get(coord[:-2]+'_l').getAbsolutePathString()
    #         symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/value',
    #                                                                coord_path_opp + '/value'))
    #         symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/speed',
    #                                                                coord_path_opp + '/speed'))
    #     # Check for left side
    #     elif coord.endswith('_l'):
    #         # Add symmetry with right side pair for value and speed
    #         coord_path = osim_model.getCoordinateSet().get(coord).getAbsolutePathString()
    #         coord_path_opp = osim_model.getCoordinateSet().get(coord[:-2] + '_r').getAbsolutePathString()
    #         symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/value',
    #                                                                coord_path_opp + '/value'))
    #         symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/speed',
    #                                                                coord_path_opp + '/speed'))
    #     # Otherwise add matching coordinate value and speed pairs
    #     # TODO: in 3d version some will need to be negated
    #     else:
    #         coord_path = osim_model.getCoordinateSet().get(coord).getAbsolutePathString()
    #         symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/value'))
    #         symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(coord_path + '/speed'))
    #
    # # Symmetric muscle activations
    # # TODO: ensure this is set-up to work with both 2d and 3d models ---
    # for musc_ind in range(osim_model.getMuscles().getSize()):
    #     # Get muscle name
    #     musc = osim_model.getMuscles().get(musc_ind).getName()
    #     # Check for right side
    #     if musc.endswith('_r'):
    #         # Add symmetry with left side pair for activation and normalised tendon force
    #         musc_path = osim_model.getMuscles().get(musc).getAbsolutePathString()
    #         musc_path_opp = osim_model.getMuscles().get(musc[:-2]+'_l').getAbsolutePathString()
    #         symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(musc_path + '/activation',
    #                                                                musc_path_opp + '/activation'))
    #         symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(musc_path + '/normalized_tendon_force',
    #                                                                musc_path_opp + '/normalized_tendon_force'))
    #     # Check for left side
    #     elif musc.endswith('_l'):
    #         # Add symmetry with right side pair for activation and normalised tendon force
    #         musc_path = osim_model.getMuscles().get(musc).getAbsolutePathString()
    #         musc_path_opp = osim_model.getMuscles().get(musc[:-2] + '_r').getAbsolutePathString()
    #         symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(musc_path + '/activation',
    #                                                                musc_path_opp + '/activation'))
    #         symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(musc_path + '/normalized_tendon_force',
    #                                                                musc_path_opp + '/normalized_tendon_force'))
    #
    # # Symmetric controls
    # # TODO: ensure this is set-up to work with both 2d and 3d models ---
    # control_names = problem.createRep().createControlInfoNames()
    # for control in control_names:
    #     # Check for right side control
    #     if control.endswith('_r'):
    #         # Pair with left side control
    #         symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair(control, control[:-2]+'_l'))
    #     elif control.endswith('_l'):
    #         # Pair with right side control
    #         symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair(control, control[:-2] + '_r'))
    #     else:
    #         # Pair with self
    #         # TODO: negated controls for some?
    #         symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair(control))
    #
    # # Add to problem
    # problem.addGoal(symmetryGoal)

    # Prescribed average gait speed
    # -------------------------------------------------------------------------

    # Get average speed from reference file
    avg_speed = states_data.getDependentColumn('/jointset/ground_pelvis/pelvis_tx/speed').to_numpy().mean()

    # Create goal
    speedGoal = osim.MocoAverageSpeedGoal('speed')
    speedGoal.set_desired_average_speed(avg_speed)

    # Add to problem
    problem.addGoal(speedGoal)

    # =========================================================================
    # Set bounds in problem
    # =========================================================================

    # Coordinate values and speeds
    # -------------------------------------------------------------------------
    # t_final is fixed in the tracking problem
    # TODO: set these more appropriately with 3D vs. 2D --- these also might not be relevant coming directly from DentonUmberger code
    problem.setStateInfo('/jointset/ground_pelvis/pelvis_tilt/value', [-20 * np.pi / 180, 10 * np.pi / 180])
    problem.setStateInfo('/jointset/ground_pelvis/pelvis_tx/value', [-1, 2], 0, [0.5, 1.0])
    problem.setStateInfo('/jointset/ground_pelvis/pelvis_ty/value', [0.75, 1.25])
    problem.setStateInfo('/jointset/hip_l/hip_flexion_l/value', [-20 * np.pi / 180, 60 * np.pi / 180])
    problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value', [-20 * np.pi / 180, 60 * np.pi / 180])
    problem.setStateInfo('/jointset/knee_l/knee_angle_l/value', [-80 * np.pi / 180, 10 * np.pi / 180])
    problem.setStateInfo('/jointset/knee_r/knee_angle_r/value', [-80 * np.pi / 180, 10 * np.pi / 180])
    problem.setStateInfo('/jointset/ankle_l/ankle_angle_l/value', [-35 * np.pi / 180, 25 * np.pi / 180])
    problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value', [-35 * np.pi / 180, 25 * np.pi / 180])
    problem.setStateInfo('/jointset/mtp_r/mtp_angle_r/value', [-25 * np.pi / 180, 75 * np.pi / 180])
    problem.setStateInfo('/jointset/mtp_l/mtp_angle_l/value', [-25 * np.pi / 180, 75 * np.pi / 180])

    # Activation and control bounds
    # -------------------------------------------------------------------------
    problem.setStateInfoPattern('/forceset/.*/normalized_tendon_force', [0, 1.5], [], [])
    problem.setStateInfoPattern('/forceset/.*/activation',  [0.01, 1.0], [], [])
    # problem.setControlInfoPattern('.*', [0.01, 1.0], [], [])  # TODO: limit coordinate actuator controls?

    # =========================================================================
    # Define and configure the solver
    # =========================================================================

    # Get a reference to the solver
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    solver.resetProblem(problem)

    # Set solver options
    solver.set_optim_max_iterations(2000)
    solver.set_num_mesh_intervals(meshInterval)
    solver.set_optim_constraint_tolerance(1e-3)
    solver.set_optim_convergence_tolerance(1e-3)
    solver.set_minimize_implicit_auxiliary_derivatives(True)
    solver.set_implicit_auxiliary_derivatives_weight(globalAuxDerivWeight)

    # Get the initial guess to manipulate
    initial_guess = solver.getGuess()

    # Push up pelvis_ty in starting guess to avoid contact spheres starting too far in ground
    # Set the pelvis_ty starting value to a small increment above where it is in coordinates data
    initial_guess.setState('/jointset/ground_pelvis/pelvis_ty/value',
                           initial_guess.getState('/jointset/ground_pelvis/pelvis_ty/value').to_numpy() + 0.1)

    # Set the guess in solver
    solver.setGuess(initial_guess)

    # Reset problem to finalise
    solver.resetProblem(problem)

    # Solve the problem
    # -------------------------------------------------------------------------
    solution = study.solve()

    '''
    Notes on muscle driven approach:
        >    
    '''

    # # Option to visualise solution
    # study.visualize(solution)

    # Save files and finalize
    # -------------------------------------------------------------------------

    # # Write solution to file
    # # TODO: different names?
    # if solution.isSealed():
    #     solution.unseal()
    # solution.write(f'{participant_id}_{trial_name}_reference-solution_{model_variant}.sto')

    # Remove initial tracked states and markers file
    os.remove(f'{participant_id}_{trial_name}_reference_{model_variant}_tracked_states.sto')

    # Return to home directory
    os.chdir(home_dir)

# =========================================================================
# Run analysis
# =========================================================================

if __name__ == '__main__':
    print('run main code here')

# %% ---------- end of tracking_sims.py ---------- %% #
