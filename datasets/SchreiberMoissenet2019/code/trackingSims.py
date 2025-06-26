# -*- coding: utf-8 -*-
"""

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This code generates the muscle-driven tracking sims from the SchreiberMoissenet2019
    dataset. After extracting the data and generating the baseline simulations, this
    code can be used to generate tracking simulations with different weight parameters.



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

# Set tracking weight for optimal control problem
trackingWeight = 1e0
trackingWeight_grf = trackingWeight * 1e4

# Set the trial name
trial = 'C4_05'

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

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', 'model', 'Geometry'))

# Set standard weights for optimisations
activationsWeight = 1e-1

# Set mesh interval
# TODO: appropriate?
meshInterval = 25

# Set actuator forces to drive simulations
actForces = {
    # TODO: are residuals needed for this to solve nicely?
    'pelvis_tx': {'actuatorType': 'residual', 'optForce': 2.5},
    'pelvis_ty': {'actuatorType': 'residual', 'optForce': 2.5},
    'pelvis_tz': {'actuatorType': 'residual', 'optForce': 2.5},
    'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 1.0},
    'pelvis_list': {'actuatorType': 'residual', 'optForce': 1.0},
    'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 1.0},
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

# Set marker names for contact sphere locations to foot
contactMarkers = [
    'R_FM1_ground', 'R_FM2_ground', 'R_FM5_ground', 'R_FCC_ground', 'R_FM1_mid_ground', 'R_FM5_mid_ground',
    # 'R_FM2_mid_ground',
    'L_FM1_ground', 'L_FM2_ground', 'L_FM5_ground', 'L_FCC_ground', 'L_FM1_mid_ground', 'L_FM5_mid_ground',
    # 'L_FM2_mid_ground',
]

# Set markers to look up for ground contact constraints
constraintMarkers = [
    'R_FM1_ground', 'R_FM2_ground', 'R_FM5_ground', 'R_FCC_ground',
    # 'R_FM1_mid_ground', 'R_FM2_mid_ground', 'R_FM5_mid_ground',
    'L_FM1_ground', 'L_FM2_ground', 'L_FM5_ground', 'L_FCC_ground'
    # 'L_FM1_mid_ground', 'L_FM2_mid_ground', 'L_FM5_mid_ground'
]

# Contact sphere settings
stiffness = 3067776
dissipation = 2.0
staticFriction = 0.8
dynamicFriction = 0.8
viscousFriction = 0.5
transitionVelocity = 0.2

# Set contact sphere radii
contactHeelRadius = 0.030
contactMidfootRadius = 0.025
contactToeRadius = 0.020

# =========================================================================
# Set-up folders and files for trial
# =========================================================================

# Create over-arching folder for data
os.makedirs(os.path.join('..', 'data', participant, 'tracking'), exist_ok=True)
os.makedirs(os.path.join('..', 'data', participant, 'tracking', trial), exist_ok=True)

# Create folder for storing simulation specific data
os.makedirs(os.path.join('..', 'data', participant, 'tracking', trial,
                         'tw'+str(trackingWeight).replace('.','-')), exist_ok=True)

# Navigate to simulation folder for ease of use
homeDir = os.getcwd()
os.chdir(os.path.join('..', 'data', participant, 'tracking', trial,
                      'tw'+str(trackingWeight).replace('.','-')))

# Copy external loads file to directory
shutil.copyfile(os.path.join('..', '..', '..', 'dynamic', f'{participant}_{trial}_grf.mot'),
                f'{participant}_{trial}_grf.mot')
shutil.copyfile(os.path.join('..', '..', '..', 'dynamic', f'{participant}_{trial}_grf.xml'),
                f'{participant}_{trial}_grf.xml')

# Copy baseline simulation solution file to directory
shutil.copyfile(os.path.join('..', '..', '..', 'baseline', trial, f'{participant}_{trial}_baselineSolution.sto'),
                f'{participant}_{trial}_baselineSolution.sto')

# Create a values and speeds file for tracking purposes

# Load in file
trackingData = osim.TimeSeriesTable(f'{participant}_{trial}_baselineSolution.sto')

# Remove any columns that aren't values or speeds
for colName in trackingData.getColumnLabels():
    if not colName.endswith('/value') and not colName.endswith('/speed'):
        trackingData.removeColumn(colName)

# Export to file
osim.STOFileAdapter().write(trackingData, f'{participant}_{trial}_trackingData.sto')

# =========================================================================
# Set up model for tracking simulation
# =========================================================================

# Load the model
osimModel = osim.Model(os.path.join('..', '..', '..', 'scaling', f'{participant}_scaledModelAdjusted.osim'))

# There are already some upper body torque actuators in the model
# Clear these to add our own with consistent naming
# This is a little tricky to loop through as constantly removing forces changes their indice

# Get the force names that are to be removed (they will be those that aren't muscles)
removeForces = []
for fInd in range(osimModel.getForceSet().getSize()):
    if 'Muscle' not in osimModel.getForceSet().get(fInd).getConcreteClassName():
        removeForces.append(osimModel.getForceSet().get(fInd).getName())

# Loop through forces to remove
for fName in removeForces:
    # Get the current index of force
    fInd = [ii for ii in range(osimModel.getForceSet().getSize()) if osimModel.getForceSet().get(ii).getName() == fName][0]
    # Remove the force
    osimModel.updForceSet().remove(fInd)

# Finalise model connections
osimModel.finalizeConnections()

# Use model processors to speed up parts of the process
# -------------------------------------------------------------------------

# Construct a model processor to further edit the model
modelProc = osim.ModelProcessor(osimModel)

# Weld desired locked joints
# Create vector string object
weldVectorStr = osim.StdVectorString()
[weldVectorStr.append(joint) for joint in ['subtalar_r', 'subtalar_l', 'mtp_r', 'mtp_l',
                                           'radius_hand_r', 'radius_hand_l']]
# Append to model processor
modelProc.append(osim.ModOpReplaceJointsWithWelds(weldVectorStr))

# Convert muscles to DeGrooteFregley model
modelProc.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())

# Increase muscle isometric force by a scaling factor to deal with potentially higher muscle forces
modelProc.append(osim.ModOpScaleMaxIsometricForce(1.5))

# Set to ignore tendon compliance
# TODO: appropriate?
modelProc.append(osim.ModOpIgnoreTendonCompliance())

# Ignore passive fibre forces
# TODO: appropriate?
modelProc.append(osim.ModOpIgnorePassiveFiberForcesDGF())

# Scale active force curve width
modelProc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

# Process model for further edits
trackingModel = modelProc.process()

# Append additional actuators required in model
# -------------------------------------------------------------------------

# Add coordinate actuators to upper body of model
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
    # if actForces[coordinate]['actuatorType'] == 'residual':
    #     actu.setMinControl(-1)
    #     actu.setMaxControl(1)
    # else:
    actu.setMinControl(np.inf * -1)
    actu.setMaxControl(np.inf * 1)
    # Append to model force set
    trackingModel.updForceSet().cloneAndAppend(actu)

# Add contact geometry to model
# -------------------------------------------------------------------------

# Create the half space for the floor
contactHalfSpace = osim.ContactHalfSpace(osim.Vec3(0, 0, 0),
                                         osim.Vec3(0, 0, -1.5707963267948966),
                                         trackingModel.getGround(),
                                         'floor')

# Connect to model
trackingModel.addContactGeometry(contactHalfSpace)

# Create and add the contact spheres
for marker in contactMarkers:

    # Get location and frame of marker
    location = trackingModel.updMarkerSet().get(marker).get_location()
    frame = osim.PhysicalFrame.safeDownCast(
        trackingModel.updBodySet().get(trackingModel.updMarkerSet().get(marker).getParentFrameName().split('/')[-1]))

    # Create contact geometry
    if '_FCC' in marker:
        contactSphere = osim.ContactSphere(contactHeelRadius, location, frame,
                                           f'{marker}_contactSphere')
    elif '_mid' in marker:
        contactSphere = osim.ContactSphere(contactMidfootRadius, location, frame,
                                           f'{marker}_contactSphere')
    else:
        contactSphere = osim.ContactSphere(contactToeRadius, location, frame,
                                           f'{marker}_contactSphere')

    # Connect to model
    trackingModel.addContactGeometry(contactSphere)

    # Create the sphere force to go with the contact spheres
    sphereForce = osim.SmoothSphereHalfSpaceForce(f'{marker}_contactForce',
                                                  contactSphere, contactHalfSpace)

    # Set sphere force parameters
    sphereForce.set_stiffness(stiffness)
    sphereForce.set_dissipation(dissipation)
    sphereForce.set_static_friction(staticFriction)
    sphereForce.set_dynamic_friction(dynamicFriction)
    sphereForce.set_viscous_friction(viscousFriction)
    sphereForce.set_transition_velocity(transitionVelocity)
    sphereForce.connectSocket_half_space(contactHalfSpace)
    sphereForce.connectSocket_sphere(contactSphere)

    # Add the force to model
    trackingModel.addForce(sphereForce)

# Add metabolics to the model muscles
# -------------------------------------------------------------------------

# Create the metabolic cost model
metabolics = osim.Bhargava2004SmoothedMuscleMetabolics()
metabolics.setName('metCost')
metabolics.set_use_smoothing(True)

# Add muscle components to the metabolic cost model
for muscleInd in range(trackingModel.getMuscles().getSize()):
    metabolics.addMuscle(trackingModel.getMuscles().get(muscleInd).getName(),
                         trackingModel.getMuscles().get(muscleInd))

# Add the metabolics component to the model
trackingModel.addComponent(metabolics)

# Finalise and save model
# -------------------------------------------------------------------------

# Finalise model connections
trackingModel.finalizeConnections()

# Print model to file in tracking directory
trackingModel.printToXML(f'{participant}_{trial}_trackingModel.osim')

# =========================================================================
# Run tracking simulation
# =========================================================================

# Create tracking tool
track = osim.MocoTrack()
track.setName(f'{participant}_{trial}_tracking_{"tw"+str(trackingWeight).replace(".","-")}')

# Set model
trackModelProc = osim.ModelProcessor(f'{participant}_{trial}_trackingModel.osim')
track.setModel(trackModelProc)

# Set the coordinates reference file
tableProcessor = osim.TableProcessor(f'{participant}_{trial}_trackingData.sto')
track.setStatesReference(tableProcessor)

# Set to ignore unused columns
track.set_allow_unused_references(True)

# Set global tracking weight
track.set_states_global_tracking_weight(trackingWeight)

# Set tracked states to guess
track.set_apply_tracked_states_to_guess(True)

# Set the timings
# Note that these are edited lately given final time isn't constrained
track.set_initial_time(osim.TimeSeriesTable(f'{participant}_{trial}_trackingData.sto').getIndependentColumn()[0])
track.set_final_time(osim.TimeSeriesTable(f'{participant}_{trial}_trackingData.sto').getIndependentColumn()[-1])

# Initialise to a Moco study and problem to finalise
# -------------------------------------------------------------------------

# Get study and problem
study = track.initialize()
problem = study.updProblem()

# Update control effort goal
# -------------------------------------------------------------------------

# Get a reference to the MocoControlCost goal and set parameters
effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'))
effort.setWeight(activationsWeight)
effort.setExponent(3)  # cubed activations potentially better representation of metabolic cost?

# Put high weight on residual use
effort.setWeightForControlPattern('/forceset/.*_residual', 5.0)

# Update individual weights in control effort goal
# Put lower weight on torque acuators so that they contribute little to ojective
effort.setWeightForControlPattern('/forceset/.*_torque', 1e-3)

# Update states tracking goal
# -------------------------------------------------------------------------

# Get a reference to the states tracking goal
tracking = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'))

# Scale individual tracking weights with data range
tracking.setScaleWeightsWithRange(True)

# Add contact tracking goal
# -------------------------------------------------------------------------

# Set right and left contact sphere groups
forcesRightFoot = osim.StdVectorString()
forcesLeftFoot = osim.StdVectorString()
for forceInd in range(trackingModel.updForceSet().getSize()):
    if '_contactForce' in trackingModel.getForceSet().get(forceInd).getAbsolutePathString():
        if trackingModel.getForceSet().get(forceInd).getName().startswith('R_'):
            forcesRightFoot.append(trackingModel.updForceSet().get(forceInd).getAbsolutePathString())
        elif trackingModel.getForceSet().get(forceInd).getName().startswith('L_'):
            forcesLeftFoot.append(trackingModel.updForceSet().get(forceInd).getAbsolutePathString())

# Read in external loads to identify tracking groups
exLoads = osim.ExternalLoads(f'{participant}_{trial}_grf.xml', True)

# Set the left and right contact tracking groups
if exLoads.get(0).getAppliedToBodyName() == 'calcn_l':
    trackLeftGRF = osim.MocoContactTrackingGoalGroup(forcesLeftFoot, exLoads.get(0).getName())
    trackRightGRF = osim.MocoContactTrackingGoalGroup(forcesRightFoot, exLoads.get(1).getName())
else:
    trackLeftGRF = osim.MocoContactTrackingGoalGroup(forcesLeftFoot, exLoads.get(1).getName())
    trackRightGRF = osim.MocoContactTrackingGoalGroup(forcesRightFoot, exLoads.get(0).getName())

# Set alternative frame paths
trackLeftGRF.set_alternative_frame_paths(0,'/bodyset/toes_l')
trackRightGRF.set_alternative_frame_paths(0,'/bodyset/toes_r')

# Create normalised contact tracking goal in each axis of data
contactTracking = []
for ii in ['X','Y','Z']:
    contactTracking.append(osim.MocoContactTrackingGoal(f'GRF_tracking_{ii}', trackingWeight_grf))
    contactTracking[-1].setExternalLoadsFile(f'{participant}_{trial}_grf.xml')
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

# Add speed goal to problem
# -------------------------------------------------------------------------

# Create the goal
speedGoal = osim.MocoAverageSpeedGoal('speed')

# Calculate the average speed from the tracking kinematics
avgSpeed = np.nanmean(osim.TimeSeriesTable(f'{participant}_{trial}_trackingData.sto').getDependentColumn(
    '/jointset/ground_pelvis/pelvis_tx/speed').to_numpy())

# Set speed in goal
speedGoal.set_desired_average_speed(avgSpeed)

# Add to problem
problem.addGoal(speedGoal)

# Add constraints to problem
# -------------------------------------------------------------------------

# Constrain the muscle activations at the initial time point to equal
# the initial muscle excitation value.
problem.addGoal(osim.MocoInitialActivationGoal('initial_activation'))

# Constrain contact markers to at minimum be at ground level
# Note that this uses a simplified set to not overly complicate constraints in the problem

# Loop through markers to create path constraints
for marker in constraintMarkers:
    # Create constraint
    markerConstraint = osim.MocoOutputConstraint()
    markerConstraint.setName(f'{marker}_constraint')
    # Set path to marker location
    markerConstraint.setOutputPath(f'/markerset/{marker}|location')
    # Set output index to y-axis
    markerConstraint.setOutputIndex(1)
    # Create and set the bounds to not go below ground level and a reasonable height
    markerBounds = osim.StdVectorMocoBounds()
    markerBounds.append(osim.MocoBounds(0.00, 0.25))
    markerConstraint.updConstraintInfo().setBounds(markerBounds)
    # Add to problem
    problem.addPathConstraint(markerConstraint)

# Ensure periodicity of the half gait cycle via symmetry constraints
# -------------------------------------------------------------------------

# Create symmetry goal
symmetryGoal = osim.MocoPeriodicityGoal('symmetry')

# Initialise model state to get state names
trackingModel.initSystem()

# Loop through states and pair up appropriately
# The below individualised checking style is a bit verbose, but allows for specific aspects to be included vs. commented out
# Right side values, speeds and muscle activations are paired with their corresponding left side
# Left side values, speeds and muscle activations are paired with their corresponding right side
# Pelvis tilt, pelvis ty, pelvis tz and lumbar extension + pelvis tx speed are set to be symmetric with self
# Pelvis list, pelvis rotation, lumbar bending and lumbar rotation are set to be negated symmetric with self
# Pelvis tx speed is set to be symmetric
for stateInd in range(trackingModel.getStateVariableNames().getSize()):

    # Get state name
    stateName = trackingModel.getStateVariableNames().get(stateInd)

    # Check for right side coordinate value
    if stateName.endswith('_r/value'):
        symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(stateName, stateName.replace('_r/', '_l/')))

    # Check for left side coordinate value
    if stateName.endswith('_l/value'):
        symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(stateName, stateName.replace('_l/', '_r/')))

    # Check for right side coordinate speed
    if stateName.endswith('_r/speed'):
        symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(stateName, stateName.replace('_r/', '_l/')))

    # Check for left side coordinate speed
    if stateName.endswith('_l/speed'):
        symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(stateName, stateName.replace('_l/', '_r/')))

    # Check for coordinates or speeds that are set to be symmetric with self
    if any(ix in stateName for ix in ['pelvis_tilt', 'pelvis_ty', 'pelvis_tz', 'lumbar_extension', 'pelvis_tx/speed']):
        symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(stateName))

    # Check for coordinates or speeds that are to be negated symmetric with self
    if any(ix in stateName for ix in ['pelvis_list', 'pelvis_rotation', 'lumbar_bending', 'lumbar_rotation']):
        symmetryGoal.addNegatedStatePair(osim.MocoPeriodicityGoalPair(stateName))

    # Check for right side muscle activation
    if stateName.endswith('_r/activation'):
        symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(stateName, stateName.replace('_r/', '_l/')))

    # Check for left side muscle activation
    if stateName.endswith('_l/activation'):
        symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(stateName, stateName.replace('_l/', '_r/')))

# Loop through torque controls and pair up appropriately
# Right side controls are paired with their corresponding left side
# Left controls are paired with their corresponding right side

# TODO: is this necessary given we don't care much about the torque control signals?

# # Add to problem
# problem.addGoal(symmetryGoal)

# TODO: symmetry goal commented out to see if it's causing issues...

# Set bounds in problem
# -------------------------------------------------------------------------

# Set time bounds so that problem always starts at the same time
# An end time bound range is set so that it is in the rough same duration as experimental data
# This shouldn't be a huge problem as the speed goal should guide completion

# Get initial time and duration
initialTime = osim.TimeSeriesTable(f'{participant}_{trial}_trackingData.sto').getIndependentColumn()[0]
duration = osim.TimeSeriesTable(f'{participant}_{trial}_trackingData.sto').getIndependentColumn()[-1] - initialTime

# Set time bounds
problem.setTimeBounds(initialTime, [initialTime + (duration * 0.9), initialTime + (duration * 1.1)])

# Define and configure the solver
# -------------------------------------------------------------------------
solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

# Set solver options
solver.set_optim_max_iterations(1000)
solver.set_num_mesh_intervals(meshInterval)  # TODO: testing with coarse to begin with for speed...update!!!!!
solver.set_optim_constraint_tolerance(1e-2)  # TODO: appropriate for this problem?
solver.set_optim_convergence_tolerance(1e-3)  # TODO: appropriate for this problem?

# Reset problem to avoid any issues
solver.resetProblem(problem)

# Modify initial guess
# -------------------------------------------------------------------------

# TODO: any issues with guess mostly staying as is?

#Get the initial guess to manipulate
initialGuess = solver.getGuess()

# Push up pelvis_ty in starting guess to avoid contact spheres starting too far in ground
# Set the pelvis_ty starting value to a small increment above where it is in coordinates data
initialGuess.setState('/jointset/ground_pelvis/pelvis_ty/value',
                      initialGuess.getState('/jointset/ground_pelvis/pelvis_ty/value').to_numpy() + 0.1)

# Set the guess in solver
solver.setGuess(initialGuess)

# Reset problem to finalise
solver.resetProblem(problem)

# Solve the problem
# -------------------------------------------------------------------------
solution = study.solve()

# # Option to visualise solution
# study.visualize(solution)

# Save files and finalize
# -------------------------------------------------------------------------

# # Write solution to file
# if solution.isSealed():
#     solution.unseal()
# solution.write(f'{participant}_{trial}_solved.sto')
#
# # Remove initial tracked states and markers file
# os.remove(f'{participant}_{trial}_baseline_tracked_states.sto')

# TODO: metabolic cost outputs...

# Return to home directory
os.chdir(homeDir)

# Print out confirmation for participant
# -------------------------------------------------------------------------
# print(f'Baseline simulations completed for {participant}...')

# =========================================================================
# Finalise and exit
# =========================================================================

# Exit console to avoid exit code error
os._exit(00)

# %% ----- end of trackingSims.py ----- %% #