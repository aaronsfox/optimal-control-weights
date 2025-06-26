%% Script to optimise muscle parameters in scaled models
% This is built from the walking example from the muscle redundancy solver
% as in DeGroote2016

%% Set participant information
% TODO: create loop for participants, set to work through all in some way?
% TODO: auto identify participants and trials

% % % participant_id = '2014014';
% % % trial_name = 'C4_04';
% % % model_variant = '2d';

% Run function
runModelParameterEstimation('SUBJ01', '3d');


%% Define function to solve muscle redundancy problem with parameter estimation

function runModelParameterEstimation(participant_id, model_variant)

    % Settings
    % ---------------------------------------------------------------------

    % Import opensim for use in function
    import org.opensim.modeling.*
    
    % Set data folder paths
    home_dir = pwd;
    data_path = fullfile(home_dir, '..', 'data', participant_id, 'mrs',  model_variant);
    
    % Set file details
    ik_file = {fullfile(data_path,[participant_id,'_walk_ik_',model_variant,'.mot'])};
    id_file = {fullfile(data_path,[participant_id,'_walk_id_',model_variant,'.sto'])};
    model_file = fullfile(home_dir,'..','data',participant_id,'model',[participant_id,'_model_',model_variant,'.osim']);

    % Get muscle names from model for inputs
    participant_model = Model(model_file);
    muscles = participant_model.getMuscles();
    muscle_list = {};
    for ii = 0:muscles.getSize()-1
        muscle_list{ii+1} = string(muscles.get(ii).getName().toString());
    end
    muscle_names = cellfun(@convertStringsToChars, muscle_list, 'UniformOutput', false);
    
    % Pair muscles up by side
    
    % Extract base muscle names and sides
    base_names = cellfun(@(x) x(1:end-2), muscle_names, 'UniformOutput', false);
    sides = cellfun(@(x) x(end), muscle_names, 'UniformOutput', false);
    
    % Create a map of base names to their left/right versions
    unique_names = unique(base_names);
    paired_muscles = cell(length(unique_names), 2);
    
    % Extract paired muscles
    for i = 1:length(unique_names)
        name = unique_names{i};
        % Find indexes for left and right
        idx_r = find(strcmp(base_names, name) & strcmp(sides, 'r'));
        idx_l = find(strcmp(base_names, name) & strcmp(sides, 'l'));
    
        % Fill in left and right columns (empty if not found)
        if ~isempty(idx_r)
            paired_muscles{i,1} = muscle_names{idx_r};
        else
            paired_muscles{i,1} = '';
        end
    
        if ~isempty(idx_l)
            paired_muscles{i,2} = muscle_names{idx_l};
        else
            paired_muscles{i,2} = '';
        end
    end
    
    % Set filter frequency
    filt_freq = 6;
    
    % Input information for muscle redundancy solver
    % ---------------------------------------------------------------------
    
    % Set the IK, ID and model file paths
    Misc.IKfile = ik_file;
    Misc.IDfile = id_file;
    Misc.model_path = model_file;

    % Set analysis options
    Misc.AnalysisID = 'v1';
    Misc.OutPath = fullfile(data_path,'results');
    
    % Get start and end time of the trial
    % At times the tracking solutions can trim to slightly different time
    % ends, so a frame of padding is applied to the time on each end to
    % avoid any errors.
    trial_data = TimeSeriesTable(Misc.IKfile{1});
    trial_size = trial_data.getNumRows();
    time = [double(trial_data.getIndependentColumn().get(2)) ...
        double(trial_data.getIndependentColumn().get(trial_size-2))];
        
    % Filtering
    Misc.f_cutoff_IK = filt_freq;
    Misc.f_cutoff_ID = filt_freq;
    
    % Joint coordinates to use in optimisation
    if strcmp(model_variant,'2d')
        Misc.DofNames_Input = {'hip_flexion_r', 'knee_angle_r', 'ankle_angle_r', ...
            'hip_flexion_l', 'knee_angle_l', 'ankle_angle_l'};
    elseif strcmp(model_variant,'3d')
        Misc.DofNames_Input = {'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', ...
            'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l'};
    end
    
    % Sides to consider in optimisation
    % This is needed, but not sure it actually does anything?
    Misc.opt_sides = {'right'};
    
    % Plotter Bool: Boolean to select if you want to plot lots of output information of intermediate steps in the script
    Misc.PlotBool = true;
    
    % MRS Bool: Select if you want to run the generic muscle redundancy solver
    % Turn off as this tends to fail with generic model
    Misc.MRSBool = 0;
    
    % Validation Bool: Select if you want to run the muscle redundancy solver with the optimized parameters
    Misc.ValidationBool = 1;
    
    % Set name for output
    Misc.OutName = strjoin({participant_id model_variant},'_');
    
    % Set muscles to estimate optimal fibre length for
    % Currently set to all muscles
    Misc.Estimate_OptimalFiberLength = muscle_names;
    % Pair right and left side muscles for optimising fibre lengths
    Misc.Coupled_fiber_length = paired_muscles;
    
    % Set muscles to estimate tendon slack length fot
    % Currently set to all muscles
    Misc.Estimate_TendonSlackLength = muscle_names;
    % Pair right and left side muscles for optimising tendon slack lengths
    Misc.Coupled_slack_length = paired_muscles;
    
    % Set to adapt the stiffness of only the achilles tendon muscles
    % Couple tendon stiffness across plantarflexor muscles
    if strcmp(model_variant,'2d')
        Misc.Estimate_TendonStiffness = {'gastroc_r', 'soleus_r', 'gastroc_l', 'soleus_l'};
        Misc.Coupled_TendonStifness = {'gastroc_r','soleus_r','gastroc_l','soleus_l'};
    elseif strcmp(model_variant,'3d')
        Misc.Estimate_TendonStiffness = {'gaslat_r', 'gasmed_r', 'soleus_r', ...
            'gaslat_r', 'gasmed_r', 'soleus_r'};
        Misc.Coupled_TendonStifness = {'gaslat_r','gasmed_r','soleus_r', ...
            'gaslat_l','gasmed_l','soleus_l'};
    end
    
    % Set bounds for parameter scaling
    Misc.lb_lMo_scaling = 0.5;    % default = 0.7
    Misc.ub_lMo_scaling = 5.0;    % default = 1.5
    Misc.lb_lTs_scaling = 0.1;    % default = 0.7
    Misc.ub_lTs_scaling = 5.0;    % default = 1.5
    Misc.lb_kT_scaling = 0.1;     % default = 0.2
    Misc.ub_kT_scaling = 2.5;     % default = 1.2
    
    % Run muscle redundancy solver estimator
    % ---------------------------------------------------------------------
    [Results,DatStore] = solveMuscleRedundancy(time,Misc);
    
    % NOTE: muscles seem to be reaching optimal fibre length bounds at 3.0 -
    % increase? Similarly tendon slack length is reaching lower bounds for
    % certain muscles too at 0.1 - decrease? Going outside of bounds outlined
    % in above code is possibly extending a bit too much anyway

    % Copy updated parameters model to base folder
    [~, model_name] = fileparts(model_file);
    copyfile(fullfile(Misc.OutPath,[model_name,'_newParams_',Misc.AnalysisID,'.osim']), ...
        replace(model_file, '.osim', '_optim.osim'))

end

