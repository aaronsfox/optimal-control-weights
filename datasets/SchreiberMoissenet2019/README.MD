# SchreiberMoissenet2019

*Human* / *Walking* / *Overground*

The data in this folder comes from a [publicly available dataset](https://doi.org/10.6084/m9.figshare.7734767) released with the paper:

**Schreiber C & Moissenet F (2019). A multimodal dataset of human gait at different walking speeds established on injury-free adult participants. *Sci Data*, 6: 111. doi: [10.1038/s41597-019-0124-4](https://doi.org/10.1038/s41597-019-0124-4)**

Any additional use must be in line with that described in the original paper.

## Steps for Extracting & Processing Data

The following steps describe the processes and code (see **Code** section below) used to generate the data for test cases related to the overall aim of this work (i.e. integrating ZMP estimates of GRFs into OpenSim). The final data is already available in this folder (see **Data** section below), however the processes to produce these data can be replicated using these steps.

1. Download the archive of participant data from [this link](https://doi.org/10.6084/m9.figshare.7734767). 
2. Create a `raw` folder in the `data` directory and unzip each participants data into this folder. Each participant should therefore have their own folder within the `raw` directory (e.g. `data\\raw\\2014001`) that contains the `.c3d` files from Schreiber & Moissenet (2019). The participant folder should simply contain all of the individual `.c3d` files for that participant to be appropriately extracted (e.g. `data\\raw\\2014001\\2014001_C1_01.c3d`).
3. Run the **`extractData.py`** script. This will extract and convert the necessary experimental `.c3d` files from the raw folder into individual participant folders within this directory. Note that only the self-selected preferred (C4) speed is extracted. The script also only extracts data for participants who have a viable walking trial for the selected conditions — with a viable trial deemed as those including a gait cycle where each foot contacted a separate force plate. One viable trial is randomly selected from each participant from the selected conditions (i.e. C4) to be used in subsequent analyses. After this you may delete the `raw` folder as it will no longer be used.
4. **[TODO: data processing scripts]**

## Code

The Python scripts in this folder were developed using OpenSim v4.5 alongside Python 3.10 on Windows. The code may work with other versions or operating systems, but has not been tested for this. The script uses relative file paths to extract the data so the code must be run from its own directory (i.e. the `code` folder) — which is typically the default of most Python IDEs.

There are various dependencies associated with the code, which include:

- **[TODO: add dependencies]**

## Model

The model included in this folder and used to process data is a modified version from the work of Rajagopal et al. and Uhlrich et al.:

**Rajagopal A et al. (2016). Full-Body Musculoskeletal Model for Muscle-Driven Simulation of Human Gait. *IEEE Trans Biomed Eng*, 63: 2068-2079. doi: [10.1109/tbme.2016.2586891](https://doi.org/10.1109/tbme.2016.2586891)**

**Uhlrich SD et al. (2022). Muscle coordination retraining inspired by musculoskeletal simulations reduces knee contact force. *Sci Rep*, 12: 9842. doi: [10.1038/s41598-022-13386-9](https://doi.org/10.1038/s41598-022-13386-9)**

The modifications to the model were:

- Adapting the marker set to match that of the experimental data being processed, with some markers on the scapulae removed due to not being tracked in simulations.

## Data [TODO: update according to final layout]

This dataset includes overground walking at various speeds, processed using an OpenSim Moco marker tracking approach. The idea behind this dataset is to demonstrate how ZMP estimates of GRFs are impacted by: (1) walking at different speeds (i.e. self-selected preferred and self-selected fast); and (2) a double support movement. Each participant's processed data will end up in a folder labelled with their unique identification number (see original [Schreiber & Moissenet paper](https://doi.org/10.1038/s41597-019-0124-4) for details). Within these folders, data is separated into:

- **dynamic:** original `.c3d` files alongside the created `.trc` marker files, and `.mot` and `.xml` ground reactions files from dynamic walking trials.
- **events:** contains the extracted timing between toe-off events where kinematic data is extracted between.
- **ik:** set-up files and data processed using OpenSim's inverse kinematics tool. These data serve as the joint angles tracked in simulations.
- **scaling:** set-up files and data from the scaling of the generic model (see *Model* section above).
- **static:** original `.c3d` file alongside the created `.trc` marker file from a static trial.
- **[TODO: simulation folders]:** TODO