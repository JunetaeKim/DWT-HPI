# DWT-HPI
An interpretable deep learning model based on discrete wavelet transforms to predict intraoperative hypotension.

## Overal Descriptions
### Data
Since the data used in research cannot be opened due to policy, simulated data can be used for learning and post-analysis of the proposed model. 
Therefore, we provide code to generate simulated data in [DataGeneration.ipynb](https://github.com/JunetaeKim/DWT-HPI/blob/main/DataSimulation/DataGeneration.ipynb) in the [DataSimulation](https://github.com/JunetaeKim/DWT-HPI/tree/main/DataSimulation) folder.
The generated data is stored in the [ProcessedData](https://github.com/JunetaeKim/DWT-HPI/tree/main/ProcessedData) folder.


### Main Study
The source codes for developing the main model and training the model are written in [ModelTraining.py](https://github.com/JunetaeKim/DWT-HPI/blob/main/MainModel/ModelTraining.py) in the [MainModel](https://github.com/JunetaeKim/DWT-HPI/tree/main/MainModel) folder.
The weights of the model, which are the result of running ModelTraining.py, are saved in the hdf5 format in the [Logs](https://github.com/JunetaeKim/DWT-HPI/tree/main/MainModel/Logs) folder.
The file given in the folder is the results of training the model based on the research data and may be used for post-analysis. 

### Ablation Study 
The source codes for developing ablation models and training these models are written in [AblationModel1.py](https://github.com/JunetaeKim/DWT-HPI/blob/main/AblationStudy/Models/AblationModel1.py) and [AblationModel2.py](https://github.com/JunetaeKim/DWT-HPI/blob/main/AblationStudy/Models/AblationModel2.py), respectively in the /AblationStudy/[Models](https://github.com/JunetaeKim/DWT-HPI/tree/main/AblationStudy/Models) folder.
Training results are saved in the hdf5 format in the [Logs](https://github.com/JunetaeKim/DWT-HPI/tree/main/AblationStudy/Logs) folder.
The file given in the folder is the results of training the model based on the research data and may be used for post-analysis in [AblationStudyResult.ipynb](https://github.com/JunetaeKim/DWT-HPI/blob/main/AblationStudy/AblationStudyResult.ipynb).

### Post-hoc Analysis
Post-hoc analysis involves traditional statistical analysis and SHAP analysis to evaluate the interpretabililty of the proposed model.
[MainResult.ipynb](https://github.com/JunetaeKim/DWT-HPI/blob/main/MainResult.ipynb) shows the results based on the actual data used in the study, but is provided for read-only purposes as the data cannot be disclosed.
As an alternative, we have provided an environment for performing post hoc analyzes based on simulated data via [MainResultSIM.ipynb](https://github.com/JunetaeKim/DWT-HPI/blob/main/MainResultSIM.ipynb).
The weight of the model learned by the researcher can be used by loading the given file in the [ModelResults](https://github.com/JunetaeKim/DWT-HPI/tree/main/ModelResults) folder.
That file and the one given in MainModel/Logs are identical.

