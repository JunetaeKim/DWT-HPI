# DWT-HPI
An interpretable deep learning model based on discrete wavelet transforms to predict intraoperative hypotension.
<!--The research on the model is currently under 2nd round review at [IEEE Transactions on Neural Networks and Learning Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385).-->
### Model Structures
<img src="https://github.com/JunetaeKim/DWT-HPI/blob/main/Figures/ModelStructure.jpg" width=80% height=80%>

### Scenario-based Guideline
<img src="https://github.com/JunetaeKim/DWT-HPI/blob/main/Figures/ScenarioBasedGuideline.jpg" width=70% height=70%>

## Overal Descriptions
### Data
Since the data used in research cannot be disclosed due to policy, simulated data can be used to train the proposed model and perform post-hoc analysis. 
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

## Notification
The models in this study were written and tested based on Tensorflow(version==2.4.0 and 2.10.0), Tensorflow-probability(version==0.12.1 and 0.18.0) and SHAP(version==0.40.0 and 0.41.0).
We noticed that the arguments of shap.GradientExplainer are slightly different depending on the version of the SHAP package, so be sure to check the instructions in the [SHAP](https://shap.readthedocs.io/en/latest/) package.

We wrote the model development code in a hackish way. In other words, in Tensorflow, custom layers can be created in a subclassing method, which is the standard way. However, this way places the weight matrices inside the custom layer class, which may limit flexible debugging and make it difficult for new readers to understand the code from a procedural perspective. Thus, we wrote the code to enable flexible operation by defining the weight matrices in a custom layer and returning them as Keras Symbolic Tensors without any operation inside the custom layer. Anyone can rewrite the code of this model in a standard way (i.e. overriding class functions inherited from Tensorflow) or extend it with other deep learning development tools such as pytorch.


## Development Contributorship
[Junetae Kim](https://github.com/JunetaeKim) developed, trained, and tuned the main- and ablation models. 
[Eugene Hwang](https://github.com/joyce0215) tuned the main- and ablation models, and conducted experiments on model interpretability, performed benchmarking tests, and visualized scenario-based guidelines. 
[Jinyoung Kim](https://github.com/sacross93) refactored and structured the code.

