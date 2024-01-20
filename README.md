# CMAPSS_master
These are the files I created to do my Master Thesis about Predictive Maintenance of the CMAPSS dataset.
I created regression models to calculate RUL, based on the following papers, 
**[Variational encoding approach for interpretable assessment of remaining useful life estimation](https://www.sciencedirect.com/science/article/pii/S0951832022000321?via%3Dihub)** and **[Ensemble Neural Networks for Remaining Useful Life (RUL) Prediction](https://arxiv.org/abs/2309.12445)**.  
  
Also created classification models to calculate the health index of the engines. This work is based on **[Temporal Classification of Turbofan Engine Health using Elman Recurrent Network](https://www.researchgate.net/profile/Cairo-Nascimento-Jr/publication/343647117_Temporal_Classification_of_Turbofan_Engine_Health_using_Elman_Recurrent_Network/links/5f9bed89299bf1b53e5149fb/Temporal-Classification-of-Turbofan-Engine-Health-using-Elman-Recurrent-Network.pdf)**

## Files in this Repository
* \CMAPSSData: CMAPSS dataset
* Data_exploration_Final.ipynb: explore, clean and preprocess dataset.
* Plots_and_reports.ipynb: plot results for regression and classification models.
* Regression.ipynb: build and training of the regression models.
* classification.ipynb: build and training of the classification models.
* utils.py: collection of the functions used in the notebooks.
