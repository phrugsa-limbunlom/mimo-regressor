# Feature-Driven Optimization of Syngas Production in Biomass Gasifiers using Machine Learning

**Abstract**—Biomass gasification is a significant thermal conversion method that uses fluidized bed reactors to generate syngas compositions with low heating values. Machine learning models were adopted to
predict biomass composition and operating conditions. However, there has not yet been a comprehensive model developed through selective feature optimization. In this research, four regression machine-learning
models were employed. The predictive capacity of syngas compositions and lower heating values (LHV) were assessed. The output products were derived from various lignocellulosic biomass feedstocks across
a wide array of operating conditions. The four regression machine learning algorithms are Decision Tree, Support Vector Machine (SVM), XGBoost, and Random Forest (RF), which were adopted to evaluate prediction performance after undergoing hyperparameter and feature selection optimization. Pearson correlation was applied to validate the correlation between input and output variables. XGBoost and RF established good performance results (XGBoost: R2 = 0.567–0.892, RMSE= 0.880–9.645; RF: R2 = 0.675–0.855, RMSE = 1.336–10.558). XGBoost provided low RMSE scores in CH4, LHV, and Tar yield (1.495,0.880, and 9.645) and a high R2 score in LHV (0.892), whereas RF produced low RMSE scores in LHV and Tar yield (1.215 and 9.614). The XGBoost algorithm selected seven features after optimization, including cellulose, hemicellulose, lignin, temperature, pressure, equivalence ratio (ER), and steam-to-biomass ratio (SBR). In contrast, the RF algorithm selected all features, including cellulose, hemicellulose, lignin, temperature, pressure, equivalence ratio (ER), steam-to-biomass ratio (SBR), and superficial gas velocity.

[Paper](https://github.com/phrugsa-limbunlom/CE880_Case_Study_MIMO_Biosyngas_Prediction/blob/main/%5B2311569%5D_CE880_Case_Study.pdf)

## To run the file
1. Update the folder and path name of the data set
![image](https://github.com/Gift-eiei/CE880_Case_Study/assets/59916413/31017832-c360-48c9-87f6-0360a1466b54)
2. Run the model to tune hyperparameters
![image](https://github.com/Gift-eiei/CE880_Case_Study/assets/59916413/c7586a39-4c09-4fe1-aea6-5fa9cdd89f79)
3. Verify the model score by running unit test
![image](https://github.com/Gift-eiei/CE880_Case_Study/assets/59916413/ce48d705-8e59-4ef9-a8b9-ffe98a31cd7c)
4. Run functions of train model and prediction data
![image](https://github.com/Gift-eiei/CE880_Case_Study/assets/59916413/305d1f10-2ffe-4428-8c7b-2a797268a5e6)
![image](https://github.com/Gift-eiei/CE880_Case_Study/assets/59916413/ed4660d8-4379-4b8e-8b1e-39ebd9afb59b)
6. Verify the scores of predicted data by runnning unit test
![image](https://github.com/Gift-eiei/CE880_Case_Study/assets/59916413/f4f1979f-c699-4290-bba9-312cc8124662)



