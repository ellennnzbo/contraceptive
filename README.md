# Contraceptive Method Choice
This project classifies the contrapcetive method choices of married women surveyed in the 1987 National Indonesia Contraceptive Prevalence Survey. The target variable is the contraceptive method used (=1 no-use, =2 long-term, =3 short-term). The features consist of several demographic, family-related and occupational variables such as the woman's age, education, number of children, religion, employment status as well as the husband's education and occupation. Media exposure and standard-of-living index were also included as attributes.

# Methodology
With a random forest classifier, random search training was used to narrow down the possible values of parameters, `n_estimators`, `max_features`, `max_depth`, `min_samples_split`, `min_sample_leaf`, `bootstrap`. Grid search was then used to test all possible combinations of parameters within a specified range based on the random search parameters. Both random search training and grid search were found to improve accuracy of the model.   
   
From the features importance, it was found that the wife's age and number of children ever born had the largest feature importance in class prediction.

# Dataset
The dataset is obtained from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
* Origin: 1987 National Indonesia Contraceptive Prevalence Survey
* Creator: Tjen-Sien Lim (limt '@' stat.wisc.edu) 
* Donor: Tjen-Sien Lim (limt '@' stat.wisc.edu)
