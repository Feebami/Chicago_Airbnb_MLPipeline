# Chicago_Airbnb_MLPipeline
 An ML pipeline to determine rental unit nightly charges showcased in Jupyter Notebooks

The goal of this project was to maximize the revenue of Airbnb listings given the dataset found [here](https://www.kaggle.com/datasets/jinbonnie/chicago-airbnb-open-data)

My method was to build a model to predict the number of bookings in a year given other relevant features in the data. Using this model that has the nightly price as a feature, iteratively testing prices to see which has the highest predicted revenue (yearly bookings * nightly price) would be possible. Ulitmately, the model was unable to capture the relationship between the price and bookings well enough to drop predicted bookings low enough to get a nuanced listing price. Best listing prices suggested by the algorithm were hugely overestimated. 

With the right data and model this method should be useful and the method with this dataset could be aided by some synthetic data indicating to the model that large prices for certain listing types equate to no bookings

# General pipeline
Step one was preparing the data for modeling. This is done in the [data_preparation.ipynb](https://github.com/Feebami/Chicago_Airbnb_MLPipeline/blob/main/data_preparation.ipynb) file. Missing values, outliers, feature transformations and feature engineering were all included. Then every data transformation was wrapped into a function and saved to the [utils.py](https://github.com/Feebami/Chicago_Airbnb_MLPipeline/blob/main/utils.py) file for later use. 

Next I used the prepared data to build a model. Models were limitied to scale-invariant algorithms to limit the overhead needed later for the price optimization step. The ML model building is held in the [model_training.ipynb](https://github.com/Feebami/Chicago_Airbnb_MLPipeline/blob/main/model_training.ipynb) file. This file also includes a fuction that attempts to optimize the price by iteratively testing different price inputs in the model. This function was also saved in the [utils.py](https://github.com/Feebami/Chicago_Airbnb_MLPipeline/blob/main/utils.py) file.

The final step was testing the price optimization on 100 mock live samples that were split from the training dataset in the beginning of data preparation. The results can be seen in the [final_test.ipynb](https://github.com/Feebami/Chicago_Airbnb_MLPipeline/blob/main/final_test.ipynb) file.
