# Chicago_Airbnb_MLPipeline
 An ML pipeline to determine rental unit nightly charges showcased in Jupyter Notebooks

The goal of this project was to maximize the revenue of Airbnb listings given the dataset found [here](https://www.kaggle.com/datasets/jinbonnie/chicago-airbnb-open-data)

The approach leveraged machine learning to predict yearly bookings based on relevant features of the dataset. By incorporating nightly price as a key variable, the model aimed to identify the price point that maximized predicted yearly revenue, calculated as:

`Revenue = Predicted Yearly Bookings × Nightly Price`

To achieve this goal, a predictive model was developed to forecast the number of bookings in a year based on features such as location, amenities, and nightly price. Using the trained model, the pipeline tested different nightly price points to identify the price that yielded the highest predicted revenue. Despite iterative testing, the model struggled to capture the relationship between nightly price and bookings. Prices suggested by the algorithm were significantly overestimated and did not align with realistic booking patterns. 

The approach was hindered by the inability of the predictive model to accurately reflect how nightly prices influence bookings. The relationship between price and demand proved too complex for the current dataset and model. With these limitations, the pipeline struggled to produce reliable nightly price recommendations. However, the proposed methodology remains promising. With enhancements such as improved dataset quality, synthetic data augmentation, or more advanced models, this technique could become a powerful tool for optimizing Airbnb rental pricing.

# General pipeline
Step one was preparing the data for modeling. This is done in the [data_preparation.ipynb](https://github.com/Feebami/Chicago_Airbnb_MLPipeline/blob/main/data_preparation.ipynb) file. Missing values, outliers, feature transformations and feature engineering were all included. Every data transformation was wrapped into a function and saved to the [utils.py](https://github.com/Feebami/Chicago_Airbnb_MLPipeline/blob/main/utils.py) file for later use. 

Next, prepared data was used to build a model, with algorithms tired limitied to scale-invariant algorithms. This aided by limiting overhead required for the iterative price optimization step. The ML model building code is held in the [model_training.ipynb](https://github.com/Feebami/Chicago_Airbnb_MLPipeline/blob/main/model_training.ipynb) file. This file also includes a function to optimize the price by iteratively testing different price values in the trained model. This function was saved in the [utils.py](https://github.com/Feebami/Chicago_Airbnb_MLPipeline/blob/main/utils.py) file for pipeline evaluation and deployment.

The final step was testing the price optimization on 100 mock live samples that were split from the training dataset before any data preparation. The results can be seen in the [final_test.ipynb](https://github.com/Feebami/Chicago_Airbnb_MLPipeline/blob/main/final_test.ipynb) file.
