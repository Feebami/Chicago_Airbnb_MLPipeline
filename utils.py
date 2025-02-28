# Imports
import joblib
import numpy as np
import pandas as pd

pipe = joblib.load('model.joblib')

# Function to prepare data for modeling
def prepare_data(x):
    temp = x.copy()
    temp.drop(columns='neighbourhood_group', inplace=True)
    temp['last_review'] = pd.to_datetime(temp.last_review)
    temp['log_days_since_last_review'] = (pd.to_datetime('2021-01-01') - temp.last_review).dt.days
    temp['log_days_since_last_review'] = temp.log_days_since_last_review.fillna(0)
    temp['log_days_since_last_review'] = np.log10(temp.log_days_since_last_review + 0.1)
    temp['reviews_per_month'] = temp.reviews_per_month.fillna(0)
    temp['log_reviews_per_month'] = np.log10(temp.reviews_per_month + 0.1)
    temp['log_number_of_reviews'] = np.log10(temp.number_of_reviews + 0.1)
    temp = temp[temp.price <= 6871] # Hardcoded values from training data
    temp = temp[temp.price >= 14]
    temp['log_price'] = np.log10(temp.price)
    temp = temp[temp.minimum_nights <= 180]
    temp['log_minimum_nights'] = np.log10(temp.minimum_nights)
    temp['log_host_listings_count'] = np.log10(temp.calculated_host_listings_count)
    keephoods = ['West Town', 'Near North Side', 'Lake View', 'Logan Square', 'Loop',
        'Near West Side', 'Lincoln Park', 'Lower West Side', 'Edgewater',
        'Uptown', 'Irving Park', 'Avondale', 'Near South Side', 'North Center',
        'Rogers Park', 'Bridgeport', 'Grand Boulevard', 'East Garfield Park',
        'Hyde Park', 'South Shore', 'Lincoln Square', 'Woodlawn',
        'Portage Park', 'West Ridge', 'Armour Square', 'Albany Park',
        'Humboldt Park', 'Douglas', 'Austin']
    temp.loc[temp.neighbourhood.isin(keephoods) == False, 'neighbourhood'] = 'infrequent'
    temp['neighbourhood'] = temp['neighbourhood'].astype('category')
    temp['room_type'] = temp['room_type'].astype('category')
    temp['log_nights_booked'] = 365 - temp.availability_365
    temp['log_nights_booked'] = np.log10(temp.log_nights_booked + 0.1)
    temp['host_listings_minimum_nights'] = temp.log_host_listings_count * temp.log_minimum_nights
    temp['price^2'] = temp.price ** 2
    temp['price^3'] = temp.price ** 3
    temp['price^4'] = temp.price ** 4
    temp.drop(columns=['id', 'name', 'host_id', 'host_name', 'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365'], inplace=True)
    return temp

# Function to add best price, bookings, and revenue to a DataFrame
def optimize_income(x):
    x = x.copy()
    # This minimum and max price seem reasonable to capture most of the data
    price_min = 20
    price_max = 1000
    best_revenue = 0
    best_price = 0
    best_bookings = 0
    # Loop through prices and calculate revenue
    # Use log scale for price because price differences are less impactful at higher prices
    for price in np.linspace(price_min, price_max, 100):
        x['price'] = price
        x['log_price'] = np.log10(price)
        x['price^2'] = price ** 2
        x['price^3'] = price ** 3
        x['price^4'] = price ** 4
        predicted_bookings = 10 ** pipe.predict(x.to_frame().T)
        projected_revenue = price * predicted_bookings
        if projected_revenue > best_revenue:
            best_revenue = projected_revenue
            best_price = price
            best_bookings = predicted_bookings
    return best_revenue, best_price, best_bookings