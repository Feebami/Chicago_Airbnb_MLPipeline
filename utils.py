# Imports
import joblib
import numpy as np
import pandas as pd

pipe = joblib.load('model.joblib')

# Function to prepare data for modeling
def prepare_data(x):
    df = x.copy(deep=True)
    df.drop(columns='neighbourhood_group', inplace=True)
    df['last_review'] = pd.to_datetime(df.last_review)
    df['log_days_since_last_review'] = (pd.to_datetime('2021-01-01') - df.last_review).dt.days + 1
    df['log_days_since_last_review'] = df.log_days_since_last_review.fillna(0)
    df['log_days_since_last_review'] = np.log10(df.log_days_since_last_review + 0.1)
    df['reviews_per_month'] = df.reviews_per_month.fillna(0)
    df['log_reviews_per_month'] = np.log10(df.reviews_per_month + 0.1)
    df['log_number_of_reviews'] = np.log10(df.number_of_reviews + 0.1)
    df = df[df.price <= df.price.quantile(0.999)]
    df = df[df.price >= df.price.quantile(0.001)]
    df['log_price'] = np.log10(df.price)
    df = df[df.minimum_nights <= 180]
    df['log_minimum_nights'] = np.log10(df.minimum_nights)
    df['log_host_listings_count'] = np.log10(df.calculated_host_listings_count)
    keephoods = ['West Town', 'Near North Side', 'Lake View', 'Logan Square', 'Loop',
        'Near West Side', 'Lincoln Park', 'Lower West Side', 'Edgewater',
        'Uptown', 'Irving Park', 'Avondale', 'Near South Side', 'North Center',
        'Rogers Park', 'Bridgeport', 'Grand Boulevard', 'East Garfield Park',
        'Hyde Park', 'South Shore', 'Lincoln Square', 'Woodlawn',
        'Portage Park', 'West Ridge', 'Armour Square', 'Albany Park',
        'Humboldt Park', 'Douglas', 'Austin']
    df.loc[df.neighbourhood.isin(keephoods) == False, 'neighbourhood'] = 'infrequent'
    df['neighbourhood'] = df['neighbourhood'].astype('category')
    df['room_type'] = df['room_type'].astype('category')
    df['log_nights_booked'] = 365 - df.availability_365
    df['log_nights_booked'] = np.log10(df.log_nights_booked + 0.1)
    df['minimum_booking_price'] = df.log_price * df.log_minimum_nights
    df['host_listings_minimum_nights'] = df.log_host_listings_count * df.log_minimum_nights
    df['price^2'] = df.price ** 2
    df['price^3'] = df.price ** 3
    df['price^4'] = df.price ** 4
    df.drop(columns=['id', 'name', 'host_id', 'host_name', 'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365'], inplace=True)
    return df

# Function to add best price, bookings, and revenue to a DataFrame
def optimize_income(x):
    x = x.copy()
    # Log price is standardized, so this is + or - 1.5 standard deviations
    price_min = 20
    price_max = 1000
    best_revenue = 0
    best_price = 0
    best_bookings = 0
    for price in np.linspace(price_min, price_max, 100):
        x['price'] = price
        x['log_price'] = np.log(price)
        x['price^2'] = price ** 2
        x['price^3'] = price ** 3
        x['price^4'] = price ** 4
        predicted_bookings = 10 ** pipe.predict(x.to_frame().T)[0]
        projected_revenue = price * predicted_bookings
        if projected_revenue > best_revenue:
            best_revenue = projected_revenue
            best_price = price
            best_bookings = predicted_bookings
    return best_revenue, best_price, best_bookings