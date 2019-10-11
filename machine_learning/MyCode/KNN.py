import pandas as pd
import sklearn


file_path = 'F:/ZLW/ZLW_lab/machine_learning/KNN/KNN/listings.csv'
features = ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'price', 'minimum_nights', 'maximum_nights', 'number_of_reviews']
dc_listings = pd.read_csv(file_path)
dc_listings = dc_listings[features]
#print(dc_listings)
print(dc_listings.head)
