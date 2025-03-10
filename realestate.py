import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import plotly.express as px 


##########################FUNCTIONS############

#Outlier Detection 
def outliers(data,col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1 

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    return outliers

####### Data Cleaning and Summary Statistics #########

data = pd.read_csv('housedata.csv')
print("Original Data\n", data.head())
print(data.size)

new_data = data.dropna()
print("Removed NaN Data\n", new_data.head())

price_outlier = outliers(new_data,"Price")
bond_outlier = outliers(new_data,"Bond")
print(f"Price Outliers: {len(price_outlier)}")
print(f"Bond Outliers: {len(bond_outlier)}")

#Visualization of Price and Bond
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.boxplot(y=new_data["Price"], color="Yellow")
plt.title("Boxplot of Price")

plt.subplot(1,2,2)
sns.boxplot(y=new_data["Bond"], color="Red")
plt.title("Boxplot of Bond")

plt.savefig("Boxplot of Price & Bond.png")

#Summary Statistics 
print(f"Description of Bond:\n", new_data["Bond"].describe())
print(f"Description of Price:\n", new_data["Price"].describe())

############## Bivariate Analysis #################

#Correlation Matrix
corr_matrix = new_data.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("Correlation Heatmap.png")


plt.figure(figsize=(15,5))
# Price vs Bedrooms
plt.subplot(1,3,1)
sns.boxplot(x=new_data["Bed"], y=new_data["Price"], color="Green")
plt.title("Bedrooms vs Price")

# Price vs Bath
plt.subplot(1,3,2)
sns.boxplot(x=new_data["Bath"], y=new_data["Price"], color="Yellow")
plt.title("Bath vs Price")

# Price vs Parking
plt.subplot(1,3,3)
sns.boxplot(x=new_data["Parking"], y=new_data["Price"], color="Orange")
plt.title("Parking vs Price")

plt.savefig("Utilities vs Price.png")

#Price vs Bond
plt.figure(figsize=(8,6))
sns.scatterplot(x=new_data["Bond"], y=new_data["Price"])
plt.title("Bond vs Price")
plt.xlabel("Bond")
plt.ylabel("Price")
plt.savefig("Bond vs Price.png")


##########GEO SPATIAL ANALYSIS ####################

# Define center of Victoria (Melbourne)
centre = [-37.8136, 144.9631]
victoria_map = folium.Map(location=[-37.8136, 144.9631], zoom_start=20)

# Create a marker cluster for better visualization
marker_cluster = MarkerCluster().add_to(victoria_map)

# Add markers for properties
for _, row in new_data.dropna(subset=["Lat", "Long"]).iterrows():
    folium.Marker(
        location=[row["Lat"], row["Long"]],
        popup=f"Price: ${row['Price']:.2f}\nType: {row['Type']}",
        tooltip=row["Address"]
    ).add_to(marker_cluster)
# Show the map
victoria_map.save("House Locations.html")

# Static Map
fig = px.density_map(new_data, lat="Lat", lon="Long", z="Price", radius=10,
                        center=dict(lat=-37.8136, lon=144.9631), zoom=11, map_style="open-street-map")

fig.show()











