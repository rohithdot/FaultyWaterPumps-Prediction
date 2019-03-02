import pandas as pd
from geopy.geocoders import Nominatim
import sys
import requests
import matplotlib.pyplot as plot
import numpy as np
from math import log
import time
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
import seaborn as sns

get_ipython().magic('matplotlib inline')
d_val = pd.read_csv("training_set_values.csv")
d_lab = pd.read_csv("training_set_labels.csv")

for latlon in ["latitude","longitude"]:
    d_val[latlon] = d_val[latlon].map(lambda y: round(y, 7))
    d_val[latlon] = d_val[latlon].replace(to_replace=-0.0, value=0)
    
#get missing longitudes from geopy
for lga in d_val[(d_val.latitude == 0) | (d_val.longitude == 0)]["lga"].unique():
    time.sleep(1)
    gloc = Nominatim(user_agent="mlProject")
    loc = gloc.geocode(lga)
    d_val.loc[d_val.lga == lga, "latitude"] =loc.latitude
    d_val.loc[d_val.lga == lga, "longitude"] = loc.longitude

d_val["date_recorded"]=pd.DatetimeIndex(d_val["date_recorded"]).year


pd.crosstab(data_values.region, data_labels.status_group, normalize="index").plot.bar(figsize=(25,10));

pd.crosstab(data_values.quality_group, data_labels.status_group, normalize="index").plot.bar(figsize=(25,10));

pd.crosstab(data_values.quantity, data_labels.status_group, normalize="index").plot.bar(figsize=(25,10));

pd.crosstab(data_values.source, data_labels.status_group, normalize="index").plot.bar(figsize=(25,10));

plot.figure(figsize=(15,15))
plot.matshow(d_val.corr(), fignum=1, cmap = 'hot')
plot.xticks(rotation=90)
plot.xticks(range(len(d_val.columns)), d_val.columns)
plot.yticks(range(len(d_val.columns)), d_val.columns)
plot.colorbar()
plot.show()


removeitems = ["funder","installer","gps_height","wpt_name","num_private","public_meeting","recorded_by","scheme_name",
           "permit","extraction_type_class","management_group",
           "payment_type","quantity_group","waterpoint_type_group","source_class"]
d_val.drop(removeitems, axis=1, inplace=True)


d_val.replace('', np.nan, inplace=True)
d_val.dropna()


label_encoder = LabelEncoder()

d_val["basin"] = d_val["basin"].astype('category')
d_val["basin"] = label_encoder.fit_transform(d_val["basin"].fillna(method='pad'))

d_val["district_code"] = d_val["district_code"].map(lambda x: str(x))

d_val["subvillage"] = d_val["subvillage"].astype('category')
d_val["subvillage"] = label_encoder.fit_transform(d_val["subvillage"].fillna(method='pad'))

d_val["region"] = d_val["region"].astype('category')
d_val["region"] = label_encoder.fit_transform(d_val["region"].fillna(method='pad'))
d_val["region_code"] = d_val["region_code"].map(lambda x: str(x))

d_val["lga"] = d_val["lga"].astype('category')
d_val["lga"] = label_encoder.fit_transform(d_val["lga"].fillna(method='pad'))

d_val["population"] = d_val["population"].apply(lambda x: log(x) if x > 0 else 0)

d_val["ward"] = d_val["ward"].astype('category')
d_val["ward"] = label_encoder.fit_transform(d_val["ward"].fillna(method='pad'))


d_val["scheme_management"] = d_val["scheme_management"].astype('category')
d_val["scheme_management"] = label_encoder.fit_transform(d_val["scheme_management"].fillna(method='pad'))

d_val["extraction_type"] = d_val["extraction_type"].astype('category')
d_val["extraction_type"] = label_encoder.fit_transform(d_val["extraction_type"].fillna(method='pad'))


d_val["extraction_type_group"] = d_val["extraction_type_group"].astype('category')
d_val["extraction_type_group"] = label_encoder.fit_transform(d_val["extraction_type_group"].fillna(method='pad'))

d_val["management"] = d_val["management"].astype('category')
d_val["management"] = label_encoder.fit_transform(d_val["management"].fillna(method='pad'))

d_val["amount_tsh"] = d_val["amount_tsh"].apply(lambda x: log(round(x)) if round(x) > 0 else 0)

d_val["payment"] = d_val["payment"].astype('category')
d_val["payment"] = label_encoder.fit_transform(d_val["payment"].fillna(method='pad'))

d_val["water_quality"] = d_val["water_quality"].astype('category')
d_val["water_quality"] = label_encoder.fit_transform(d_val["water_quality"].fillna(method='pad'))

d_val["quality_group"] = d_val["quality_group"].astype('category')
d_val["quality_group"] = label_encoder.fit_transform(d_val["quality_group"].fillna(method='pad'))

d_val["quantity"] = d_val["quantity"].astype('category')
d_val["quantity"] = label_encoder.fit_transform(d_val["quantity"].fillna(method='pad'))

d_val["source"] = d_val["source"].astype('category')
d_val["source"] = label_encoder.fit_transform(d_val["source"].fillna(method='pad'))

d_val["source_type"] = d_val["source_type"].astype('category')
d_val["source_type"] = label_encoder.fit_transform(d_val["source_type"].fillna(method='pad'))

d_val["waterpoint_type"] = d_val["waterpoint_type"].astype('category')
d_val["waterpoint_type"] = label_encoder.fit_transform(d_val["waterpoint_type"].fillna(method='pad'))


d_val["amount_tsh"] = RobustScaler().fit_transform(d_val[["amount_tsh"]])
d_val["longitude"] = RobustScaler().fit_transform(d_val[["longitude"]])
d_val["latitude"] = RobustScaler().fit_transform(d_val[["latitude"]])

d_val = pd.merge(d_val, d_lab, on="id")
d_val.shape


plot.figure(figsize=(10,10))
plot.matshow(d_val.corr(), fignum=1, cmap = 'hot')
plot.xticks(rotation=90)
plot.xticks(range(len(d_val.columns)), d_val.columns)
plot.yticks(range(len(d_val.columns)), d_val.columns)
plot.colorbar()
plot.show()


removeitems = ["scheme_management","quantity","source_type"]
d_val.drop(removeitems, axis=1, inplace=True)

plot.figure(figsize=(15,15))
plot.matshow(d_val.corr(), fignum=1, cmap = 'hot')
plot.xticks(rotation=90)
plot.xticks(range(len(d_val.columns)), d_val.columns)
plot.yticks(range(len(d_val.columns)), d_val.columns)
plot.colorbar()
plot.show()

get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(15,15))
sns.heatmap(d_val.corr(),annot=True)


d_val.to_csv("dataset_processed.csv", index = False)

