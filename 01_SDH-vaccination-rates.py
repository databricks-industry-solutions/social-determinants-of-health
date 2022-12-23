# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/social-determinants-of-health.git. For more information about this accelerator, visit https://www.databricks.com/solutions/accelerators/social-determinants-of-health.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a SDOH Database based on publically available data
# MAGIC There are a number of datasets from various sources we used to explore their affects on vacination rates of COVID-19. 
# MAGIC The datasets include population, income, education, poverty, health traits and more at a county level
# MAGIC Many of the files needed some level of data cleansing before they can be used. We applied these transformations and created the silver layer and share cleansed tables using deltashare.
# MAGIC 
# MAGIC Having access to the data, now we simply create a database based on these cleaned datasets and start with our analysis.
# MAGIC 
# MAGIC As an example of the pre-processing steps, to create the `Income` table, we first clean up the FIPS field as part initial importing:
# MAGIC ```
# MAGIC from pyspark.sql.functions import regexp_replace, trim
# MAGIC from pyspark.sql.types import  StructType   
# MAGIC import json
# MAGIC new_schema = StructType.fromJson(json.loads(incomeSchema))
# MAGIC 
# MAGIC dfIncome = spark.read.csv(storageBase + "/all_counties_income.csv", header=True, schema=new_schema) \
# MAGIC   .filter("Description ='Per capita personal income (dollars) 2/'") \
# MAGIC   .withColumn("GeoFIPS",regexp_replace('GeoFIPS','"',"")) \
# MAGIC   .withColumn("GeoFIPS",regexp_replace('GeoFIPS',' ',""))
# MAGIC ```
# MAGIC 
# MAGIC Similarly we created education, health, pverty and vaccination tables in the bronze layer.

# COMMAND ----------

# MAGIC %sh pip install delta-sharing

# COMMAND ----------

# MAGIC %md
# MAGIC To get access to the data, we first need to specify the location where the [deltashare credentials file](https://docs.databricks.com/data-sharing/delta-sharing/recipient.html#download-the-credential-file) is stored.

# COMMAND ----------

# DBTITLE 1,retrieve share credentials file 
import delta_sharing
dbutils.fs.cp('s3://hls-eng-data-public/delta_share/rearc_hls_data.share','/tmp/')
share_file_path = "/tmp/rearc_hls_data.share"
client = delta_sharing.SharingClient(f"/dbfs{share_file_path}")
shared_tables = client.list_all_tables()

# COMMAND ----------

# DBTITLE 1,setup deltashare
dataset_urls = {
  "bronze_income":f"{share_file_path}#rearc_databricks_hls_share.hls_sdoh.bronze_income",
  "silver_poverty":f"{share_file_path}#rearc_databricks_hls_share.hls_sdoh.poverty_county",
  "silver_education":f"{share_file_path}#rearc_databricks_hls_share.hls_sdoh.education_county",
  "silver_health_stats":f"{share_file_path}#rearc_databricks_hls_share.hls_sdoh.health_stats_county",
  "silver_race":f"{share_file_path}#rearc_databricks_hls_share.hls_sdoh.race_county",
  "silver_vaccinations":f"{share_file_path}#rearc_databricks_hls_share.hls_covid19_usa.vaccinations_county_utd",
}

# Reinitiate schema
spark.sql("DROP SCHEMA IF EXISTS sdoh CASCADE")
spark.sql("CREATE SCHEMA sdoh")

# Add tables
for ds, url in dataset_urls.items():
  spark.sql(f"CREATE TABLE IF NOT EXISTS sdoh.{ds} USING deltaSharing LOCATION '{url}'")

# COMMAND ----------

# MAGIC %sql
# MAGIC use sdoh

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables

# COMMAND ----------

# DBTITLE 1,Education Status
# MAGIC %sql
# MAGIC select * from silver_education limit 20

# COMMAND ----------

# DBTITLE 1,Health Stats
# MAGIC %sql
# MAGIC select * from silver_health_stats limit 20

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the correlation between different health stats. To do so, we use `Correlation` function from `pyspark.ml.stat` package to calculate pairwise pearson correlation coefficients among features of interest (such as smoking, obesity etc). This approach enables us to leverage distributed processing to accelerate computation. 

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import plotly.express as px

# create the dataframe of selected features
_df=sql('select SmokingPct, ObesityPct, HeartDiseasePct, CancerPct, NoHealthInsPct, AsthmaPct from silver_health_stats')
# convert columns of the dataframe to vectors 
vecAssembler = VectorAssembler(outputCol="features")
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=_df.columns, outputCol=vector_col)
df_vector = assembler.transform(_df).select(vector_col)

# Calculate the correlation matrix
corr_matrix = Correlation.corr(df_vector, vector_col).select('pearson(corr_features)').collect()[0]['pearson(corr_features)'].toArray()

# COMMAND ----------

# DBTITLE 1,Pairwise correlation among different health stats 
col_names=_df.columns
_pdf=pd.DataFrame(corr_matrix,columns=col_names,index=col_names)
px.imshow(_pdf,text_auto=True)

# COMMAND ----------

# MAGIC %md
# MAGIC From the matrix above we see a very significant correlation between rate of smoking and other risk factors such as obesity, heart disease and asthma. Perhaps a more rigorous analysis would require taking into account estimation errors due to the sizes of counties.

# COMMAND ----------

# DBTITLE 1,vaccinations
# MAGIC %sql
# MAGIC select * from silver_vaccinations limit 20

# COMMAND ----------

# DBTITLE 1,Vaccination rates for 12 and older across counties 
from urllib.request import urlopen
import json
import pandas as pd
import plotly.express as px
import numpy as np

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
_pdf = sql('select fips,avg(Series_Complete_12PlusPop_Pct) as vaccination_rate  from silver_vaccinations group by fips').toPandas()
fig = px.choropleth(_pdf, geojson=counties, locations='fips', color='vaccination_rate',
                           color_continuous_scale="Viridis",
                           scope="usa",
                           labels={'vaccination_rate':'vaccination_rate'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Build the ML model
# MAGIC Now we proceed to create a dataset for downstream analysis using ML. 
# MAGIC The target value to predict is vaccination rate for people 12 years and older which is `Series_Complete_12PlusPop_Pct`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create training data
# MAGIC Here, we also need population density data. In this case, we directly read the `csv` files and register the resulting dataset as a view.

# COMMAND ----------

# DBTITLE 1,adding population density
spark.read.csv('wasb://data@sdohworkshop.blob.core.windows.net/sdoh/Population_Density_By_County.csv', header=True, inferSchema=True).createOrReplaceTempView('population_density')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from population_density limit 20

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have all the data, we create the dataset needed for our downstream analysis.

# COMMAND ----------

# MAGIC %sql
# MAGIC use sdoh;
# MAGIC create or replace temp view vaccine_data_pct
# MAGIC as
# MAGIC select v.fips, Recip_County, Recip_State, ifnull(Series_Complete_12PlusPop_Pct, Series_Complete_Pop_Pct) Series_Complete_12PlusPop_Pct, pd.Density_per_square_mile_of_land_area population_density, r.County_Population, round((r.County_Population - r.White_population) / r.County_Population,3) * 100 Minoirity_Population_Pct, 
# MAGIC i.`2019` income, p.All_Ages_in_Poverty_Percent, round(e.25PlusHS / r.County_Population,2) * 100 25PlusHSPct, round(e.25PlusAssociate / r.County_Population,2) * 100 25PlusAssociatePct, h.SmokingPct, h.ObesityPct, h.HeartDiseasePct, h.CancerPct, h.NoHealthInsPct,h.AsthmaPct
# MAGIC from silver_race r join sdoh.silver_vaccinations v on(r.fips = v.fips)
# MAGIC join bronze_income i on(i.geofips = v.fips)
# MAGIC join silver_poverty p on (p.fips = v.fips)
# MAGIC join silver_education e on (e.fips = v.fips)
# MAGIC join silver_health_stats h on(h.locationid = v.fips)
# MAGIC join population_density pd on (pd.GCT_STUBtarget_geo_id2 = v.fips)

# COMMAND ----------

# DBTITLE 1,View the dataset
# MAGIC %sql
# MAGIC select * from vaccine_data_pct limit 20

# COMMAND ----------

# MAGIC %md
# MAGIC Now we create a pandas data frame which will be used by the ML framework

# COMMAND ----------

parsed_pd = spark.table("vaccine_data_pct").toPandas().dropna(subset=['Series_Complete_12PlusPop_Pct'])
parsed_pd.set_index("fips")

X = parsed_pd.drop(["Series_Complete_12PlusPop_Pct", "fips","Recip_County","Recip_State","County_Population"], axis=1)
y = parsed_pd["Series_Complete_12PlusPop_Pct"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression with XG Boost
# MAGIC Now, we use XG Boost to train a linear regression model and use MLFlow autolog to track model performance

# COMMAND ----------

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
import numpy as np
import pandas as pd
import shap
import mlflow

mlflow.autolog()

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

max_depth=3
learning_rate=.1
reg_alpha=.1

# We use n_estimators=500 for to accelerate the training, in practice you may want to use higher values for more accurare results
xgb_regressor = XGBRegressor(objective='reg:squarederror', max_depth=max_depth, learning_rate=learning_rate, reg_alpha=reg_alpha, n_estimators=500, importance_type='total_gain', random_state=0)

xgb_model = xgb_regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)],\
                              eval_metric='rmse', early_stopping_rounds=25)

n_estimators = len(xgb_model.evals_result()['validation_0']['rmse'])
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_pred, y_test)
rmse = sqrt(mean_squared_error(y_pred, y_test))
displayHTML(f"mae =<b>{mae}</b> and rsme: <b>{rmse}</b>")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Explainability and Feature Importance
# MAGIC Now, to explore the impact of SDOH factors affecting vacination rates, we use SHAP values. 
# MAGIC For an introduction to SHAP see [this blog](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30#:~:text=In%20a%20nutshell%2C%20SHAP%20values,answer%20the%20%E2%80%9Chow%20much%E2%80%9D').

# COMMAND ----------

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC Add FIPS Codes to SHAP Values

# COMMAND ----------

df = pd.DataFrame(shap_values, columns=X.columns) 
df['fips'] =spark.sql("select fips from vaccine_data_pct order by fips").toPandas()
dfShaps = spark.createDataFrame(df)
dfShaps.createOrReplaceTempView("shap")

# COMMAND ----------

sql('select * from shap limit 20').display()

# COMMAND ----------

# MAGIC %md
# MAGIC Pivot the columns back to rows to make reporting easier and add to the database

# COMMAND ----------

usa_model_county_vaccine_shap_df = sql("""
select fips, stack(12,'Minority_Population_Pct',Minoirity_Population_Pct,'income', income, '25PlusHSPct', 25PlusHSPct,
'All_Ages_in_Poverty_Percent',All_Ages_in_Poverty_Percent,'population_density', population_density, '25PlusAssociatePct', 25PlusAssociatePct, 
'SmokingPct',SmokingPct, 
'ObesityPct', ObesityPct, 
'HeartDiseasePct', HeartDiseasePct, 
'CancerPct', CancerPct, 
'NoHealthInsPct', NoHealthInsPct,
'AsthmaPct', AsthmaPct
)  
as (factor, value)
from shap
""").limit(50)
display(usa_model_county_vaccine_shap_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We note that the values across the USA income, smoking and health insurance are the leading factors of vaccation rates:

# COMMAND ----------

# DBTITLE 1,Top SDH effecting vaccination rates 
mean_abs_shap = np.absolute(shap_values).mean(axis=0).tolist()
_pdf=pd.DataFrame(sorted(list(zip(mean_abs_shap, X.columns)), reverse=True)[:6],columns=["Mean(SHAP)", "Column"])
px.bar(_pdf,x='Column',y='Mean(SHAP)')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC In this notebook we quickly and easily:
# MAGIC   * Brought in pre-processed data from a varitey of datasources using deltasharing.
# MAGIC   * We then created a dataset of all SDH alogside vacciantion rates for each county in the US
# MAGIC   * Trained a regression model to predict vaccination rates based on SDH and county-level geographic information such as population density and size
# MAGIC   * We used SHAP to explain the impact of each SDH factor on vaccination rates
