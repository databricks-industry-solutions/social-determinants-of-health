# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/social-determinants-of-health.git. For more information about this accelerator, visit https://www.databricks.com/solutions/accelerators/social-determinants-of-health.

# COMMAND ----------

# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC <img src=" https://hls-eng-data-public.s3.amazonaws.com/img/Databricks_HLS.png" width=700> 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Social Determinants of Health (SDH)
# MAGIC Many aspects of a person's life style and status can impact health outcomes. Multiple studies suggest that Social Determinents of Health (SDH) account for between 30-55% of health outcomes. 
# MAGIC [The WHO defines SDH as]((https://www.who.int/health-topics/social-determinants-of-health#tab=tab_1):
# MAGIC >.. the non-medical factors that influence health outcomes. They are the conditions in which people are born, <br>
# MAGIC > grow, work, live, and age, and the wider set of forces and systems shaping the conditions of daily life. <br>
# MAGIC > These forces and systems include economic policies and systems, development agendas, social norms, social <br>
# MAGIC > policies and political systems.
# MAGIC 
# MAGIC <img src='https://www.uclahealth.org/sites/default/files/styles/max_width_012000_480/public/images/SOCIAL%20DETERMINANTS%20OF%20HEALTH%20GRAPHIC.png' width=25%/>
# MAGIC 
# MAGIC 
# MAGIC Correlation between SDH and health outcomes is very clear: the lower the socioeconomic position, the worse the health, which in turn creates a negative feedback loop (poor health resulting in poor socioeconomic status) which results in widening the gap even more. 
# MAGIC 
# MAGIC There are many public sources of SDH data with different levels of granularity (country level, state/province, county, or zipcode/postal code level) that can be used in analysis of the impact of SDH on health outcomes. One of the main challenges for data analysis is finding the right data source and data cleaning. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Access SDH data via Delta Share
# MAGIC In this solution accelerator, we use pre-processed and cleansed tables that have been made available by [Rearc](rearc.io) via [delta sharing protocol](https://www.databricks.com/blog/2021/05/26/introducing-delta-sharing-an-open-protocol-for-secure-data-sharing.html). We explore income, healthcare, education and other aspects affect counties vaccinations rates for COVID-19. 
# MAGIC 
# MAGIC Delta sharing allows us to offload the "bronze to silver" data prep step to a data provider or internal data team, beginning our analysis with "silver" or "gold" data.
# MAGIC Using these data, we train a machine learning model to predict vaccination rates based on different SDH features and then use [SHAP](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) to offer insights into different factors impacting vaccination rates.
# MAGIC 
# MAGIC ![](https://databricks.com/wp-content/uploads/2022/03/delta-lake-medallion-architecture-2.jpeg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## License
# MAGIC Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the  the [Databricks License] (https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library License|Library License URL|Library Source URL| 
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
# MAGIC |Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
# MAGIC |Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
# MAGIC |SHAP| MIT | https://github.com/slundberg/shap/blob/master/LICENSE | https://github.com/slundberg/shap/|
# MAGIC |Plotly Express | MIT License | https://github.com/plotly/plotly_express/blob/master/LICENSE.txt | https://github.com/plotly/plotly_express/|
# MAGIC 
# MAGIC 
# MAGIC |Author|
# MAGIC |-|
# MAGIC |Databricks Inc.|
