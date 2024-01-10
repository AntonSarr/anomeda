# Introduction to Anomeda

"Why has the number of visits of our website decreased this week? Who are the users who caused that?" - **anomeda** will help to answer such questions quickly and easily using Python with **pandas** package installed.

You are welcome to visit [GitHub repo](https://github.com/AntonSarr/anomeda) and [Documentation](https://anomeda.readthedocs.io/en/latest/) page of the project.

**anomeda** is a Python package developed for Data Analysts, Data Scientists, BI Analysts and other specialists dealing with Data Analysis. It helps to identify important metric changes and quickly find clusters in data which changed the trend of the metric or caused the anomaly. You just need to pass a `pandas.DataFrame` object to anomeda to start analyzing it.

Let's say your data contains two sets of columns. Firstly, values you want to track in an aggregated or non-aggregated way, like "visit a website", "purchase" events or rows with aggregated "number of errors per minute". In anomeda, we call it a ***metric***. Secondly, corresponding characteristics, like "country", "device name", "goods category" or similar (we call it ***measures***). If you know that the metric changed its values and you need to know the reason, with **anomeda** you can run a few lines of code:

```python
import anomeda

normal_period = anomeda.DataFrame(
    data1, # pandas.DataFrame
    measures_names=['dummy_measure', 'dummy_continuous_measure'], # columns represending measures
    measures_types={
        'categorical': ['dummy_measure'], 
        'continuous': ['dummy_continuous_measure']
    },
    index_name='date',
    metric_name='metric',
    agg_func='sum' # function that is used to aggregate metric if more than 1 metric value will be found for a particular set of measure values
)

abnormal_period = anomeda.DataFrame(
    data2, # pandas.DataFrame
    measures_names=['dummy_measure', 'dummy_numeric_measure'], # columns represending measures
    measures_types={
        'categorical': ['dummy_measure'], 
        'continuous': ['dummy_numeric_measure']
    },
    index_name='date',
    metric_name='metric',
    agg_func='sum' # function that is used to aggregate metric if more than 1 metric value will be found for a particular set of measure values
)

anomeda.explain_values_difference(
    normal_period,
    abnormal_period,
    measures_to_iterate='combinations'
)
```

Output will look like a pandas.DataFrame object containig **average values**, **trends** of the metric in every single cluster, the description of **how they differ** and **how the differences contribute** to the overall metric. It will let you easily find the culprits of anomalies and take actions rapidly.

> As you might have noticed, anomeda is capable of processing **continous measures**, which means you still can them as measures! It is possible because of automatical discretization mechanism embedded in the package. However, you may pass your own discrete values of continous features.

Visit the [Documentation](https://anomeda.readthedocs.io/en/latest/) page of the project to know more.

# Other use cases

**anomeda** also lets do the following:
- Detect metric's anomalies and find the clusters of rows, sharing common features, which caused that
- Find clusters of rows sharing common features which make the metric increase or decrease (clusters where metric's values have a positive or a negative trend)

See the [Usage](user_guide.md) page of the documentation for the details.

# Installing

The [GitHub repo](https://github.com/AntonSarr/anomeda) contains the source distribution and built distribution files in *dist* folder.

You must have *pandas*, *numpy*, *sklearn*, *scipy* installed. 

# Contribution

You are very welcome to contribute to the project!

# Contacts

If you have any questions related to **anomeda** project, feel free reaching out to the author:

Anton Saroka, Data Scientist

anton.soroka.1313@gmail.com


