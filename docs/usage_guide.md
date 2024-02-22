# Usage Guide

## What is anomeda.DataFrame

`anomeda.DataFrame` is class used to store the time-series data and its metadata. It inherits `pandas.DataFrame`. One of the implications of that fact is that `anomeda.DataFrame.__init__` processes the same parameters as its ancestor and *a few more*. Specifically:

::: DataFrame.DataFrame
    options:
      heading_level: 5
      show_docstring_classes: false
      merge_init_into_class: true
      show_root_heading: false
      show_root_full_path: false
      show_docstring_examples: false
      show_docstring_description: false
      show_bases: false
      members:
      - __init__
    
As you may have noticed, most of the parameters are optional. If you don't specify a parameter, a default value will be used. Or you will be notified once you use `anomeda.fit_trends`, `anomeda.find_anomalies` or other methods that you need to specify something additionally.

Here is some examples of how you can initialize a new `anomeda.DataFrame`:

```python
# With just a pandas.DataFrame and metric name
anomeda.DataFrame(pandas_df, metric_name='my_metric')

# With some measures
anomeda.DataFrame(pandas_df, measures_names=['A', 'B', 'C'], metric_name='my_metric')

# In a pandas.DataFrame style
anomeda.DataFrame(
    {
        'A': [0, 1, 2],
        'B': [3, 4, 5],
        'C': [6, 7, 8],
        'my_metric': [10, 20, 30]
    }
    measures_names=['A', 'B', 'C'], 
    metric_name='my_metric'
)


# With a discretization mappging
anomeda.DataFrame(
    pandas_df, 
    measures_names=['A', 'B', 'C'], 
    metric_name='my_metric',
    discretized_measures_mapping={
        'A': {
            0: [[20, 80]],           # map values from 20 to 80 to 0 - "normal values"
            1: [[0, 20], [80, 100]], # map values from 0 and 20 or between 80 and 100 to 1 - "abnormal values"
        }
    }
)

# And many more...
```

---
**NOTE 1**

Some *pandas* methods are not yet adapted for *anomeda*. They return a new `pandas.DataFrame` instead of a `anomeda.DataFrame`. You just need to initialize an *anomeda* object with a returned object in that case. 

---

---
**NOTE 2**

The scale of undex increments is extracted automatically. It can be 1 (*Integer*, if your index are integers) or a part of a timestamp (*second*, *minute*, *hour*, etc). 

For example, if your index consists of two values ['2024-01-01 00:00:00', '2024-01-01 00:01:00'], the step is *hour*. However, the step may become *minute* once you add only one value - ['2024-01-01 00:00:00', '2024-01-01 00:01:00', '2024-01-01 00:01:**01**'], since *minute* is the smallest increment now.

By default, anomeda does not fill the missing index values. So if there are infrequent index values, anomeda will use only them to use fit trends.

*TODO in new versions: add filling options*

---

A list of methods available to manipulate `anomeda.DataFrame`, such as *getters*, *setters*, *copying*, *modifying the object*, etc. Please follow [anomeda.DataFrame API Reference](dataframe_api.md) for the details. Her is the full list:

::: DataFrame.DataFrame
    options:
      heading_level: 6
      show_docstring_classes: false
      merge_init_into_class: false
      show_root_heading: false
      show_root_full_path: false
      show_docstring_examples: false
      show_docstring_description: false
      show_docstring_parameters: false
      show_docstring_examples: false
      show_docstring_returns: false
      show_signature: false
      show_bases: false

## How we handle continuous measures

Text

## How we fit trends

Text

## How we detect anomalies

Text

## Create an anomeda.DataFrame object

The base object anomeda is working with is the DataFrame. It is based on pandas.DataFrame object which contains events represented by rows or time series (aggregated events). It may contain:
- Index
    - Usually a datetime of a corresponding event
- Measures
    - Columns which describes an event, like "country", "region", "client category", etc. Measures can be numerical and categorical
- Metric
    - Metric you want to track, can be either aggregated, like "count of visits in a period", or non-aggregated, like "fact of visit"

```python
anmd_df = anomeda.DataFrame(
    data, # pandas.DataFrame
    measures_names=['dummy_measure', 'dummy_numeric_measure'], # columns represending measures
    measures_types={
        'categorical': ['dummy_measure'], 
        'continuous': ['dummy_numeric_measure']
    },
    index_name='dt',
    metric_name='metric',
    agg_func='sum' # function that is used to aggregate metric if more than 1 metric value will be found for a particular set of measure values
)
```

Each of the parameters can be changed with corresponding method, like `anomeda.DataFrame.set_measures_names`, `anomeda.DataFrame.set_measures_types`, `anomeda.DataFrame.set_index_name`, etc. The parameters can be retrieved with a corresponding getter, like `anomeda.DataFrame.get_measures_names`, `anomeda.DataFrame.get_measures_types`, `anomeda.DataFrame.get_index_name`, etc.

The underlying pandas.DataFrame object can be changed without creating a new instance of anomeda.DataFrame with `mod_data` method.

IF you need to get the pandas.DataFrame representation of an object, you can use `aspandas` method.

### Discretization of numeric measures

When continuous measures are passed, they will be mapped to discrete values by sklearn.mixture.BayesianGaussianMixture by default. You can pass your own discrete values of continuous measure when creating an anomeda.DataFrame object (see `discretized_measures` parameter of `anomeda.DataFrame` constructor) or later on (see method `anomeda.DataFrame.set_discretized_measures`)

If you need to see the discretized values, you can use `anomeda.DataFrame.get_discretization_mapping` method

## Find an anomaly change of the metric

`anomeda.find_anomalies` and `anomeda.find_anomalies_by_clusters` are responsible for looking for unusual metric changes in the data. They both currently use a method based of fitting a generic trend line and analyzing differences from a trend.

There are some parameters for the method which represent which part of anomalies from both tailes to present:

```python
p_large = 1 # 100% of anomalies with high values will be returned
p_low = 1 # 100% of anomalies with low values will be returned
```

The output is unique and sorted index values and an indication if total metric value is an anomaly or not. The values are aggregated using `agg_func` parameter.
```
index_values, anomalies_flag = anomeda.find_anomalies(anmd_df, n=n, p=p, normal_whole_window=normal_whole_window, read_deltas_consequently=read_deltas_consequently)
```

The methods use an alghorithm based on analyzing the **historical changes** of metric values in order to decide if the current point is an anomaly. The parameters mentioned before can tune the method for your needs.

## Find the root cause of an anomaly change of the metric

When you found an unusual period (whether by using `anomeda.find_anomalies` or by yourself), you can compare it to other period in order to find which events caused the difference between metric values.

All you need to do is to call `anomeda.explain_values_difference` with 2 anomeda.DataFrame objects containing data you want to compare. It will automatically detect clusters of events using measure values, compare the metric values and return the result.

```python
anomeda.explain_values_difference(
    anmd_df1,
    anmd_df2
)
```

The output is a pandas.DataFrame with average metric value in each cluster in both periods and the *differences* between them. Thus, sorting by the absolute or relative differences and average values, you can find the most important clusters in terms of its contribution to the overall metric's change. 

## Find clusters of events which make your metric rise or decrease

Anomeda can calculate trends of the metric for each cluster in your data and show the most significat positive and negative contributions. It aggregates metric values by index, fit a line a * x + b, where x is index numeric value, and produce the output.

```python
anomeda.describe_trends_by_clusters(anmd_df)
```

The output is a pandas.DataFrame with the following columns:
- Measures
- trend_coeff
    - Coefficient *a* of a fitted line a * x + b
- trend_bias
    - Coefficient *b* of a fitted line a * x + b
- avg_value 
- contribution_metric 
    - Coefficient which describes how important the cluster for the total metric is. It is calculated as follows: `sign(a) * abs(a * avg_value)`, where a is coefficient *a* of a fitted line a * x + b