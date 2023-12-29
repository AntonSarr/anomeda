# API reference

Here you can find the documentation for all available endpoints of **anomeda** Python package.

## DataFrame

```python
anomeda.DataFrame(
    data,
    measures_names=None, 
    measures_types=None, 
    discretized_measures=None, 
    index_name=None, 
    metric_name=None, 
    agg_func='sum'
)
```

Class containing data to be processed by anomeda package.
    
### Parameters

**data** : *pandas.DataFrame*
    Underlying data must be pandas.DataFrame object

**measures_names** : *list of str or tuple objects*
    List containing columns considered as measures in the data

**measures_types** : *dict*
    Dict containing 'categorical' and/or 'continuous' keys and list of measures as values. Continuous measures will be discretized automatically if not presented in discretized_measures parameter.

**discretized_measures** : *dict*
    Dict containig name of the measure as key and array-like object containing discretized values of the measure of the same shape as original data. If measure is in 'continuous' list of measures_types parameter, it will be discretized automatically.

**index_name** : *str or list*
    Columns to be considered as an index (usually a date or a timestamp)

**metric_name** : *str*
    Column with a metric to be analyzed

**agg_func**: *str*
    Way of aggregating metric_name by measures. Can be 'sum', 'avg' or callable compatible with pandas.DataFrame.groupby

### Example

```python
anmd_df = anomeda.DataFrame(
    data,
    measures_names=['dummy_measure', 'dummy_numeric_measure'],
    measures_types={
        'categorical': ['dummy_measure'], 
        'continuous': ['dummy_numeric_measure']
    },
    index_name='dt',
    metric_name='metric',
    agg_func='sum'
)
```

## DataFrame.aspandas

Text

## DataFrame.copy

Text

## DataFrame.get_agg_func

Text

## DataFrame.get_discretization_mapping

Text

## DataFrame.get_discretized_measures

Text

## DataFrame.get_index_name

Text

## DataFrame.get_measures_names

Text

## DataFrame.get_measures_types

Text

## DataFrame.get_metric_name

Text

## DataFrame.mod_data

Text

## DataFrame.set_agg_func

Text

## DataFrame.set_discretization_mapping

Text

## DataFrame.set_discretized_measures

Text

## DataFrame.set_index_name

Text

## DataFrame.set_measures_names

Text

## DataFrame.set_measures_types

Text

## DataFrame.set_metric_name

Text

## describe_trend

Text

## describe_trends_by_clusters

Text

## describe_variance_by_clusters

Text

## explain_values_difference

Text

## explain_variance_difference

Text

## find_anomalies

Text

## find_anomalies_by_clusters

Text