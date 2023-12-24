import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
from sklearn.mixture import BayesianGaussianMixture

from anomeda.DataFrame import DataFrame, _to_discrete_values

EMPTY_ARGUMENT_MSG = "If {0} is not provided, data's type must be anomeda.DataFrame with provided {0}"
INVALID_DATA_TYPE_MSG = "Data type must be anomeda.DataFrame, while {0} is provided"


def linreg(x, a, b):
    """Return the value of f(x) = a * x + b"""
    
    return a * x + b


def describe_trend(data: 'anomeda.DataFrame') -> (float, float):
    """Fit and return parameters of a linear trend for the given metric.
    
    Parameters
    ----------
    data : anomeda.DataFrame
        Object containing data to be analyzed
        
    Return
    ------
    a, b : float
        Coefficients a, b of a line f(x) = a + b * x fitted to the metric, where x is time values and f(x) is metric values
    """
    
    if type(data) != DataFrame:
        raise TypeError(INVALID_DATA_TYPE_MSG.format(type(data)))
    
    data_pandas = data.aspandas() # get pandas.DataFrame
    
    if data.get_index_name() is None:
        raise KeyError('not-None "index_name" must be set for anomeda.DataFrame object')
    index_name = data.get_index_name()
    
    if data.get_metric_name() is None:
        raise KeyError('not-None "metric_name" must be set for anomeda.DataFrame object')
    metric_name = data.get_metric_name()
    
    if data.get_agg_func() is None:
        raise KeyError('not-None "agg_func" must be set for anomeda.DataFrame object')
    agg_func = data.get_agg_func()
    
    data_pandas = data_pandas.reset_index()
    columns_to_use = np.append(index_name, metric_name)
    
    if data_pandas.shape[0] >= 2:
        
        ydata = data_pandas[columns_to_use].groupby(index_name).agg(agg_func)[metric_name]
        xdata = np.arange(len(ydata)) 

        coef, _ = curve_fit(linreg, xdata, ydata)
        a, b = coef

        return a, b
    
    elif data_pandas.shape[0] == 1:
        a, b = 1, data_pandas[metric_name].mean()
        
        return a, b
    
    else:
        return None, None 


def describe_trends_by_clusters(data: 'anomeda.DataFrame') -> 'pandas.DataFrame':
    """Fit and return parameters of a linear trend of each cluster for given metric. Output object type is a pandas.DataFrame object.
    
    Parameters
    ----------
    data : anomeda.DataFrame
        Object containing data to be analyzed
        
    Return
    ------
    output : pandas.DataFrame
        Object describing the trends. 
    """
    
    if type(data) != DataFrame:
        raise TypeError(INVALID_DATA_TYPE_MSG.format(type(data)))
         
    if data.get_measures_names() is None:
        raise KeyError('not-None "measures_names" must be set for anomeda.DataFrame object')
    measures_names = data.get_measures_names()
    
    if data.get_index_name() is None:
        raise KeyError('not-None "index_name" must be set for anomeda.DataFrame object')
    index_name = data.get_index_name()
    
    if data.get_metric_name() is None:
        raise KeyError('not-None "metric_name" must be set for anomeda.DataFrame object')
    metric_name = data.get_metric_name()
    
    if data.get_agg_func() is None:
        raise KeyError('not-None "agg_func" must be set for anomeda.DataFrame object')
    agg_func = data.get_agg_func()
    
    data_pandas = data.aspandas() # get pandas.DataFrame
    data_pandas = data_pandas.reset_index()
    columns_to_use = np.append(index_name, metric_name)
    
    measures_types = data.get_measures_types()
    if measures_types is not None and 'continuous' in measures_types:
        for measure in measures_types['continuous']:
            data_pandas[measure] = data.get_discretized_measures()[measure]
    
    res_values = []
    for measures_values in data_pandas[measures_names].drop_duplicates().values:
        str_arr = []
        for (measure_name, measure_value) in zip(measures_names, measures_values):
            str_arr.append('`' + measure_name + '`' + '==' + str(measure_value))
        query = ' and '.join(str_arr)
        
        ydata = data_pandas.query(query)[columns_to_use].groupby(index_name).agg(agg_func)[metric_name]
        xdata = np.arange(len(ydata)) 
        
        if ydata.shape[0] >= 2:
            coef, _ = curve_fit(linreg, xdata, ydata)
            a, b = coef
        else:
            a, b = 1, data_pandas[metric_name].mean()
            
        res_values.append(np.append(measures_values, [a, b, np.mean(ydata), (int(a > 0) * 2 - 1) * abs(a * np.mean(ydata))]))

    res = pd.DataFrame(
        res_values, 
        columns=np.append(measures_names, ['trend_coeff', 'trend_bias', 'avg_value', 'contribution_metric']))\
    .sort_values(by='trend_coeff', ascending=False).reset_index(drop=True)

    return res


def describe_variance_by_clusters(data: 'anomeda.DataFrame') -> 'pandas.DataFrame':
    """Return variances of metric in each cluster in given data. Output object type is a pandas.DataFrame object.
    
    Parameters
    ----------
    data : anomeda.DataFrame
        Object containing data to be analyzed
        
    Return
    ------
    output : pandas.DataFrame
        Object describing the variances. 
    """
    
    if type(data) != DataFrame:
        raise TypeError(INVALID_DATA_TYPE_MSG.format(type(data)))
        
    if data.get_measures_names() is None:
        raise KeyError('not-None "measures_names" must be set for anomeda.DataFrame object')
    measures_names = data.get_measures_names()
    
    if data.get_index_name() is None:
        raise KeyError('not-None "index_name" must be set for anomeda.DataFrame object')
    index_name = data.get_index_name()
    
    if data.get_metric_name() is None:
        raise KeyError('not-None "metric_name" must be set for anomeda.DataFrame object')
    metric_name = data.get_metric_name()
    
    if data.get_agg_func() is None:
        raise KeyError('not-None "agg_func" must be set for anomeda.DataFrame object')
    agg_func = data.get_agg_func()
    
    data_pandas = data.aspandas() # get pandas.DataFrame
    data_pandas = data_pandas.reset_index()
    columns_to_use = np.append(index_name, metric_name)
    
    measures_types = data.get_measures_types()
    if measures_types is not None and 'continuous' in measures_types:
        for measure in measures_types['continuous']:
            data_pandas[measure] = data.get_discretized_measures()[measure]
    
    res_values = []
    for measures_values in data_pandas[measures_names].drop_duplicates().values:
        str_arr = []
        for (measure_name, measure_value) in zip(measures_names, measures_values):
            str_arr.append('`' + measure_name + '`' + '==' + str(measure_value))
        query = ' and '.join(str_arr)
        
        ydata = data_pandas.query(query)[columns_to_use].groupby(index_name).agg(agg_func)[metric_name]

        var = np.var(ydata)

        res_values.append(np.append(measures_values, [var]))

    res = pd.DataFrame(
        res_values, 
        columns=np.append(measures_names, ['variance']))\
    .sort_values(by='variance', ascending=False).reset_index(drop=True)
    
    return res


def explain_values_difference(
    data1: 'anomeda.DataFrame', 
    data2: 'anomeda.DataFrame'
) -> 'pandas.DataFrame':
    """Find clusters in data which caused the most significant changes of average value of a metric when comparing data1 and data2. Output object type is a pandas.DataFrame object.
    
    Parameters
    ----------
    data1 : anomeda.DataFrame
        Object containing data to be analyzed
    data2 : anomeda.DataFrame
        Object containing data to be analyzed
        
    Return
    ------
    output : pandas.DataFrame
        Object describing the clusters with the most significant changes of average value of a metric. 
    """
    
    if type(data1) != DataFrame:
        raise TypeError(INVALID_DATA_TYPE_MSG.format(type(data1)))
        
    if type(data2) != DataFrame:
        raise TypeError(INVALID_DATA_TYPE_MSG.format(type(data2)))
        
    if data1.get_measures_names() is None:
        raise KeyError('not-None "measure_names" must be set for anomeda.DataFrame object')
    measure_names1 = data1.get_measures_names()
    
    if data2.get_measures_names() is None:
        raise KeyError('not-None "measure_names" must be set for anomeda.DataFrame object')
    measure_names2 = data2.get_measures_names()
        
    data1_pd = data1.aspandas()
    data2_pd = data2.aspandas()
    
    trends1 = describe_trends_by_clusters(data1)
    trends2 = describe_trends_by_clusters(data2)
    
    measures_names = list(set(measure_names1).intersection(set(measure_names2)))
    
    res = trends1.merge(trends2, how='outer', on=measures_names).fillna(0)
    res['abs_avg_values_diff'] = res['avg_value_y'] - res['avg_value_x']
    res['rlt_avg_values_diff'] = res['abs_avg_values_diff'] / res['avg_value_x']
    
    res['abs_trend_coeff_diff'] = res['trend_coeff_y'] - res['trend_coeff_x']
    res['rlt_trend_coeff_diff'] = res['trend_coeff_y'] / res['trend_coeff_x']
    
    res = res[np.append(measures_names, [
        'avg_value_x', 'avg_value_y', 
        'abs_avg_values_diff', 
        'rlt_avg_values_diff', 
        'trend_coeff_x', 'trend_coeff_y',
        'abs_trend_coeff_diff',
        'rlt_trend_coeff_diff'
    ])]\
    .rename(columns={
        'avg_value_x': 'avg_value_1', 
        'avg_value_y': 'avg_value_2',
        'trend_coeff_x': 'trend_coeff_1',
        'trend_coeff_y': 'trend_coeff_2'
    }).\
    sort_values(by='abs_trend_coeff_diff', ascending=False)
    
    return res


def explain_variance_difference(
    data1: 'anomeda.DataFrame', 
    data2: 'anomeda.DataFrame'
) -> 'pandas.DataFrame':
    """Find clusters in data which caused the most significant changes of variance of a metric when comparing data1 and data2. Output object type is a pandas.DataFrame object.
    
    Parameters
    ----------
    data1 : anomeda.DataFrame
        Object containing data to be analyzed
    data2 : anomeda.DataFrame
        Object containing data to be analyzed
        
    Return
    ------
    output : pandas.DataFrame
        Object describing the clusters with the most significant changes of variance of a metric. 
    """
    
    if type(data1) != DataFrame:
        raise TypeError(INVALID_DATA_TYPE_MSG.format(type(data1)))
        
    if type(data2) != DataFrame:
        raise TypeError(INVALID_DATA_TYPE_MSG.format(type(data2)))
        
    if data1.get_measures_names() is None:
        raise KeyError('not-None "measure_names" must be set for anomeda.DataFrame object')
    measure_names1 = data1.get_measures_names()
    
    if data2.get_measures_names() is None:
        raise KeyError('not-None "measure_names" must be set for anomeda.DataFrame object')
    measure_names2 = data2.get_measures_names()
        
    data1_pd = data1.aspandas()
    data2_pd = data2.aspandas()
    
    variances1 = describe_variance_by_clusters(data1)
    variances2 = describe_variance_by_clusters(data2)
    
    measures_names = list(set(measure_names1).intersection(set(measure_names2)))
    
    res = variances1.merge(variances2, how='outer', on=measures_names).fillna(0)
    res['abs_variance_diff'] = res['variance_y'] - res['variance_x']
    res['rlt_variance_diff'] = res['abs_variance_diff'] / res['variance_x']
    
    res = res[np.append(measures_names, [
        'variance_x', 'variance_y', 
        'abs_variance_diff', 
        'rlt_variance_diff'
    ])]\
    .rename(columns={
        'variance_x': 'avg_value_1', 
        'variance_y': 'avg_value_2'
    }).\
    sort_values(by='abs_variance_diff', ascending=False)
    
    return res


def find_anomalies(
    data: 'anomeda.DataFrame', 
    n=1, 
    p=(0.02, 0.98), 
    normal_whole_window=True, 
    read_deltas_consequently=False
):
    """Find metric anomalies by looking for the most extreme metric changes.
    
    The method decides if a metric value is an anomaly by looking at the n metric changes before and comparing them to "normal" changes we observed from historical or all available data.
    
    Parameters
    ----------
    data : anomeda.DataFrame
        Object containing data to be analyzed
    n : int
        The number of index values the model looks back when gathering metric changes for a particular point to decide if it is an anomaly. Larger values will make the model mark a few point after the first anomaly as anomalies as well.
    p : tuple of float
        Tuple of lowest and highest percentiles of day-to-day metric changes. Deltas within this range will be considered as "normal"
    normal_whole_window : bool
        Bool indicating if all the metric deltas in period [dt - n days, dt] period have to in "normal" range (True) or at least one them (False) in order metric value to be considered as a normal point
    read_deltas_consequently : bool
        Bool indicating if to scan metric deltas consequently (day by day, only historical data) (True) or to scan all available metric changes at once (False)
        
    Return
    ------
    index : numpy.array
        Array of unique indexes
    anomalies : numpy.array of bool
        Bool array indicating if a metric was abnormal at a particar index point
    """
    
    m = 0
    
    if type(data) != DataFrame:
        raise TypeError(INVALID_DATA_TYPE_MSG.format(type(data)))
    
    if data.get_index_name() is None:
        raise KeyError('not-None "index_name" must be set for anomeda.DataFrame object')
    index_name = data.get_index_name()
    
    if data.get_metric_name() is None:
        raise KeyError('not-None "metric_name" must be set for anomeda.DataFrame object')
    metric_name = data.get_metric_name()
    
    if data.get_agg_func() is None:
        raise KeyError('not-None "agg_func" must be set for anomeda.DataFrame object')
    agg_func = data.get_agg_func()
    
    data_pandas = data.aspandas() # get pandas.DataFrame
    data_pandas = data_pandas.reset_index()
    columns_to_use = np.append(index_name, metric_name)
    
    df = data_pandas[columns_to_use].groupby(index_name).agg(agg_func).reset_index().sort_values(by=index_name, ascending=True)
    
    for i in range(1, n + 1):
        df['metric_shifted_left_' + str(i)] = df[metric_name].shift(i)
        df['delta_left_' + str(i)] = df[metric_name] - df['metric_shifted_left_' + str(i)]

    for i in range(1, m + 1):
        df['metric_shifted_right_' + str(i)] = df[metric_name].shift(-i)
        df['delta_right_' + str(i)] = df[metric_name] - df['metric_shifted_right_' + str(i)]

    deltas_columns = np.append(['delta_left_' + str(i + 1) for i in range(n)], ['delta_right_' + str(i + 1) for i in range(m)])

    normality_arr = []

    for delta_col in deltas_columns:
        if read_deltas_consequently:
            df['low_percentile'] = [df[delta_col].head(i).quantile(p[0]) for i in range(df.shape[0])]
            df['high_percentile'] = [df[delta_col].head(i).quantile(p[1]) for i in range(df.shape[0])]
        else:
            df['low_percentile'] = df[delta_col].quantile(p[0])
            df['high_percentile'] = df[delta_col].quantile(p[1])

        normality_arr.append(((df[delta_col] <= df['high_percentile']) & (df[delta_col] >= df['low_percentile'])).values)

    normality_arr = np.vstack(normality_arr).T

    if normal_whole_window:
        normal_points = normality_arr.any(axis=1)
    else:
        normal_points = normality_arr.all(axis=1)

    return df[index_name].values, ~normal_points
        
        
def find_anomalies_by_clusters(
    data: 'anomeda.DataFrame', 
    n=1, 
    p=(0.02, 0.98), 
    normal_whole_window=True, 
    read_deltas_consequently=False
):
    """Find metric anomalies in each cluster by looking for the most extreme metric changes.
    
    The method decides if a metric value is an anomaly by looking at the n metric changes before and comparing them to "normal" changes we observed from historical or all available data.
    
    Parameters
    ----------
    data : anomeda.DataFrame
        Object containing data to be analyzed
    n : int
        The number of index values the model looks back when gathering metric changes for a particular point to decide if it is an anomaly. Larger values will make the model mark a few point after the first anomaly as anomalies as well.
    p : tuple of float
        Tuple of lowest and highest percentiles of day-to-day metric changes. Deltas within this range will be considered as "normal"
    normal_whole_window : bool
        Bool indicating if all the metric deltas in period [dt - n days, dt] period have to in "normal" range (True) or at least one them (False) in order metric value to be considered as a normal point
    read_deltas_consequently : bool
        Bool indicating if to scan metric deltas consequently (day by day, only historical data) (True) or to scan all available metric changes at once (False)
        
    Return
    ------
    clusters_anomalies : list of dict
        List containing cluster and its anomalies. The keys in the cluster-dict are "cluster", "indeces", "anomalies"
    """
    
    if type(data) != DataFrame:
        raise TypeError(INVALID_DATA_TYPE_MSG.format(type(data)))
        
    if data.get_measures_names() is None:
        raise KeyError('not-None "measures_names" must be set for anomeda.DataFrame object')
    measures_names = data.get_measures_names()
    
    if data.get_index_name() is None:
        raise KeyError('not-None "index_name" must be set for anomeda.DataFrame object')
    index_name = data.get_index_name()
    
    if data.get_metric_name() is None:
        raise KeyError('not-None "metric_name" must be set for anomeda.DataFrame object')
    metric_name = data.get_metric_name()
    
    if data.get_agg_func() is None:
        raise KeyError('not-None "agg_func" must be set for anomeda.DataFrame object')
    agg_func = data.get_agg_func()
    
    data_pandas = data.aspandas() # get pandas.DataFrame
    data_pandas = data_pandas.reset_index()
    
    unchanged_data_pandas = data_pandas.copy()
    
    columns_to_use = np.append(index_name, metric_name)
    
    measures_types = data.get_measures_types()
    if measures_types is not None and 'continuous' in measures_types:
        for measure in measures_types['continuous']:
            data_pandas[measure] = data.get_discretized_measures()[measure]
    
    clusters_anomalies = []
    for measures_values in data_pandas[measures_names].drop_duplicates().values:
        str_arr = []
        for (measure_name, measure_value) in zip(measures_names, measures_values):
            str_arr.append('`' + measure_name + '`' + '==' + str(measure_value))
        query = ' and '.join(str_arr)
        
        filtered_data = data_pandas.query(query)
        
        ydata = filtered_data[columns_to_use].groupby(index_name).agg(agg_func)[metric_name]
        
        indeces, anomalies = find_anomalies(data.mod_data(unchanged_data_pandas.loc[filtered_data.index]), n, p, normal_whole_window, read_deltas_consequently)
        
        cluster = dict(zip(measures_names, measures_values))
        
        clusters_anomalies.append({
            'cluster': cluster,
            'indeces': indeces,
            'anomalies': anomalies
        })
        
    return clusters_anomalies
        

