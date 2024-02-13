from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.stats import beta
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import BayesianGaussianMixture

from anomeda.DataFrame import DataFrame, _to_discrete_values

EMPTY_ARGUMENT_MSG = "If {0} is not provided, data's type must be anomeda.DataFrame with provided {0}"
INVALID_DATA_TYPE_MSG = "Data type must be anomeda.DataFrame, while {0} is provided"


def __regularizer(x):
    return beta.pdf(x, 0.4, 0.4)


def __find_trend_breaking_point(x, y):
    metric_vals = []
    points_candidates = x[2:-1]

    if len(points_candidates) == 0:
        return None
    
    for dt in range(len(x))[2:-1]:
    
        y_true1 = y[:dt]
        x1 = np.arange(len(y_true1))
    
        y_true2 = y[dt:]
        x2 = np.arange(len(y_true2))
    
        linreg_fitted1 = linregress(x1, y_true1)
        y_pred1 = linreg_fitted1.slope * x1 + linreg_fitted1.intercept
    
        linreg_fitted2 = linregress(x2, y_true2)
        y_pred2 = linreg_fitted2.slope * x2 + linreg_fitted2.intercept
        
        ratio1 = len(y_true1) / (len(y_true1) + len(y_true2))
        ratio2 = len(y_true2) / (len(y_true1) + len(y_true2))
    
        metric = __regularizer(ratio1) * np.var(np.abs(y_pred1 - y_true1)) \
        + __regularizer(ratio2) * np.var(np.abs(y_pred2 - y_true2))
        
        metric_vals.append(metric)
    
    return points_candidates[np.argmin(metric_vals)]


def linreg(x, a, b):
    """Return the value of f(x) = a * x + b."""
    return a * x + b


def extract_trends(x, y, max_trends='auto', min_var_reduction=0.5, verbose=False):
    """Fit and return parameters of a linear trend for the given metric.
    
    Parameters
    ----------
    x : np.array[int]
        Indeces. Must start with 0 and increase sequentially.
    y : np.array[float]
        Values
    max_trends : int | 'auto'
        Number of trends to extract. If int, the method extracts this amount of trends or less. Less trends may be returned if no more trends are found or if the min_var_reduction is reached (variance is already explaines by less amount of trends). If 'auto', the method defines the number of trends automatically using min_var_reduction parameter. Default is 'auto'.
    min_var_reduction : float[0, 1] or None
        % of the variance of approximation error that must be reduced comparing to the variance of the initial approximation with only one trend. Values closer to 1 will produce cause extracting more trends, since more trends reduce the variance better. Values closer to 0 will cause producing less trends. If max_trends is set and it is reached, the extraction finished regardless the value of the variance. If None, then not used. Default is 0.5.
    verbose : bool
        If to produce more logs. Default False.
        
    Returns
    -------
    trends : dict
        Dict contains the extracted trends in the format {trend_id: (xmin_inc, xmax_exc, (trend_slope, trend_intersept)), ...}
    """
    if max_trends == 'auto':
        if min_var_reduction is None:
            raise ValueError("Either max_trends or min_var_reduction parameters must be set. max_trends='auto' and min_var_reduction=None at the same time is not permitted.")
        max_trends = np.inf

    if min_var_reduction is None:
        min_var_reduction = np.inf
    
    linreg_fitted = linregress(x, y)
    y_fitted = linreg_fitted.slope * x + linreg_fitted.intercept
    error_var = np.var(np.abs(y_fitted - y))
    
    trends = {
        0: (
            x[0], x[-1] + 1, 
            (linreg_fitted.slope, linreg_fitted.intercept), 
            (y.shape[0], np.mean(y), error_var, np.sum(y))
        )
    }
    
    best_vars = [error_var]

    if max(best_vars) == 0:
        reducted_variance = 1
    else:
        reducted_variance = 1 - min(best_vars) / max(best_vars) # that much of variance we explained
    
    n = len(trends.keys())
    while n < max_trends and reducted_variance < min_var_reduction:
    
        min_var_diff = 0
        best_id = None
        best_dt = None
        best_params = None
        best_var = None
        for id in trends.keys():
            xmin, xmax, (a, b), _ = trends[id]
            
            dt = __find_trend_breaking_point(x[xmin: xmax], y[xmin: xmax])
            
            if dt is None:
                continue
    
            y_true1 = y[xmin: dt]
            x1 = np.arange(len(y_true1))
            
            y_true2 = y[dt: xmax]
            x2 = np.arange(len(y_true2))
            
            linreg_fitted1 = linregress(x1, y_true1)
            y_pred1 = linreg_fitted1.slope * x1 + linreg_fitted1.intercept
            
            linreg_fitted2 = linregress(x2, y_true2)
            y_pred2 = linreg_fitted2.slope * x2 + linreg_fitted2.intercept
    
            y_base_diffs = []
            y_new_diffs = []
            for trend_id in trends.keys():
                trend_xmin, trend_xmax, (trend_a, trend_b), _ = trends[trend_id]
                y_trend_true = y[trend_xmin: trend_xmax]
                y_trend_predicted = trend_a * np.arange(len(y_trend_true)) + trend_b
                y_trend_diff = y_trend_predicted - y_trend_true
                y_base_diffs.append(y_trend_diff)
                
                if trend_id == id:
                    y_new_diffs.append(y_pred1 - y_true1)
                    y_new_diffs.append(y_pred2 - y_true2)
                else:
                    y_new_diffs.append(y_trend_diff)
                
            y_base_diffs = np.concatenate(y_base_diffs)
            y_new_diffs = np.concatenate(y_new_diffs)
    
            new_var = np.var(np.abs(y_new_diffs))
            old_var = np.var(np.abs(y_base_diffs))
            var_diff = new_var - old_var
    
            if var_diff < min_var_diff:
                min_var_diff = var_diff
                best_id = id
                best_dt = dt
                best_params = [(linreg_fitted1.slope, linreg_fitted1.intercept), (linreg_fitted2.slope, linreg_fitted2.intercept)]
                best_var = new_var
                best_aggs = [
                    (
                        y_true1.shape[0], 
                        np.mean(y_true1), 
                        np.var(np.abs(y_true1 - y_pred1)), 
                        np.sum(y_true1)
                    ),
                    (
                        y_true2.shape[0], 
                        np.mean(y_true2), 
                        np.var(np.abs(y_true2 - y_pred2)), 
                        np.sum(y_true2)
                    )
                ]
    
        if best_id is not None:
            left_trend = (trends[best_id][0], best_dt, best_params[0], best_aggs[0])
            right_trend = (best_dt, trends[best_id][1], best_params[1], best_aggs[1])
    
            trends[best_id] = left_trend
            trends[max(trends.keys()) + 1] = right_trend
            best_vars.append(best_var)
        else:
            if verbose:
                print('No more trends were found. Finish with {} trends.'.format(n))
            break
    
        if max(best_vars) == 0:
            reducted_variance = 1
        else:
            reducted_variance = 1 - min(best_vars) / max(best_vars) # that much of variance we explained
        
        if reducted_variance <= min_var_reduction:
            if verbose:
                print('Variance reduced by {} comparing to initial value, while the reduction of {} is needed. Finish with {} trends'.format(reducted_variance, min_var_reduction, n))
            break
        
        n += 1
        continue

    return trends


def describe_trend(data):
    """Fit and return parameters of a linear trend for the given metric.
    
    Parameters
    ----------
    data : anomeda.DataFrame
        Object containing data to be analyzed
        
    Returns
    -------
    a : float
        Coefficient a of a line f(x) = a + b * x fitted to the metric, where x is time values and f(x) is metric values
    b : float
        Coefficient b of a line f(x) = a + b * x fitted to the metric, where x is time values and f(x) is metric values
    """
    if type(data) != DataFrame:
        raise TypeError(INVALID_DATA_TYPE_MSG.format(type(data)))
    
    data_pandas = data.to_pandas() # get pandas.DataFrame
    
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


def describe_trends_by_clusters(data):
    """Fit and return parameters of a linear trend of each cluster for given metric. Output object type is a pandas.DataFrame object.
    
    Parameters
    ----------
    data : anomeda.DataFrame
        Object containing data to be analyzed
        
    Returns
    -------
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
    
    data_pandas = data.to_pandas() # get pandas.DataFrame
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


def describe_variance_by_clusters(data):
    """Return variances of metric in each cluster in given data. Output object type is a pandas.DataFrame object.
    
    Parameters
    ----------
    data : anomeda.DataFrame
        Object containing data to be analyzed
        
    Returns
    -------
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
    
    data_pandas = data.to_pandas() # get pandas.DataFrame
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
    data1, 
    data2,
    measures_to_iterate='shared'
):
    """Find clusters in data which caused the most significant changes of an average value of the metric.
    
    data1 and data2 must have a common metric and at least 1 common measure.
    
    Parameters
    ----------
    data1 : anomeda.DataFrame
        Object containing data to be analyzed
    data2 : anomeda.DataFrame
        Object containing data to be analyzed
    measures_to_iterate: list, 'shared' or 'combinations'. Default is 'shared'
        Measures combinations used to create clusters to look for differences between metric values. If 'shared', then one set consisting of all measures shared between data objects is used. If 'combinations', then all possible combinations of measures are used. If list, then lists inside are used to create sets.
        
    Examples
    --------
    ```python
    anomeda.explain_values_difference(
        data1,
        data2,
        measures_to_iterate=[
            ['dummy_measure'], 
            ['dummy_measure', 'dummy_numeric_measure']
        ] # equivalent to measures_to_iterate='combinations' if data1 and data2 have only 'dummy_measure' and 'dummy_numeric_measure' in common
    )
    ```
        
    Returns
    -------
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
    
    measures_names = list(set(measure_names1).intersection(set(measure_names2)))
    
    if len(measures_names) == 0:
        raise ValueError('data1 and data2 objects must have at least one measure in common')
        
    if data1.get_metric_name() != data2.get_metric_name():
        raise ValueError('data1 and data2 must have a common metric')
    
    if measures_to_iterate is not None:
        if not any([
            measures_to_iterate == 'shared', 
            measures_to_iterate == 'combinations', 
            type(measures_to_iterate) == list]
        ):
            raise ValueError("measures_to_iterate attribute must be 'shared', 'combinations' or list of lists of measures")
            
    if measures_to_iterate == 'shared':
        measures_to_iterate_list = [measures_names] 
    
    if measures_to_iterate == 'combinations':
        measures_to_iterate_list = []
        for i in range(1, len(measures_names) + 1):
            for el in combinations(measures_names, i):
                measures_to_iterate_list.append(list(el))
    
    if type(measures_to_iterate) == list:
        measures_to_iterate_list = measures_to_iterate
    
    reference_columns = np.append(measures_names, [
        'avg_value_x', 'avg_value_y', 
        'abs_avg_values_diff', 
        'rlt_avg_values_diff', 
        'trend_coeff_x', 'trend_coeff_y',
        'abs_trend_coeff_diff',
        'rlt_trend_coeff_diff'
    ])
    
    reference_df = pd.DataFrame([[0] * len(reference_columns)], columns=reference_columns)
      
    res_arr = []
    for cur_measures in measures_to_iterate_list:
        
        data1_cp = data1.copy()
        data1_cp.set_measures_names(cur_measures)
        
        data2_cp = data2.copy()
        data2_cp.set_measures_names(cur_measures)
        
        trends1 = describe_trends_by_clusters(data1_cp)
        trends2 = describe_trends_by_clusters(data2_cp)

        res = trends1.merge(trends2, how='outer', on=cur_measures).fillna(0)
        res['abs_avg_values_diff'] = res['avg_value_y'] - res['avg_value_x']
        res['rlt_avg_values_diff'] = res['abs_avg_values_diff'] / res['avg_value_x']

        res['abs_trend_coeff_diff'] = res['trend_coeff_y'] - res['trend_coeff_x']
        res['rlt_trend_coeff_diff'] = res['trend_coeff_y'] / res['trend_coeff_x']
        
        res, _ = res.align(reference_df, axis=1)
        
        res_arr.append(res)
        
    res = pd.concat(res_arr).reset_index(drop=True)
    res = res[reference_columns].rename(columns={
        'avg_value_x': 'avg_value_1', 
        'avg_value_y': 'avg_value_2',
        'trend_coeff_x': 'trend_coeff_1',
        'trend_coeff_y': 'trend_coeff_2'
    }).\
    sort_values(by='abs_trend_coeff_diff', ascending=False)
    
    return res


def explain_variance_difference(
    data1, 
    data2
):
    """Find clusters in data which caused the most significant changes of variance of a metric when comparing data1 and data2. Output object type is a pandas.DataFrame object.
    
    Parameters
    ----------
    data1 : anomeda.DataFrame
        Object containing data to be analyzed
    data2 : anomeda.DataFrame
        Object containing data to be analyzed
        
    Returns
    -------
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
        
    data1_pd = data1.to_pandas()
    data2_pd = data2.to_pandas()
    
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
    data,
    p_large=1,
    p_low=1,
    trend='linear',
    **kwargs
):
    """Find metric anomalies by looking for the most extreme metric changes.
    
    The method finds differences between real metric and a fitted trend line, find points with the most extreme differences and marks them as anomalies.
    
    Parameters
    ----------
    data : anomeda.DataFrame
        Object containing data to be analyzed
    p_large : float, [0, 1], default 1
        What part of anomalies which are higher than usual values needs to be returned. For example, if you set it to 0.7, then only 70% of the anomalies with the largest values will be returned. Default is 1.
    p_low : float, [0, 1], default 1
        What part of anomalies which are lower than usual values needs to be returned. For example, if you set it to 0.5, then only 50% of the anomalies with the lowest values will be returned. Default is 1.
    trend : 'linear', 'adjusted-linear'
        The way which is used to fit a trend of the metric. If 'linear', then one linear function is fitted. If 'adjusted-linear', then trends are extracted and fitted automatically, the anomeda.extract_trends method is used. Its parameters can be passed with **kwargs argument. Default is 'linear'.
        
    Returns
    -------
    index : numpy.ndarray
        Array of unique indexes
    anomalies : numpy.ndarray of bool
        Bool array indicating if a metric was abnormal at a particar index point
    """
    if trend not in ['linear', 'adjusted-linear']:
        raise ValueError('trend parameter must be either "linear" or "adjusted-linear"')
    
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
    
    data_pandas = data.to_pandas() # get pandas.DataFrame
    data_pandas = data_pandas.reset_index()
    columns_to_use = np.append(index_name, metric_name)
    
    df = data_pandas[columns_to_use].groupby(index_name).agg(agg_func).reset_index().sort_values(by=index_name, ascending=True)

    y = df[metric_name].values
    x = np.arange(len(y))

    if trend == 'adjusted-linear':
        extracted  = extract_trends(x, y, **kwargs)
        y_diff = []
        x_labels = []
        for t in extracted.values():
            xmin, xmax, (slope, intercept), _ = t
            y_fitted = slope * np.arange(xmax - xmin) + intercept
            y_diff.append(y[xmin: xmax] - y_fitted)
            x_labels.append(np.arange(xmin, xmax))
        x_labels = np.concatenate(x_labels)
        y_diff = np.concatenate(y_diff)
        y_diff = y_diff[np.argsort(x_labels)]
            
    if trend == 'linear':
        linreg_fitted = linregress(x, y)
        y_fitted = linreg_fitted.slope * x + linreg_fitted.intercept
        y_diff = y - y_fitted
    
    clusterizator = LocalOutlierFactor(n_neighbors=min(len(y) - 1, 24), novelty=False)
    outliers = clusterizator.fit_predict(y_diff.reshape(-1, 1)) == -1

    # Remove inliers, i.e. isolated points not from the left or right tail of values range
    # We want to keep only the largest or the lowest differences
    # We also keep only % of anomalies defined by p_low and p_large
    indeces_sorted_by_y_diff = np.argsort(y_diff)
    not_outliers_indeces = np.where(outliers[indeces_sorted_by_y_diff] == False)[0]
    outliers[indeces_sorted_by_y_diff[np.min(not_outliers_indeces): np.max(not_outliers_indeces)]] = False
    
    sorted_outliers = outliers[indeces_sorted_by_y_diff]
    
    n_low_anomalies = 0
    n_large_anomalies = 0
    
    low_anomalies = False
    
    for i in range(len(sorted_outliers)):
        if i == 0 and sorted_outliers[i]:
            low_anomalies = True
        elif not sorted_outliers[i]:
            low_anomalies = False
    
        if sorted_outliers[i]:
            if low_anomalies:
                n_low_anomalies += 1
            else:
                n_large_anomalies += 1
    
    outliers[indeces_sorted_by_y_diff[np.min(not_outliers_indeces) - int(n_low_anomalies * (1 - p_low)): np.max(not_outliers_indeces) + int(n_large_anomalies * (1 - p_large)) + 1]] = False

    return df[index_name].values, outliers
        
        
def find_anomalies_by_clusters(
    data,
    p_large=1,
    p_low=1
):
    """Find metric anomalies in each cluster.
    
    The method finds differences between real metric and a fitted trend line, find points with the most extreme differences and marks them as anomalies. It skips clusters with less than 2 samples.
    
    Parameters
    ----------
    data : anomeda.DataFrame
        Object containing data to be analyzed
    p_large : float, [0, 1], default 1
        What part of anomalies which are higher than usual values needs to be returned. For example, if you set it to 0.7, then only 70% of the anomalies with the largest values will be returned. Default is 1.
    p_low : float, [0, 1], default 1
        What part of anomalies which are lower than usual values needs to be returned. For example, if you set it to 0.5, then only 50% of the anomalies with the lowest values will be returned. Default is 1.
        
    Returns
    -------
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
    
    data_pandas = data.to_pandas() # get pandas.DataFrame
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
        
        if len(ydata) >= 2:
            indeces, anomalies = find_anomalies(data.mod_data(unchanged_data_pandas.loc[filtered_data.index]), p_large=p_large, p_low=p_low)
            
            cluster = dict(zip(measures_names, measures_values))
            
            clusters_anomalies.append({
                'cluster': cluster,
                'indeces': indeces,
                'anomalies': anomalies
            })
        
    return clusters_anomalies


def explore_ts(
    data : 'anomeda.DataFrame | numpy.ndarray', 
    trend_fitting_conf : 'dict' = {'max_trends': 'auto', 'min_var_reduction': 0.75},
    filters : 'str' = None,
    breakdown : "'no' | 'all-clusters' | list[str]" = 'no',
    min_cluster_size : 'int | None' = None,
    max_cluster_size : 'int | None' = None,
    plot : 'bool' = True,
    df : 'bool' = True,
    verbose : 'bool' = True
):
    def resp_to_df(resp):
        flattened = []
        for row in resp:
            cluster, trends = row
            for t in trends:
                x_min, x_max, (slope, intercept), (cnt, mean, mae_var, y_sum) = trends[t]
                flattened.append((cluster, x_min, x_max, slope, intercept, cnt, mean, mae_var, y_sum))
        return pd.DataFrame(flattened, columns=['cluster', 'tren_start_dt', 'trend_end_dt', 'slope', 'intercept', 'cnt', 'mean', 'mae_var', 'sum'])
    
    
    def plot_trends(df):
        clusters = df['cluster'].drop_duplicates()
        cluster_c = {cluster: np.random.choice(list(CSS4_COLORS.keys())) for cluster in clusters}
        for c in clusters:
            df_tmp = df[df['cluster'] == c].sort_values(by='tren_start_dt')
            x_cluster = []
            y_trend_cluster = []
            for trend in df_tmp.iterrows():
                i, t = trend
                cluster = t['cluster']
                x = np.arange(t['tren_start_dt'], t['trend_end_dt'])
                x_from_0 = x - t['tren_start_dt']
                y_trend = x_from_0 * t['slope'] + t['intercept']

                x_cluster.append(x)
                y_trend_cluster.append(y_trend)

            x_cluster = np.concatenate(x_cluster)
            y_trend_cluster = np.concatenate(y_trend_cluster)
            plt.plot(x_cluster, y_trend_cluster, label=cluster, color=cluster_c[cluster])
        plt.legend()
    
    if min_cluster_size is None:
        min_cluster_size = -np.inf
    if max_cluster_size is None:
        max_cluster_size = np.inf

    if type(data) == DataFrame:
        
        index_name = data.get_index_name()
        if index_name is None:
            raise KeyError('not-None "index_name" must be set for anomeda.DataFrame object')
        
        metric_name = data.get_metric_name()
        if metric_name is None:
            raise KeyError('not-None "metric_name" must be set for anomeda.DataFrame object')
        
        agg_func = data.get_agg_func()
        if agg_func is None:
            raise KeyError('not-None "agg_func" must be set for anomeda.DataFrame object')
    
        if filters is not None:
            data_pandas = data.to_pandas().query(filters).reset_index()
        else:
            data_pandas = data.to_pandas().reset_index()

        if breakdown == 'all-clusters':
            measures_names = data.get_measures_names()
            if measures_names is None:
                raise KeyError('not-None "measures_names" must be set for anomeda.DataFrame object')
            measures_to_iterate = []
            for i in range(1, len(measures_names) + 1):
                for el in combinations(measures_names, i):
                    measures_to_iterate.append(list(el))
        elif type(breakdown) == list:
            measures_to_iterate = breakdown.copy()
        elif breakdown == 'no':
            measures_to_iterate = []
        else:
            raise ValueError('Breakdown must be either "all-clusters", list or "no"')
        
        measures_types = data.get_measures_types()
        if measures_types is not None and 'continuous' in measures_types:
            for measure in measures_types['continuous']:
                data_pandas[measure] = data.get_discretized_measures()[measure]
        
        res_values = []
        skipped_clusters = 0
        skipped_indeces = []
        total_clusters = 0
        columns_to_use = np.append(index_name, metric_name)

        query = 'total'
        ydata = data_pandas[columns_to_use].groupby(index_name).agg(agg_func)[metric_name].values
        xdata = np.arange(len(ydata)) 
        trends = extract_trends(
            xdata, ydata, 
            max_trends=trend_fitting_conf.get('max_trends'), 
            min_var_reduction=trend_fitting_conf.get('min_var_reduction')
        )
        res_values.append((query, trends))
        
        for measures_set in measures_to_iterate:
            for measures_values in data_pandas[measures_set].drop_duplicates().values:
                str_arr = []
                for (measure_name, measure_value) in zip(measures_names, measures_values):
                    str_arr.append('`' + measure_name + '`' + '==' + str(measure_value))
                query = ' and '.join(str_arr)
                
                yseries = data_pandas.query(query)[columns_to_use].groupby(index_name).agg(agg_func)[metric_name]
                yindex = yseries.index
                ydata = yseries.values
                xdata = np.arange(len(ydata)) 

                total_clusters += 1
                if ydata.shape[0] >= min_cluster_size and ydata.shape[0] <= max_cluster_size:
                    if ydata.shape[0] == 1:
                        res_values.append((
                            query, 
                            {
                                0: (
                                    yindex[0], yindex[0] + 1, 
                                    (1, ydata[0]), 
                                    (1, ydata[0], 0, ydata[0])
                                )
                            }
                        ))
                    else:
                        trends = extract_trends(
                            xdata, ydata, 
                            max_trends=trend_fitting_conf.get('max_trends'), 
                            min_var_reduction=trend_fitting_conf.get('min_var_reduction')
                        )
                        res_values.append((query, trends))
                else:
                    skipped_clusters += 1
                    skipped_indeces.append(yindex)
        if skipped_clusters > 0:
            skipped_indeces = np.unique(np.concatenate(skipped_indeces))

            ydata = data_pandas.loc[skipped_indeces, columns_to_use].groupby(index_name).agg(agg_func)[metric_name].values
            xdata = np.arange(len(ydata)) 

            query = 'skipped'

            if ydata.shape[0] == 1:
                res_values.append((
                    query, 
                    {
                        0: (
                            skipped_indeces[0], skipped_indeces[0] + 1, 
                            (1, ydata[0]), 
                            (1, ydata[0], 0, ydata[0])
                        )
                    }
                ))
            else:
                trends = extract_trends(
                    xdata, ydata, 
                    max_trends=trend_fitting_conf.get('max_trends'), 
                    min_var_reduction=trend_fitting_conf.get('min_var_reduction')
                )
                res_values.append((query, trends))
        
        if plot:
            plot_trends(res_values)
        if df:
            return resp_to_df(res_values)

    elif type(data) == np.ndarray:
        if filters is not None:
            raise ValueError('Filters may be passed only if provided data is anomeda.DataFrame')
        if breakdown is not None and breakdown != 'no':
            raise ValueError('Breakdown may be passed only if provided data is anomeda.DataFrame')
        y = data
        x = np.arange(len(y))
        query = 'total'
        res_values = []
        if y.shape[0] >= min_cluster_size and y.shape[0] <= max_cluster_size:
            if y.shape[0] == 1:
                res_values.append((
                    query, 
                    {
                        0: (
                            0, 1, 
                            (1, ydata[0]), 
                            (1, ydata[0], 0, ydata[0])
                        )
                    }
                ))
            else:
                trends = extract_trends(
                    x, y, 
                    max_trends=trend_fitting_conf.get('max_trends'), 
                    min_var_reduction=trend_fitting_conf.get('min_var_reduction')
                )
                res_values.append((query, trends))
        
        if plot:
            plot_trends(res_values)
        if df:
            return resp_to_df(res_values)
    else:
        raise ValueError('Data parameter must be either anomeda.DataFrame or numpy.ndarray with metric values')

    return trends

    