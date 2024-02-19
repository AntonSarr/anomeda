from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.stats import beta
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import BayesianGaussianMixture
from matplotlib.colors import TABLEAU_COLORS

from anomeda.DataFrame import DataFrame, _to_discrete_values

INVALID_DATA_TYPE_MSG = "'{0}' argument must be anomeda.DataFrame, but {1} is provided"


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if an object is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


def __regularizer(x):
    """Penalize an alghorithm for choosing a trend breaking point so that so that x is the ratio of the dataset got in right of left part.   
    
    Used in in anomeda.__find_trend_breaking_point method. The more the value of the regularizer, the less the probabilty to choose a given point as a breaking point is."""
    return beta.pdf(x, 0.4, 0.4)


def __find_trend_breaking_point(x : 'numpy.ndarray[int]', y : 'numpy.ndarray[float]'):
    """Find the point where the trend is the most likely to be different from before.
    
    Parameters
    ----------
    x : numpy.ndarray[int]
        Indeces corresponding to time points. Must be an increasing array of integers. Some of the values may be omitted, e.g such x is OK: [0, 1, 5, 10]. 
    y : numpy.ndarray[float]
        Metric values corresponding to time points.
        
    Returns
    -------
    points_candidates : numpy.ndarray
        List of points sorted from the best candidate to the worst candidate.

    Examples
    --------
    >>> anomeda.utils.__find_trend_breaking_point([0, 1, 4, 5], [11.2, 10.4, 3.4, 3.1])
    4
    """
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


def extract_trends(
    x : 'numpy.ndarray[int]', 
    y : 'numpy.ndarray[float]', 
    max_trends : "int | 'auto'"='auto', 
    min_var_reduction : 'float[0, 1] | None'=0.5, 
    verbose : 'bool'=False
):
    """Fit and return automatocally fitted linear trends for given X and Y.

    The method can extract more than 1 trend if the metric significantly changed its behavior. 
    The sensibility of the method to identify trend changes are set by parameters "max_trends" and "min_var_reduction".
    
    Parameters
    ----------
    x : numpy.ndarray[int]
        Indeces corresponding to time points. Must be an increasing array of integers. Some of the values may be omitted, e.g such x is OK: [0, 1, 5, 10]. 
    y : numpy.ndarray[float]
        Metric values corresponding to time points.
    max_trends : int | "auto", default "auto"
        Number of trends to extract. If int, the method extracts defined amount of trends or less. Less trends may be extracted if no more trends were found or if the min_var_reduction is reached. It would mean taht the variance is already explained by that amount of trends. If 'auto', the method defines the number of trends automatically using min_var_reduction parameter. Default is 'auto'.
    min_var_reduction : float[0, 1] | None, default 0.5
        % of the variance of approximation error that must be reduced by adding trends comparing to the variance of the initial approximation with one trend. Values closer to 1 will cause extracting more trends, since more trends reduce the variance better. Values closer to 0 will cause producing less trends. If max_trends is set and reached, the extraction finishes regardless the value of the variance. If None, then not used. Default is 0.5.
    verbose : bool, default False
        If to produce more logs. Default False.
        
    Returns
    -------
    trends : dict
        Dict contains the extracted trends in the format
            {
                trend_id: (xmin_inc, xmax_exc, (trend_slope, trend_intersept), (n_samples, metric_mean, mae, metric_sum)), 
                ...
            }

    Examples
    --------
    >>> x = np.array([0, 1, 4, 5])
    >>> y = np.array([11.2, 10.4, 3.4, 3.1])
    >>> anomeda.extract_trends(x, y, max_trends=2)
    {
        0: (0, 4, (-0.7999999999999989, 11.2), (2, 10.8, 0.0, 21.6)), # trend 1, for date points from 0 (inc) to 4 (excl)
                                                                      # with slope -0.79 and intercept 11.2 
                                                                      # consisting of 2 samples, 
                                                                      # metric mean over date points is 10.8,
                                                                      # mae for fitting trend over date points is 0.0
                                                                      # sum for all metric values is 21.6
        1: (4, 6, (-0.2999999999999998, 4.6), (2, 3.25, 0.0, 6.5))
    }
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
        for i in trends.keys():
            xmin, xmax, (a, b), _ = trends[i]
            
            index_mask = (x >= xmin) & (x < xmax)
            dt = __find_trend_breaking_point(x[index_mask], y[index_mask])
            
            if dt is None:
                continue
    
            y_true1 = y[(x >= xmin) & (x < dt)]
            x1 = x[(x >= xmin) & (x < dt)]
            
            y_true2 = y[(x >= dt) & (x < xmax)]
            x2 = x[(x >= dt) & (x < xmax)]

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
                
                if trend_id == i:
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
                best_id = i
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


def compare_clusters(
    data: 'anomeda.DataFrame',
    period1: 'str',
    period2: 'str',
    clusters : 'list' = None
):
    """Compare metric values for 2 periods.
    
    The method generates pandas.DataFrame object with descriptions for two periods, for each cluster. 
    You can use it to identify the cluster or set of clusters caused the differences in the overall metric values between two periods.
    
    Parameters
    ----------
    data : anomeda.DataFrame
        Object containing data to be analyzed
    period1 : str
        Query to filter the first period. For example, 'dt < 10'.
    period2 s: str
        Query to filter the second period. For example, 'dt >= 10'.
    clusters : list, default None
        List of clusters to use in the comparison. The objects in the list are queries used in pandas.DataFrame.query. 
        If None, all clusters are used. Default is None.
        
    Returns
    -------
    output : pandas.DataFrame
        Object describing the clusters and the changes in the metric behavior between them. 

    Examples
    --------
    ```python
    anomeda.compare_clusters(
        data,
        period1='dt < 10',
        period2='dt >= 10',
        clusters=None # means all clusters
    )
    ```
    """
    if type(data) != DataFrame:
        raise TypeError(INVALID_DATA_TYPE_MSG.format('data', type(data)))
        
    if data.get_measures_names() is None:
        raise KeyError('not-None "measure_names" must be set for anomeda.DataFrame object')
    measures_names = data.get_measures_names()

    if clusters is not None:
        clusters = clusters.copy()
        for i in range(len(clusters)):
            if clusters[i] not in ['total', 'skipped']:
                clusters[i] = ' and '.join(sorted(clusters[i].split(' and '), key=lambda v: measures_names.index(v.split('==')[0].replace('`', ''))))

    
    new_df1 = data.to_pandas().reset_index().query(period1).set_index(data.get_index_name())
    if new_df1.shape[0] == 0:
        raise ValueError(f'Failed to find data for period {period1}')

    new_df2 = data.to_pandas().reset_index().query(period2).set_index(data.get_index_name())
    if new_df2.shape[0] == 0:
        raise ValueError(f'Failed to find data for period {period2}')

    data1 = data.replace_df(new_df1, keep_discretization=True)
    data2 = data.replace_df(new_df2, keep_discretization=True)

    clusters_info_1 = fit_trends(data1, trend_fitting_conf={'max_trends': 1}, breakdown='all-clusters', plot=False, df=True)
    clusters_info_2 = fit_trends(data2, trend_fitting_conf={'max_trends': 1}, breakdown='all-clusters', plot=False, df=True)
    
    if clusters is not None:
        clusters_info_1 = clusters_info_1[clusters_info_1['cluster'].isin(clusters)]
        clusters_info_2 = clusters_info_2[clusters_info_2['cluster'].isin(clusters)]
    
    joined_info = clusters_info_1.merge(clusters_info_2, on='cluster', how='outer')

    joined_info = joined_info[[
        'cluster',
        'trend_start_dt_x', 'trend_end_dt_x', 
        'trend_start_dt_y', 'trend_end_dt_y',
        'slope_x', 'slope_y',
        'intercept_x', 'intercept_y',
        'cnt_x', 'cnt_y',
        'mean_x', 'mean_y',
        'mae_var_x', 'mae_var_y',
        'sum_x', 'sum_y'
    ]]

    return joined_info


def find_anomalies(
    data: 'anomeda.DataFrame | (numpy.ndarray[int], numpy.ndarray[float])', 
    clusters: 'list' = None, 
    anomalies_conf : 'dict' = {'p_large': 1, 'p_low': 1, 'n_neighbors': 3},
    return_all_points : 'bool' = False,
    trend_fitting_conf : 'dict' = None
):
    """Find metric anomalies by looking for the most extreme metric changes.
    
    The method finds differences between real metric and a fitted trends, find points with extreme differences and marks them as anomalies. 
    You can find anomalies for automatically extracted clusters only if passing an anomeda.DataFrame.
    
    Parameters
    ----------
    data : anomeda.DataFrame | (numpy.ndarray[int], numpy.ndarray[float])
        Object containing metric values to be analyzed. Trends must be fitted for the object with anomeda.fit_trends() method if anomeda.DataFrame is passed.
    clusters : list, default None
        List of clusters to plot. The objects in the list are queries used in pandas.DataFrame.query.
    anomalies_conf : dict, default {'p_large': 1., 'p_low': 1., 'n_neighbors': 3}
        Dict containing 'p_large' and 'p_low' values. Both are float values between 0 and 1 corresponding to the % of the anomalies with largest and lowest metric values to be returned.
        For example, if you set 'p_low' to 0, no points with abnormally low metric values will be returned; if 0.5, then 50% of points with abnormally values will be returned, etc. 
        If some of the keys is not present or None, 1 is assumed.
        'n_neighbors' means number of neighbors parameter for sklearn.neighbors.LocalOutlierFactor class. The class is used to find points with abnormally large MAE. The more the parameter, typically, the less sensitive the model to anomalies.
    return_all_points : bool, default False
        If False, only anomaly points are returned. If True, all points with anomalies marks are returned. Default False.
    trend_fitting_conf : dict, default None
        Used only if data is not anomeda.DataFrame, but numpy arrays, to run anomeda.fit_trends method for them. 
        Parameters are similar to those you would pass to the argument anomeda.fit_trends(..., trend_fitting_conf=...).
       
    Returns
    -------
    res : pandas.DataFrame
        A DataFrame containing fields 'cluster', 'index', 'metric_value', 'fitted_trend_value', 'anomaly'.

    Examples
    --------
    >>> anomeda.fit_trends(data)
    >>> anomeda.find_anomalies(data)
    """
    if anomalies_conf is not None:
        p_large = anomalies_conf.get('p_large')
        if p_large is None:
            p_large = 1

        p_low = anomalies_conf.get('p_low')
        if p_low is None:
            p_low = 1

        n_neighbors = anomalies_conf.get('n_neighbors')
        if n_neighbors is None:
            n_neighbors = 3
    else:
        p_large = 1
        p_low = 1
        n_neighbors = 3
    
    if type(data) == DataFrame:
        if hasattr(data, '_trends'):
            df = data._trends.copy()
        if not hasattr(data, '_trends') or not hasattr(data, '_clusters'):
            raise NotFittedError('The anomeda.DataFrame instance has no fitted trends or clusters. Run anomeda.fit_trends() with save_trends=True and prefered parameters first')
    
        index_name = data.get_index_name()
        if index_name is None:
            raise KeyError('not-None "index_name" must be set for anomeda.DataFrame object')
        
        metric_name = data.get_metric_name()
        if metric_name is None:
            raise KeyError('not-None "metric_name" must be set for anomeda.DataFrame object')
        
        agg_func = data.get_agg_func()
        if agg_func is None:
            raise KeyError('not-None "agg_func" must be set for anomeda.DataFrame object')

        if data.get_measures_names() is None:
            raise KeyError('not-None "measure_names" must be set for anomeda.DataFrame object')
        measures_names = data.get_measures_names()
        
        if clusters is None:
            clusters = df['cluster'].drop_duplicates()

        if trend_fitting_conf is not None:
            raise ValueError('trend_fitting_conf is not used if anomeda.DataFrame is specified. Please remove the argument or set it to None to avoid confusion.')

        data_pandas = data.to_pandas().sort_index().reset_index()
        measures_types = data.get_measures_types()
        if measures_types is not None and 'continuous' in measures_types:
            for measure in measures_types['continuous']:
                data_pandas[measure] = data.get_discretized_measures()[measure]

        resp = []
        
        for c in clusters:

            # reorder features names in accordance to how they sorted in measures names
            if c not in ['total', 'skipped']:
                c = ' and '.join(sorted(c.split(' and '), key=lambda v: measures_names.index(v.split('==')[0].replace('`', ''))))

            df_tmp = df[df['cluster'] == c]

            if df_tmp.shape[0] == 0:
                raise ValueError(f'Failed to find cluster {c}. Make sure such a breakdown was set during trends fitting.')

            yindex = data._clusters[c]['index']
            ydata = data._clusters[c]['values']
            
            y_fitted_list = []
            y_diff_list = []
            x_labels = []
            for trend in df_tmp.iterrows():
                i, t = trend

                xmin, xmax = t['trend_start_dt'], t['trend_end_dt']
                slope, intercept = t['slope'], t['intercept']

                index_mask = (yindex >= xmin) & (yindex < xmax)
                cluster_indx = yindex[index_mask]
                y_fitted = slope * cluster_indx + intercept
                y_diff_list.append(ydata[index_mask] - y_fitted)
                x_labels.append(cluster_indx)
                y_fitted_list.append(y_fitted)

            x_labels = np.concatenate(x_labels)
            y_diff = np.concatenate(y_diff_list)
            y_fitted = np.concatenate(y_fitted_list)

            sorted_ind = np.argsort(x_labels)
            y_diff = y_diff[sorted_ind]
            y_fitted = y_fitted[sorted_ind]

            
            if len(ydata) == 1:
                outliers = np.array([False])
            else:
                clusterizator = LocalOutlierFactor(n_neighbors=max(min(len(ydata) - 1, n_neighbors), 1), novelty=False)
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

            res_df = pd.DataFrame({
                'index': yindex, 
                'metric_value': ydata, 
                'fitted_trend_value': y_fitted, 
                'anomaly': outliers
            })
            res_df['cluster'] = c
            res_df = res_df[['cluster', 'index', 'metric_value', 'fitted_trend_value', 'anomaly']]
            resp.append(res_df)

        resp = pd.concat(resp).reset_index(drop=True)
        if not return_all_points:
            resp = resp[resp['anomaly']]

        return resp
    elif type(data) == tuple and type(data[0]) == np.ndarray and type(data[1]) == np.ndarray:
        if clusters is not None:
            raise ValueError('Clusters may be specified only if passing an anomeda.DataFrame object')

        yindex, ydata = data
        
        y_fitted_list = []
        y_diff_list = []
        x_labels = []

        if trend_fitting_conf is not None:
            df_tmp = fit_trends(
                data=data, 
                df=True,
                trend_fitting_conf=trend_fitting_conf
            )
        else:
            df_tmp = fit_trends(data=data, df=True)

        resp = []

        for trend in df_tmp.iterrows():
            i, t = trend

            xmin, xmax = t['trend_start_dt'], t['trend_end_dt']
            slope, intercept = t['slope'], t['intercept']

            index_mask = (yindex >= xmin) & (yindex < xmax)
            cluster_indx = yindex[index_mask]
            y_fitted = slope * cluster_indx + intercept
            y_diff_list.append(ydata[index_mask] - y_fitted)
            x_labels.append(cluster_indx)
            y_fitted_list.append(y_fitted)

        x_labels = np.concatenate(x_labels)
        y_diff = np.concatenate(y_diff_list)
        y_fitted = np.concatenate(y_fitted_list)

        sorted_ind = np.argsort(x_labels)
        y_diff = y_diff[sorted_ind]
        y_fitted = y_fitted[sorted_ind]

        if len(ydata) == 1:
            outliers = np.array([False])
        else:
            clusterizator = LocalOutlierFactor(n_neighbors=max(min(len(ydata) - 1, n_neighbors), 1), novelty=False)
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

        res_df = pd.DataFrame({
            'index': yindex, 
            'metric_value': ydata, 
            'fitted_trend_value': y_fitted, 
            'anomaly': outliers
        })
        res_df['cluster'] = 'total'
        res_df = res_df[['cluster', 'index', 'metric_value', 'fitted_trend_value', 'anomaly']]
        resp.append(res_df)

        resp = pd.concat(resp).reset_index(drop=True)
        if not return_all_points:
            resp = resp[resp['anomaly']]

        return resp
    else:
        raise TypeError('Data parameter must be either anomeda.DataFrame or (numpy.ndarray[int], numpy.ndarray[float]) tuple')
        

def plot_trends(
    data: 'anomeda.DataFrame | pandas.DataFrame returned from anomeda.fit_trends()', 
    clusters: 'list' = None, 
    colors: 'dict' = None, 
    show_points=True
):
    """Plot fitted trends.
    
    Plot trends either from anomeda.DataFrame instance or using a response from anomeda.fit_trends().
    
    Parameters
    ----------
    data : anomeda.DataFrame | pandas.DataFrame returned from anomeda.fit_trends()
        Object containing trends to be plotted.
    clusters : list, default None
        List of clusters to plot. The objects in the list are queries used in pandas.DataFrame.query.
    colors : dict, default None
        Dictionary with a mapping between clusters and colors used in matplotlib.
    show_points : bool, default True
        Indicator if to show data points on plots.
        
    Returns
    -------
    None

    Examples
    --------
    >>> anomeda.fit_trends(data)
    >>> anomeda.plot_trends(data)
    """
    if type(data) == DataFrame:
        if hasattr(data, '_trends'):
            df = data._trends.copy()
        if not hasattr(data, '_trends') or not hasattr(data, '_clusters'):
            raise NotFittedError('The anomeda.DataFrame instance has no fitted trends or clusters. Run anomeda.fit_trends() with save_trends=True and prefered parameters first')
    
        measures_names = data.get_measures_names()
        if measures_names is None:
            raise KeyError('not-None "measures_names" must be set for anomeda.DataFrame object')
    elif type(data) == pd.DataFrame:
        df = data.copy()
    else:
        raise ValueError('"data" argument must be either anomeda.DataFrame or pandas.DataFrame returned by anomeda.fit_trends()')
    
    if df is None or len(df) == 0:
        return None 
    
    if clusters is None:
        clusters = df['cluster'].drop_duplicates()
    
    if colors is None:
        replace = len(clusters) >= len(TABLEAU_COLORS)
        cluster_c = dict(zip(clusters, np.random.choice(list(TABLEAU_COLORS.keys()), size=len(clusters), replace=replace)))
    
    for c in clusters:
        
        # reorder features names in accordance to how they sorted in measures names
        if type(data) == DataFrame: 
            if c not in ['total', 'skipped']:
                c = ' and '.join(sorted(c.split(' and '), key=lambda v: measures_names.index(v.split('==')[0].replace('`', ''))))

        df_tmp = df[df['cluster'] == c].sort_values(by='trend_start_dt')

        if df_tmp.shape[0] == 0:
            raise ValueError(f'Failed to find cluster {c}. Make sure such a breakdown was set during trends fitting.')
        
        x_cluster = []
        y_trend_cluster = []
        for trend in df_tmp.iterrows():
            i, t = trend
            cluster = t['cluster']
            x = np.arange(t['trend_start_dt'], t['trend_end_dt'])
            y_trend = x * t['slope'] + t['intercept']
            
            x_cluster.append(x)
            y_trend_cluster.append(y_trend)
        
        x_cluster = np.concatenate(x_cluster)
        y_trend_cluster = np.concatenate(y_trend_cluster)
        plt.plot(x_cluster, y_trend_cluster, label=cluster, color=cluster_c[cluster])
        
        if show_points and type(data) == DataFrame:
            if c not in ['total']:
                cluster_info = data._clusters[c]
                plt.scatter(cluster_info['index'], cluster_info['values'], color=cluster_c[cluster], marker='x')
        
    plt.legend()


def fit_trends(
    data : 'anomeda.DataFrame | (numpy.ndarray[int], numpy.ndarray[float])', 
    trend_fitting_conf : 'dict' = {'max_trends': 'auto', 'min_var_reduction': 0.75},
    save_trends : 'bool' = True,
    breakdown : "'no' | 'all-clusters' | list[str]" = 'no',
    min_cluster_size : 'int | None' = None,
    max_cluster_size : 'int | None' = None,
    plot : 'bool' = False,
    df : 'bool' = True,
    verbose : 'bool' = False
):
    """Fit trends for a time series.
    
    Fit trends using the data from an anomeda.DataFrame or an numpy.ndarray with metric values. 
    You can fit trends for automatically extracted clusters only if passing an anomeda.DataFrame. 
    If anomeda.DataFrame is passed and "save_trends" is True, it stores the trends into anomeda.DataFrame._trends attribute of the class every time the method is called. 
    The method returns a pandas.DataFrame describing trends and/or plots the trends.

    Parameters
    ----------
    data : anomeda.DataFrame | (numpy.ndarray, numpy.ndarray)
        Object containing metric values. If numpy.ndarray, a tuple of arrays corresponding to x (data points) and y (metric values) respectively.
    trend_fitting_conf : dict, default {'max_trends': 'auto', 'min_var_reduction': 0.75}
        Parameters for calling anomeda.extract_trends() function. It consists of 'max_trends' parameter, which is responsible for the maximum number of trends that you want to identify, and 'min_var_reduction' parameter, which describes what part of variance must be reduced by estimating trends. Values close to 1 will produce more trends since more trends reduce variance more signigicantly. Default is {'max_trends': 'auto', 'min_var_reduction': 0.75}.
    save_trends : 'bool', default True
        If False, return pandas.DataFrame with trends description without assigning it to the anomeda.DataFrame._trends.
    breakdown : 'no' | 'all-clusters' | list[str], default 'no'
        If 'no', the metric is grouped by date points only. If 'all-clusters', then all combinations of measures are used to create clusters for fitting trends within them. If list[str], then only combinations of measures specified in the list are used.
    min_cluster_size : int, default None
        Skip clusters whose total size among all date points is less than the value.
    max_cluster_size : int, default None
        Skip clusters whose total size among all date points is more than the value.
    plot : bool, default False
        Indicator if to plot fitted trends. anomeda.plot_trends is responsibe for plotting if the flag is True.
    df : bool, default True
        Indicator if to return a pandas.DataFrame containing fitted trends.
    verbose : bool, default False
        Indicator if to print additional output.
        
    Returns
    -------
    resp : pandas.DataFrame
        An object containing information about trends

    Examples
    --------
    >>> fitted_trends = anomeda.fit_trends(data, trend_fitting_conf={'max_trends': 3}, min_cluster_size=3, plot=True, df=True)
    """
    def resp_to_df(resp):
        flattened = []
        for row in resp:
            cluster, trends = row
            for t in trends:
                x_min, x_max, (slope, intercept), (cnt, mean, mae_var, y_sum) = trends[t]
                flattened.append((cluster, x_min, x_max, slope, intercept, cnt, mean, mae_var, y_sum))
        return pd.DataFrame(flattened, columns=['cluster', 'trend_start_dt', 'trend_end_dt', 'slope', 'intercept', 'cnt', 'mean', 'mae_var', 'sum'])
    
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
    
        data_pandas = data.to_pandas().sort_index().reset_index()

        if breakdown == 'all-clusters':
            measures_names = data.get_measures_names()
            if measures_names is None:
                raise KeyError('not-None "measures_names" must be set for anomeda.DataFrame object')
            measures_to_iterate = []
            for i in range(1, len(measures_names) + 1):
                for el in combinations(measures_names, i):
                    measures_to_iterate.append(list(el))
        elif type(breakdown) == list:
            measures_names = data.get_measures_names()
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
        clusters_storage = {}

        query = 'total'
        
        yseries = data_pandas[columns_to_use].groupby(index_name).agg(agg_func)[metric_name]
        yindex = yseries.index.values
        ydata = yseries.values

        if len(ydata) == 0:
            if save_trends:
                data._trends = {}
                data._clusters = {}
                data._trends_conf = {}
            if df:
                return None

        clusters_storage[query] = {'index': yindex, 'values': ydata}
        
        trends = extract_trends(
            yindex, ydata,
            max_trends=trend_fitting_conf.get('max_trends'), 
            min_var_reduction=trend_fitting_conf.get('min_var_reduction')
        )
        res_values.append((query, trends))
        
        for measures_set in measures_to_iterate:
            for measures_values in data_pandas[measures_set].drop_duplicates().values:
                str_arr = []
                for (measure_name, measure_value) in zip(measures_set, measures_values):
                    quote = '"' if type(measure_value) == str else ''
                    str_arr.append('`' + measure_name + '`' + '==' + quote + str(measure_value) + quote)
                query = ' and '.join(str_arr)
                
                yseries = data_pandas.query(query)[columns_to_use].groupby(index_name).agg(agg_func)[metric_name]
                yindex = yseries.index.values
                ydata = yseries.values

                clusters_storage[query] = {'index': yindex, 'values': ydata}

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
                            yindex, ydata, 
                            max_trends=trend_fitting_conf.get('max_trends'), 
                            min_var_reduction=trend_fitting_conf.get('min_var_reduction')
                        )
                        res_values.append((query, trends))
                else:
                    skipped_clusters += 1
                    skipped_indeces.append(yindex)
        if skipped_clusters > 0:
            skipped_indeces = np.unique(np.concatenate(skipped_indeces))

            query = 'skipped'

            yseries = data_pandas.loc[skipped_indeces, columns_to_use].groupby(index_name).agg(agg_func)[metric_name]
            yindex = yseries.index.values
            ydata = yseries.values

            clusters_storage[query] = {'index': yindex, 'values': ydata}

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
                    yindex, ydata, 
                    max_trends=trend_fitting_conf.get('max_trends'), 
                    min_var_reduction=trend_fitting_conf.get('min_var_reduction')
                )
                res_values.append((query, trends))
        
        res_values = resp_to_df(res_values)
        
        if save_trends:
            data._trends = res_values.copy()
            data._clusters = clusters_storage.copy()
            data._trends_conf = {
                'trend_fitting_conf': trend_fitting_conf.copy(),
                'breakdown': breakdown,
                'min_cluster_size': min_cluster_size,
                'max_cluster_size': max_cluster_size
            }
        if plot:
            plot_trends(data)
        if df:
            return res_values

    elif type(data) == tuple and type(data[0]) == np.ndarray and type(data[1]) == np.ndarray:
        if breakdown is not None and breakdown != 'no':
            raise ValueError('Breakdown may be passed only if provided data is anomeda.DataFrame')
        x, y = data

        if len(y) == 0:
            return None

        sorted_indx = np.argsort(x)
        x = x[sorted_indx]
        y = y[sorted_indx]
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
        
        res_values = resp_to_df(res_values)
        
        if plot:
            plot_trends(res_values)
        if df:
            return res_values
    else:
        raise TypeError('Data parameter must be either anomeda.DataFrame or (numpy.ndarray[int], numpy.ndarray[float]) tuple')

    return trends
    