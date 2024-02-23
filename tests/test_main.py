import pandas as pd
import numpy as np

import anomeda


def test_setters_and_getters():
    
    dummy_df = pd.DataFrame({'dt': [0, 1, 2, 3], 'a': [10, 20, 20, 30], 'b': [0.1, 0.2, 0.3, 0.2], 'metric': [10.5, 12.6, 8.3, 9.8]})
    
    index_name = ['dt']
    metric_name = 'metric'
    agg_func = 'sum'
    measures_names = ['a', 'b']
    measures_types = {
        'categorical': ['a'], 
        'continuous': ['b']
    }
    
    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=index_name,
        metric_name=metric_name,
        agg_func=agg_func
    )
    
    assert measures_names == anmd_df.get_measures_names()
    assert measures_types == anmd_df.get_measures_types()
    assert index_name == anmd_df.get_index_name()
    assert metric_name == anmd_df.get_metric_name()
    assert agg_func == anmd_df.get_agg_func()

    dummy_df.set_index('dt', inplace=True)

    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=index_name,
        metric_name=metric_name,
        agg_func=agg_func
    )

    assert index_name == anmd_df.get_index_name()

    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=None,
        metric_name=metric_name,
        agg_func=agg_func
    )

    assert index_name == anmd_df.get_index_name()

    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        metric_name=metric_name,
        agg_func=agg_func
    )

    assert index_name == anmd_df.get_index_name()
    

def test_set_reset_index():
    
    dummy_df = pd.DataFrame({'dt': [0, 1, 2, 3], 'a': [10, 20, 20, 30], 'b': [0.1, 0.2, 0.3, 0.2], 'metric': [10.5, 12.6, 8.3, 9.8]})
    
    index_name = ['dt']
    metric_name = 'metric'
    agg_func = 'sum'
    measures_names = ['a', 'b']
    measures_types = {
        'categorical': ['a'], 
        'continuous': ['b']
    }
    
    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=index_name,
        metric_name=metric_name,
        agg_func=agg_func
    )
    
    assert index_name == anmd_df.get_index_name()

    anmd_df.reset_index(inplace=True)
    assert anmd_df.get_index_name() is None

    anmd_df.set_index('dt', inplace=True)
    assert anmd_df.get_index_name() == index_name

    anmd_df.set_index('dt', inplace=True)
    assert anmd_df.get_index_name() == index_name

    anmd_df.set_index(['dt'], inplace=True)
    assert anmd_df.get_index_name() == index_name

    dummy_df.set_index('dt', inplace=True)

    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=index_name,
        metric_name=metric_name,
        agg_func=agg_func
    )

    assert anmd_df.get_index_name() == index_name

    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        metric_name=metric_name,
        agg_func=agg_func
    )

    assert anmd_df.get_index_name() == index_name
    

def test_to_pandas_method():
    
    dummy_df = pd.DataFrame({'dt': [0, 1, 2, 3], 'a': [10, 20, 20, 30], 'b': [0.1, 0.2, 0.3, 0.2], 'metric': [10.5, 12.6, 8.3, 9.8]})
    
    index_name = 'dt'
    metric_name = 'metric'
    agg_func = 'sum'
    measures_names = ['a', 'b']
    measures_types = {
        'categorical': ['a'], 
        'continuous': ['b']
    }
    
    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=index_name,
        metric_name=metric_name,
        agg_func=agg_func
    )
    
    assert dummy_df.set_index(index_name).equals(anmd_df.to_pandas())


def test_find_anomalies():
    
    dummy_df = pd.DataFrame({'dt': [0, 1, 2, 3], 'a': [10, 20, 20, 30], 'b': [0.1, 0.2, 0.3, 0.2], 'metric': [10.5, 12.6, 8.3, 9.8]})
    
    index_name = 'dt'
    metric_name = 'metric'
    agg_func = 'sum'
    measures_names = ['a', 'b']
    measures_types = {
        'categorical': ['a'], 
        'continuous': ['b']
    }
    
    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=index_name,
        metric_name=metric_name,
        agg_func=agg_func
    )

    anomeda.fit_trends(anmd_df)
    anomeda.find_anomalies(anmd_df)
    anomeda.find_anomalies(anmd_df, anomalies_conf={'p_large': 1, 'p_low': 1})
    anomeda.find_anomalies(anmd_df, clusters=['total'])

    anomeda.fit_trends(anmd_df, breakdown='all-clusters')
    anomeda.find_anomalies(anmd_df, clusters=['`a`==20'])
    anomeda.find_anomalies(anmd_df, return_all_points=True)

    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([11.2, 10.4, 10.2, 30.1, 10.2, 10., 11., 10.2, 10.9, 10.5, 11.1])

    anomeda.find_anomalies((x, y), trend_fitting_conf={'max_trends': 1, 'n_neighbors': 3})
    anomeda.find_anomalies((x, y), return_all_points=True, trend_fitting_conf={'max_trends': 1, 'n_neighbors': 3})


def test_extract_trends():

    x = np.arange(50)
    y = 0.5 * x + 10 +  3 * np.random.randn(50)

    anomeda.extract_trends(x=x, y=y, max_trends=10, min_var_reduction=0.3, verbose=True)
    anomeda.extract_trends(x=x, y=y, min_var_reduction=0., verbose=True)
    anomeda.extract_trends(x=x, y=y, min_var_reduction=1, verbose=True)
    anomeda.extract_trends(x=x, y=y, max_trends=1, verbose=True)
    anomeda.extract_trends(x=x, y=y, max_trends=1000, verbose=True)


def test_plot_trends():

    x = np.arange(50)
    y = 0.5 * x + 10 +  3 * np.random.randn(50)

    dummy_df = pd.DataFrame({'dt': [0, 1, 2, 3], 'a': [10, 20, 20, 30], 'b': [0.1, 0.2, 0.3, 0.2], 'metric': [10.5, 12.6, 8.3, 9.8]})
    
    index_name = 'dt'
    metric_name = 'metric'
    agg_func = 'sum'
    measures_names = ['a', 'b']
    measures_types = {
        'categorical': ['a'], 
        'continuous': ['b']
    }
    
    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=index_name,
        metric_name=metric_name,
        agg_func=agg_func
    )

    trends = anomeda.fit_trends(anmd_df)
    anomeda.plot_trends(anmd_df)
    anomeda.plot_trends(anmd_df, clusters=['total'])
    
    trends = anomeda.fit_trends((x, y))
    anomeda.plot_trends(trends)


def test_compare_clusters():

    dummy_df = pd.DataFrame({'dt': [0, 1, 2, 3], 'a': [10, 20, 20, 30], 'b': [0.1, 0.2, 0.3, 0.2], 'metric': [10.5, 12.6, 8.3, 9.8]})
    
    index_name = 'dt'
    metric_name = 'metric'
    agg_func = 'sum'
    measures_names = ['a', 'b']
    measures_types = {
        'categorical': ['a'], 
        'continuous': ['b']
    }
    
    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=index_name,
        metric_name=metric_name,
        agg_func=agg_func
    )

    res = anomeda.compare_clusters(anmd_df, period1='dt < 2', period2='dt >= 2')
    assert res.shape[0] > 0

    res = anomeda.compare_clusters(anmd_df, period1='dt < 2', period2='dt >= 2', clusters=['total'])
    assert res.shape[0] > 0

    res = anomeda.compare_clusters(anmd_df, period1='dt < 2', period2='dt >= 2', breakdown='all-clusters', clusters=['`a`==20'])
    assert res.shape[0] > 0


def test_empty_df():

    dummy_df = pd.DataFrame({'dt': [], 'a': [], 'b': [], 'metric': []})
    
    index_name = 'dt'
    metric_name = 'metric'
    agg_func = 'sum'
    measures_names = ['a', 'b']
    measures_types = {
        'categorical': ['a'], 
        'continuous': ['b']
    }
    
    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=index_name,
        metric_name=metric_name,
        agg_func=agg_func
    )

    anomeda.fit_trends(anmd_df)
    anomeda.fit_trends(anmd_df, breakdown='all-clusters')
    anomeda.plot_trends(anmd_df)


def test_one_element_df():

    dummy_df = pd.DataFrame({'dt': [1], 'a': [10], 'b': [0.1], 'metric': [10]})
    
    index_name = 'dt'
    metric_name = 'metric'
    agg_func = 'sum'
    measures_names = ['a', 'b']
    measures_types = {
        'categorical': ['a'], 
        'continuous': ['b']
    }
    
    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=index_name,
        metric_name=metric_name,
        agg_func=agg_func
    )

    anomeda.fit_trends(anmd_df)
    anomeda.fit_trends(anmd_df, breakdown='all-clusters')
    anomeda.plot_trends(anmd_df)


def test_two_elements_df():

    dummy_df = pd.DataFrame({'dt': [1, 2], 'a': [10, 20], 'b': [0.1, 0.2], 'metric': [10, 20]})
    
    index_name = 'dt'
    metric_name = 'metric'
    agg_func = 'sum'
    measures_names = ['a', 'b']
    measures_types = {
        'categorical': ['a'], 
        'continuous': ['b']
    }
    
    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=index_name,
        metric_name=metric_name,
        agg_func=agg_func
    )

    anomeda.fit_trends(anmd_df)
    anomeda.fit_trends(anmd_df, breakdown='all-clusters')
    anomeda.plot_trends(anmd_df)
