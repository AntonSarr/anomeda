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
    assert anmd_df._index_is_numeric == True
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

    dummy_df = pd.DataFrame({
        'dt': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'], 
        'a': [10, 20, 20, 30], 
        'b': [0.1, 0.2, 0.3, 0.2], 
        'metric': [10.5, 12.6, 8.3, 9.8]
    })
    
    anmd_df = anomeda.DataFrame(
        dummy_df,
        measures_names=measures_names,
        measures_types=measures_types,
        index_name=index_name,
        metric_name=metric_name,
        agg_func=agg_func
    )
    assert anmd_df._index_is_numeric == False
    assert anmd_df._index_freq == 'D'
    

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
    
    dummy_df.set_index(index_name, inplace=True)
    try:
        dummy_df.index.astype('str').astype('int')
    except BaseException:
        dummy_df.index = pd.to_datetime(dummy_df.index)

    assert dummy_df.equals(anmd_df.to_pandas())


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
    anomeda.find_anomalies(anmd_df, breakdowns=['total'])

    anomeda.fit_trends(anmd_df, breakdowns='all')
    anomeda.find_anomalies(anmd_df, breakdowns=['`a`==20'])
    anomeda.find_anomalies(anmd_df, return_all_points=True)

    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([11.2, 10.4, 10.2, 30.1, 10.2, 10., 11., 10.2, 10.9, 10.5, 11.1])

    anomeda.find_anomalies((x, y), trend_fitting_conf={'max_trends': 1, 'n_neighbors': 3})
    anomeda.find_anomalies((x, y), return_all_points=True, trend_fitting_conf={'max_trends': 1, 'n_neighbors': 3})


def test_fit_trends():
    
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
    anomeda.fit_trends(anmd_df, breakdowns='all')
    anomeda.fit_trends(anmd_df, breakdowns='all', metric_propagate='zeros')
    anomeda.fit_trends(anmd_df, breakdowns='all', metric_propagate='ffil')

    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([11.2, 10.4, 10.2, 30.1, 10.2, 10., 11., 10.2, 10.9, 10.5, 11.1])

    anomeda.fit_trends((x, y))
    anomeda.fit_trends((x, y), metric_propagate='zeros')
    anomeda.fit_trends((x, y), metric_propagate='ffil')


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

    dummy_df = pd.DataFrame({'dt': ["2023-12-01", "2023-12-01", "2024-02-01", "2024-02-01"], 'a': [10, 20, 20, 30], 'b': [0.1, 0.2, 0.3, 0.2], 'metric': [10.5, 12.6, 8.3, 9.8]})
    
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
    anomeda.plot_trends(anmd_df, breakdowns=['total'])
    
    trends = anomeda.fit_trends((x, y))
    print(trends)
    anomeda.plot_trends(trends)


def test_compare_clusters():

    dummy_df = pd.DataFrame({'dt': ["2023-12-01", "2023-12-01", "2024-02-01", "2024-02-01"], 'a': [10, 20, 20, 30], 'b': [0.1, 0.2, 0.3, 0.2], 'metric': [10.5, 12.6, 8.3, 9.8]})
    
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

    res = anomeda.compare_clusters(anmd_df, period1='dt < "2024-01-01"', period2='dt >= "2024-01-01"')

    res = anomeda.compare_clusters(anmd_df, period1='dt < "2024-01-01"', period2='dt >= "2024-01-01"', breakdowns=['total'])

    res = anomeda.compare_clusters(anmd_df, period1='dt < "2024-01-01"', period2='dt >= "2024-01-01"', breakdowns=['`a`==20'])


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
    anomeda.fit_trends(anmd_df, breakdowns='all')
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
    anomeda.fit_trends(anmd_df, breakdowns='all')
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
    anomeda.fit_trends(anmd_df, breakdowns='all')
    anomeda.plot_trends(anmd_df)


def test_datetime_index():

    df = pd.DataFrame({
        'dt': ['2024-01-01', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-03', '2024-01-03'],
        'metric': [10, 20, -100, 30, 20, 10],
        'measure': ['A', 'A', 'A', 'A', 'B', 'B']
    })

    anomeda_df = anomeda.DataFrame(
        df,
        measures_names=['measure'], # columns represending measures or characteristics of your events
        measures_types={
            'categorical': ['measure']
        },
        index_name='dt',
        metric_name='metric', # dummy metric, always 1
        agg_func='sum' # function that is used to aggregate metric
    )

    anomeda.fit_trends(
        anomeda_df,
        metric_propagate=None,
        breakdowns='all'
    )
    anomeda.fit_trends(
        anomeda_df,
        metric_propagate='zeros',
        breakdowns='all'
    )
    anomeda.fit_trends(
        anomeda_df,
        metric_propagate='ffil',
        breakdowns='all'
    )

    anomeda.find_anomalies(anomeda_df)
    anomeda.plot_trends(anomeda_df)


def test_anomeda_df_without_measures():

    df = pd.DataFrame({
        'dt': ['2024-01-01', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-03', '2024-01-03'],
        'metric': [10, 20, -100, 30, 20, 10]
    })

    anomeda_df = anomeda.DataFrame(
        df,
        index_name='dt',
        metric_name='metric'
    )

    anomeda.fit_trends(anomeda_df)
    anomeda.find_anomalies(anomeda_df)
    anomeda.plot_trends(anomeda_df)