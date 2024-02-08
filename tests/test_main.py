import pandas as pd
import numpy as np

import anomeda


def test_setters_and_getters():
    
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
    
    assert measures_names == anmd_df.get_measures_names()
    assert measures_types == anmd_df.get_measures_types()
    assert index_name == anmd_df.get_index_name()
    assert metric_name == anmd_df.get_metric_name()
    assert agg_func == anmd_df.get_agg_func()


def test_aspandas_method():
    
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
    
    assert dummy_df.set_index(index_name).equals(anmd_df.aspandas())


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

    anomeda.find_anomalies(anmd_df, p_large=0.9, p_low=0.9)
    anomeda.find_anomalies(anmd_df, p_large=0.9, p_low=0.9, trend='adjusted-linear', n_trends=3)
    anomeda.find_anomalies_by_clusters(anmd_df, p_large=0.9, p_low=0.9)


def test_extract_trends():

    x = np.arange(50)
    y = 0.5 * x + 10 +  3 * np.random.randn(50)

    anomeda.extract_trends(x=x, y=y, n_trends=10, var_cutoff=0.7, verbose=True)
    anomeda.extract_trends(x=x, y=y, var_cutoff=0.7, verbose=True)
    anomeda.extract_trends(x=x, y=y, var_cutoff=0., verbose=True)
    anomeda.extract_trends(x=x, y=y, var_cutoff=1, verbose=True)
    anomeda.extract_trends(x=x, y=y, n_trends=1, verbose=True)
    anomeda.extract_trends(x=x, y=y, n_trends=1000, verbose=True)

