import warnings

import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings('ignore', category=ConvergenceWarning)


def _to_discrete_values(values, model=None):
    """Transform continious values into discrete ones.
    
    Parameters
    ----------
    values: : 1-dim numpy.array
        Numeric values to be mapped to discrete values
    
    model
        An object (sklearn object if preferable) which has fit_predict method taking the numeric numpy.array values of shape (n, 1) and returning mapped discrete values
    
    Returns
    -------
    labels : np.array
        Discrete labels of mapped values
    
    mapping_intervals : dict
        Dict containing unique discrete values (keys) and numeric range from initial data that is mapped into this interval (values). If None, sklearn.mixture.BayesianGaussianMixture(n_components=5, weight_concentration_prior=1e-5, max_iter=200) is used
    """
    
    if type(values) != np.ndarray:
        raise TypeError("Values to be discretized must be numpy.ndarray")
    
    if values.ndim != 1:
        raise TypeError("Values must have 1 dimension")
    
    n_components = min(int(values.shape[0] / 3) + 1, 5) # not more than 5 clusters, at least 3 samples in each one
    
    if model is None:
        model = BayesianGaussianMixture(n_components=n_components, weight_concentration_prior=1e-5, max_iter=200)

    labels = model.fit_predict(values.reshape(-1, 1))
    
    max_value = np.max(values)
    min_value = np.min(values)
    
    mapping_values = np.arange(min_value, max_value, (max_value - min_value) / len(values))
    mapped_values = model.predict(mapping_values.reshape(-1, 1))
    
    current_cluster = None
    mapping_intervals = {}
    interval_bounds = [None, None]
    
    for i in range(len(mapped_values)):
        if mapped_values[i] != current_cluster:
            
            if current_cluster is not None:
                mapping_intervals[current_cluster].append(interval_bounds)
                interval_bounds = [None, None]
            
            current_cluster = mapped_values[i]
            if current_cluster not in mapping_intervals:
                mapping_intervals[current_cluster] = []
            
            interval_bounds[0] = mapping_values[i]
            interval_bounds[1] = mapping_values[i]
        else:
            interval_bounds[1] = mapping_values[i]
            
    mapping_intervals[current_cluster].append(interval_bounds)
            
    return labels, mapping_intervals


class DataFrame:
    """Data to be processed by anomeda package.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Underlying data must be pandas.DataFrame object.
    measures_names : list or tuple
        List containing columns considered as measures in the data.
    measures_types : dict
        Dict containing 'categorical' and/or 'continuous' keys and list of measures as values. Continuous measures will be discretized automatically if not presented in discretized_measures parameter.
    discretized_measures : dict
        Dict containig name of the measure as key and array-like object containing discretized values of the measure of the same shape as original data. If measure is in 'continuous' list of measures_types parameter, it will be discretized automatically.
    index_name : str or list
        Columns to be considered as an index (usually a date or a timestamp)
    metric_name : str
        Column with a metric to be analyzed
    agg_func: str
        Way of aggregating metric_name by measures. Can be 'sum', 'avg' or callable compatible with pandas.DataFrame.groupby
    
    Examples
    --------
    ```python
    anmd_df = anomeda.DataFrame(
        data,
        measures_names=['class', 'dummy_measure', 'dummy_numeric_measure'],
        measures_types={
            'categorical': ['class', 'dummy_measure'], 
            'continuous': ['dummy_numeric_measure']
        },
        index_name='dt',
        metric_name='metric',
        agg_func='sum'
    )
    ```
    """
    
    def __init__(
        self, 
        data: pd.DataFrame,
        measures_names=None, 
        measures_types=None, 
        discretized_measures=None, 
        index_name=None, 
        metric_name=None, 
        agg_func='sum'
    ):
        
        self._data = data.copy()
        
        self.set_measures_names(measures_names)
        self.set_measures_types(measures_types)
        self.set_discretized_measures(discretized_measures)
        self.set_index_name(index_name)
        self.set_metric_name(metric_name)
        self.set_agg_func(agg_func)
        
    def __copy__(self):
        return self.copy()
        
    def aspandas(self):
        """Return a copy of a pandas.DataFrame object underlying the anomeda.DataFrame"""
        
        return self._data.copy()
    
    def mod_data(self, data: pd.DataFrame, inplace=False):
        """Replace the pandas.DataFrame object underlying the anomeda.DataFrame with a new one
        
        Parameters
        ----------
        data : pandas.DataFrame
            A new data object
        inplace : bool
            If True, then no new object will be returned. Otherwise, create and return a new anomeda.DataFrame
        """
        
        if inplace:
            self._data = data.copy()
        else:
            return DataFrame(
                data, 
                measures_names=self._measures_names, 
                measures_types=self._measures_types, 
                discretized_measures=None,
                index_name=self._index_name, 
                metric_name=self._metric_name, 
                agg_func=self._agg_func
            )
        
    def copy(self):
        """Return a copy of an anomeda.DataFrame object"""
        
        return DataFrame(
                data=self._data, 
                measures_names=self._measures_names, 
                measures_types=self._measures_types, 
                discretized_measures=self._discretized_measures,
                index_name=self._index_name, 
                metric_name=self._metric_name, 
                agg_func=self._agg_func
            )
    
    def get_discretization_mapping(self):
        """Return a dict with a mapping between discrete values and actual ranges of continous measures.
        
        In some cases, there may be more than one interval for each discrete values
        
        Examples
        --------
        ```python
        >>> anmd_df.get_discretization_mapping()
        
        {
            'dummy_numeric_measure': {
                0: [[0.08506988648110014, 0.982366623262143]],
                1: [[0.9855150328648835, 2.458970726947438]]
            }
        }
        ```
        """
        
        return self._discretized_measures_mapping
    
    def set_discretization_mapping(self, discretized_measures_mapping):
        """Set custom thresholds for discretization.
        
        Threshold must have the following format
        ```json
        {
            'measure_name': {
                discrete_value: [[threshold1, threshold2], [threshold3, threshold4], ...], 
                ...
                },
            ...
        }
        ```
        
        Parameters
        ----------
        discretized_measures_mapping : dict
            Dict with mapping between discrete value of the meause and corresponding continous values. 
            
        Examples
        --------
        ```python
        anmd_df.set_discretization_mapping({
            'dummy_numeric_measure': {
                0: [[0.01, 0.98]],
                1: [[0.99, 2.45]]
            }
        })
        ```
        """
        
        self.discretized_measures_mapping = discretized_measures_mapping
    
    def get_measures_names(self):
        """Return a list of columns considered as measures."""
        
        return self._measures_names
    
    def set_measures_names(self, measures_names):
        """Let anomeda.DataFrame object know what columns are measures. 
        
        Columns are picked from an underlying pandas.DataFrame object, so they must be present there.
        
        Parameters
        ----------
        measures_names : list of str
            List containing columns which will be considered as measures
        """
        if measures_names is not None:
            for name in measures_names:
                if name not in self._data.columns:
                    raise KeyError("All the names among measures_names must be present in the pandas.DataFrame underlying anomeda.DataFrame, but {} is cannot be found".format(name))
        self._measures_names = measures_names
    
    def get_measures_types(self):
        """Return the measures_types dict."""
        return self._measures_types
    
    def set_measures_types(self, measures_types: dict):
        """Set measures types. 
        
        Measure can be either 'categorical' or 'continous'. Types are used to clusterize the data properly.
        
        Parameters
        ----------
        measures_types : dict
            Dict containing 'continous' and/or 'categorical' keys and lists of measures as values
            
        Examples
        --------
        ```python
        anmd_df.set_measures_types({
            'continous': ['numeric_measure_1'],
            'categorical': ['measure_1']
        })
        ```
        """
        self._measures_types = measures_types
        if measures_types is not None:    
            if 'continuous' in measures_types:
                if '_discretized_measures' not in dir(self) or self._discretized_measures is None:
                    self._discretized_measures = {}
                    self._discretized_measures_mapping = {}
                for measure in measures_types['continuous']:
                    if measure not in self._discretized_measures:
                        self._discretized_measures[measure], \
                        self._discretized_measures_mapping[measure] = _to_discrete_values(self._data[measure].values)
    
    def get_discretized_measures(self):
        """Return discretized versions of continous measures."""
        return self._discretized_measures
    
    def set_discretized_measures(self, discretized_measures: dict):
        """Set custom discretization for continous measures.
        
        Parameters
        ----------
        discretized_measures : dict
            Dict containing discrete values of each measure in the format {'measure_name': [0, 1, 1, ...]}. Array of values must have same shape as original measure had.
        """
        if discretized_measures is None:
            if '_discretized_measures' not in dir(self):
                self._discretized_measures = discretized_measures
            return
        
        if type(discretized_measures) != dict:
            raise TypeError("discretized_measures argument must be dict in the format {'measure_name': [0, 1, 1, ...]}")
            
        for measure_name in discretized_measures.keys():
            if len(discretized_measures[measure_name]) != len(self._data):
                raise TypeError("Values for discretized_measures for anomeda.DataFrame must have the same length as the underlying pandas.DataFrame data")
        
        self._discretized_measures = discretized_measures
    
    def get_index_name(self):
        """Return the name of an index column."""
        
        return self._index_name
    
    def set_index_name(self, index_name):
        """Set a name of an index column.
        
        Parameters
        ----------
        index_name : str or list
            Column name or list of columns names containing index values. Must be present in an underling pandas.DataFrame object. If index is currenly present in measures list, you need to change the measures list first
        """
        if index_name is not None:
            if type(index_name) == list and len(index_name) == 0:
                raise KeyError("If index_name is list, it must have length >= 1")
        
            if self.get_measures_names() is not None:
                if type(index_name) != list:
                    index_name_arr = [index_name]
                else:
                    index_name_arr = index_name
                if len(set(index_name_arr).intersection(set(self.get_measures_names()))) > 0:
                    raise KeyError("index_name {} must not be present among measures_names. Change measures_names first".format(index_name))
                    
            self._data = self._data.set_index(index_name)
            self._index_name = index_name
            return
        else:
            if self._data.index.name is not None:
                self._index_name = self._data.index.name
                return
            
            if len(self._data.index.names) > 1:
                self._index_name = list(self._data.index.names)
                return
            
        self._index_name = None
    
    def get_metric_name(self):
        """Return the name of a metric column."""
        return self._metric_name
    
    def set_metric_name(self, metric_name):
        """Set the name of a metric to be analyzed.
        
        Parameters
        ----------
        metric_name : str
            Must be present among columns of an underlying pandas.DataFrame. If metric column is currently set as a measure, you need to change the list of measures first
        """
        if metric_name is not None and metric_name not in self._data.columns:
            raise KeyError("metric_name must be present among columns of the underlying pandas.DataFrame object")
        
        if self.get_measures_names() is not None and metric_name in self.get_measures_names():
            raise KeyError("metric_name not be present among measures_names. Change measures_names first")
            
        self._metric_name = metric_name
    
    def get_agg_func(self):
        """Return the function used to aggregate the metric by measures."""
        return self._agg_func
    
    def set_agg_func(self, agg_func: str):
        """Set a function to aggregate the metric by measures.
        
        Parameters
        ----------
        agg_func : str, callable
            Can be either 'sum', 'avg' or callable compatible with pandas.DataFrame.groupby
        """
        self._agg_func = agg_func
        
        