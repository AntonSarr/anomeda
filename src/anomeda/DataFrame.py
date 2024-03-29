import warnings

import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.exceptions import ConvergenceWarning
from enum import Enum


warnings.filterwarnings('ignore')
warnings.filterwarnings('default', module='anomeda', append=True)


class Freq(Enum):
    NANOSECOND  = {'compare': 1, 'pandas_freq': 'ns'}
    MICROSECOND = {'compare': 2, 'pandas_freq': 'us'}
    SECOND      = {'compare': 3, 'pandas_freq': 's'}
    MINUTE      = {'compare': 4, 'pandas_freq': 'min'}
    HOUR        = {'compare': 5, 'pandas_freq': 'h'}
    DAY         = {'compare': 6, 'pandas_freq': 'D'}
    MONTH       = {'compare': 7, 'pandas_freq': 'MS'}
    YEAR        = {'compare': 8, 'pandas_freq': 'YS'}

    def __lt__(self, other):
        return self.value['compare'] < other.value['compare']

    def __eq__(self, other):
        return self.value['compare'] == other.value['compare']

    def __ge__(self, other):
        return self.value['compare'] >= other.value['compare']


def _to_discrete_values(values, model=None):
    """Transform values from continious into discrete.
    
    Parameters
    ----------
    values : 1-dim numpy.ndarray
        Numeric values to be mapped to discrete values
    
    model : 'Object with fit_predict method defined (sklearn API based)'
        An object which has fit_predict method taking the numeric numpy.ndarray values of shape (n, 1) and returning mapped discrete 
        values. Default is sklearn.mixture.BayesianGaussianMixture(n_components=5, weight_concentration_prior=1e-5, max_iter=200)
    
    Returns
    -------
    labels : numpy.ndarray
        Discrete labels of mapped values
    
    mapping_intervals : dict
        Dict containing unique discrete values (keys) and numeric range from initial data that is mapped into this interval (values). 
        Lower bound is always including, higher bound is always excluding.
    """
    if type(values) != np.ndarray:
        raise TypeError("Values to be discretized must be numpy.ndarray")
    
    if values.ndim != 1:
        raise TypeError("Values must have 1 dimension")
    
    n_components = min(int(values.shape[0] / 3) + 1, 5) # not more than 5 descrete values, at least 3 samples corresponding to each one
    
    if model is None:
        model = BayesianGaussianMixture(n_components=n_components, weight_concentration_prior=1e-5, max_iter=200)

    labels = model.fit_predict(values.reshape(-1, 1))
    
    max_value = np.max(values)
    min_value = np.min(values)
    
    mapping_values = np.arange(
        min_value, 
        max_value + (max_value - min_value) / (len(values) - 1), 
        (max_value - min_value) / (len(values) - 1)
    )
    mapped_values = model.predict(mapping_values.reshape(-1, 1))
    
    current_cluster = None
    mapping_intervals = {}
    interval_bounds = [None, None]
    
    for i in range(len(mapped_values)):
        if mapped_values[i] != current_cluster: # when the mapped value changed, add new bounds to the mapping interval
            
            if current_cluster is not None:
                mapping_intervals[current_cluster].append(interval_bounds)
                interval_bounds = [None, None]
            
            current_cluster = mapped_values[i]
            if current_cluster not in mapping_intervals:
                mapping_intervals[current_cluster] = []
            
            interval_bounds[0] = mapping_values[i]
            if i + 1 < len(mapped_values): # make right bound excluding, not including
                interval_bounds[1] = mapping_values[i + 1]
            else:
                interval_bounds[1] = mapping_values[i] * 1.1
        else: # when the mapped value does not change, just move the right bound
            if i + 1 < len(mapped_values): # make right bound excluding, not including
                interval_bounds[1] = mapping_values[i + 1]
            else:
                interval_bounds[1] = mapping_values[i] * 1.1
            
    mapping_intervals[current_cluster].append(interval_bounds)
            
    return labels, mapping_intervals


def _extract_freq(values):
    """Define frequency of a time-series values"""
    x_datetime = pd.to_datetime(values)
    freq = Freq.YEAR
    for val in x_datetime:
        if pd.Timestamp(val).nanosecond > 0:
            freq = Freq.NANOSECOND
            break
        if pd.Timestamp(val).microsecond > 0:
            if freq > Freq.MICROSECOND:
                freq = Freq.MICROSECOND
                continue
        if pd.Timestamp(val).second > 0:
            if freq > Freq.SECOND:
                freq = Freq.SECOND
                continue
        if pd.Timestamp(val).minute > 0:
            if freq > Freq.MINUTE:
                freq = Freq.MINUTE
                continue
        if pd.Timestamp(val).hour > 0:
            if freq > Freq.HOUR:
                freq = Freq.HOUR
                continue
        if pd.Timestamp(val).day > 0:
            if freq > Freq.DAY:
                freq = Freq.DAY
                continue
        if pd.Timestamp(val).month > 0:
            if freq > Freq.MONTH:
                freq = Freq.MONTH
                continue
        if pd.Timestamp(val).year > 0:
            if freq > Freq.YEAR:
                freq = Freq.YEAR
                continue
    
    return freq


class DataFrame(pd.DataFrame):
    """Data to be processed by anomeda. The class inherits pandas.DataFrame.
    
    Please note that whenever the underlying pandas.DataFrame object is changed, you may need to apply the constructor again in order to keep some of the characteristics of the data consistent with the new object.
    
    Parameters
    ----------
    *args, **kwargs
        Parameters for initialization a pandas.DataFrame object. Other parameters must be passed as **kwargs only.
    measures_names : 'list | tuple' = []
        A list containing columns considered as measures. If None, your data is supposed to have no measures.
    measures_types : 'dict' = {}
        A dictionary containing 'categorical' and/or 'continuous' keys and list of measures as values. Continuous measures will be discretized automatically if not presented in discretized_measures parameter. If your data has any measures, you must provide its' types.
    discretized_measures_mapping : 'dict' = {}
        Custom dictionary with a mapping between a discrete value of the meauser and corresponding continous ranges. The lower bound must be including, the higher bound must be excluding. It uses the following format:
        ```json
        {
            'measure_name': {
                discrete_value_1: [[continuous_threshold_min_inc, continuous_threshold_max_excl], [...]],
                descrete_value_2: ... 
            }
        }
        ```
    discretized_measures : 'dict' = {}
        A dictionary containig names of the measures as keys and array-like objects containing customly discretized values of the measure. If not provided, continuous measures will be discretized automatically.
    index_name : 'str | list | None' = None
        An index column containg Integer or pandas.DatetimeIndex. If None, index is taken from the pandas.DataFrame.
    metric_name : 'str'
        A metric column.
    agg_func: '"sum" | "avg" | "count" | callable' = 'sum'
        Way of aggregating metric_name by measures. Can be 'sum', 'avg', 'count' or callable compatible with pandas.DataFrame.groupby.
    
    Examples
    --------
    ```python
    anmd_df = anomeda.DataFrame(
        df,
        measures_names=['dummy_measure_col', 'dummy_numeric_measure_col'],
        measures_types={
            'categorical': ['dummy_measure_col'], 
            'continuous': ['dummy_numeric_measure_col']
        },
        index_name='dt',
        metric_name='metric_col',
        agg_func='sum'
    )
    ```
    """
    
    def __init__(self, *args, **kwargs):
        filtered_kwargs = {}
        for arg in kwargs:
            if arg not in ['measures_names', 'measures_types', 'discretized_measures_mapping', 'discretized_measures', 'index_name', 'metric_name', 'agg_func']:
                filtered_kwargs[arg] = kwargs[arg]
        
        super().__init__(pd.DataFrame(*args, **filtered_kwargs).copy())

        index_name = kwargs.get('index_name')
   
        curr_indeces = list(filter(lambda x: x is not None, self.index.names))
        if index_name is None: 
            if len(curr_indeces) >= 1:
                self._index_name = curr_indeces
                try:
                    self.index = self.index.astype('str').astype('int64')
                except BaseException:
                    self.index = pd.to_datetime(self.index)
            else:
                self._index_name = None
            self.set_index_type()
        else:
            self.set_index(index_name, inplace=True)
        
        measures_names = kwargs.get('measures_names')
        if measures_names is None:
            measures_names = []
        self.set_measures_names(measures_names)
        
        measures_types = kwargs.get('measures_types')
        if measures_types is None:
            measures_types = {}
        self.set_measures_types(measures_types)
        
        discretized_measures = kwargs.get('discretized_measures')
        discretized_measures_mapping = kwargs.get('discretized_measures_mapping')

        if discretized_measures is not None and discretized_measures != {}\
              and discretized_measures_mapping is not None and discretized_measures_mapping != {}:
            raise NotImplementedError('As for now "discretized_measures" and "discretized_measures_mapping" cannot be passed at the same time. Please choose either mapping or measures.')
        
        self.set_discretized_measures(discretized_measures)
        self.set_discretization_mapping(discretized_measures_mapping)
        
        metric_name = kwargs.get('metric_name')
        self.set_metric_name(metric_name)
        
        agg_func = kwargs.get('agg_func')
        if agg_func is None:
            agg_func = 'sum'
        self.set_agg_func(agg_func)
        
    def __copy__(self):
        return self.copy_anomeda_df()
    
    def replace_df(
        self, 
        data : 'pandas.DataFrame', 
        inplace=False, 
        keep_clusters : 'bool' = False, 
        keep_trends : 'bool' = False, 
        keep_discretization : 'bool' = False
    ):
        """Replace the pandas.DataFrame content, underlying the anomeda.DataFrame, with a new one
        
        Parameters
        ----------
        data : pandas.DataFrame
            A new data object.
        inplace : bool
            If True, then no new object will be returned. Otherwise, create and return a new anomeda.DataFrame
        """
        new_obj = DataFrame(
            data, 
            measures_names=self._measures_names, 
            measures_types=self._measures_types, 
            discretized_measures_mapping=(self._discretized_measures_mapping if keep_discretization else None),
            discretized_measures=None,
            index_name=self._index_name, 
            metric_name=self._metric_name, 
            agg_func=self._agg_func
        )

        if keep_clusters:
            new_obj._clusters = self._clusters
        if keep_trends:
            new_obj._trends = self._trends
        
        if inplace:
            self = new_obj
        else:
            return new_obj
        
    def copy_anomeda_df(self):
        """Return a copy of an anomeda.DataFrame object"""
        new_obj = DataFrame(
            data=self, 
            measures_names=self._measures_names, 
            measures_types=self._measures_types, 
            discretized_measures=self._discretized_measures,
            index_name=self._index_name, 
            metric_name=self._metric_name, 
            agg_func=self._agg_func
        )
        
        if hasattr(self, '_trends'):
            new_obj._trends = self._trends.copy()
        if hasattr(self, '_clusters'):
            new_obj._clusters = self._clusters.copy()
        if hasattr(self, '_trends_conf'):
            new_obj._trends_conf = self._trends_conf.copy()

        return new_obj

    def to_pandas(self):
        return pd.DataFrame(self)
    
    def get_discretization_mapping(self):
        """Return a dict with a mapping between discrete values and actual ranges of continous measures.
        
        In some cases, there may be more than one interval for each discrete values
        
        Examples
        --------
        ```python
        >>> anmd_df.get_discretization_mapping()
        
        {
            'dummy_numeric_measure': {
                0: [[0.08506988648110014, 0.982366623262143]], # [[inc, exl)]
                1: [[0.9855150328648835, 2.458970726947438]] # [[inc, exl)]
            }
        }
        ```
        """
        return self._discretized_measures_mapping.copy()
    
    def set_discretization_mapping(self, discretized_measures_mapping, recalculate_measures=True):
        """Set custom thresholds for discretization.
        
        Parameters
        ----------
        discretized_measures_mapping : dict
            Dict with mapping between discrete value of the meause and corresponding continous values. Threshold must have the following format. 
            As you can see, several different ranges of continuous values may be mapped into the same descrete values if you want.
            The lower bound must be including, the higher bound must be excluding.
            ```json
            {
                'measure_name': {
                    discrete_value: [[continuous_threshold_min_inc, continuous_threshold_max_excl], [..., ...], ...], 
                    ...
                    },
                ...
            }
            ```
            
        Examples
        --------
        ```python
        anmd_df.set_discretization_mapping({
            'dummy_numeric_measure': {
                0: [[0.00, 0.05001], [0.95, 1.001]], # may correspond to "extreme" values; 0.05 are 1. are excluding bounds
                1: [[0.5, 0.94999]] # may correspond to "normal" values; 94999 is an excluding bound
            }
        })
        ```
        """
        if discretized_measures_mapping is not None:
            self._discretized_measures_mapping = discretized_measures_mapping.copy()

        if discretized_measures_mapping is not None and recalculate_measures:
            self._discretized_measures = {}
            for measure in discretized_measures_mapping.keys():
                tmp_measure_values = self[measure].copy()
                mapped_measure_values = pd.Series([None for i in range(tmp_measure_values.shape[0])], index=tmp_measure_values.index)
                for discrete_value in discretized_measures_mapping[measure].keys():
                    for interval in discretized_measures_mapping[measure][discrete_value]:
                        x_min, x_max = interval
                        mapped_measure_values[(tmp_measure_values >= x_min) & (tmp_measure_values < x_max)] = discrete_value
                self._discretized_measures[measure] = mapped_measure_values.values
    
    def get_measures_names(self):
        """Return a list of columns considered as measures."""
        return self._measures_names.copy()
    
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
                if name not in self.columns:
                    raise KeyError("All the names among measures_names must be present in the pandas.DataFrame underlying anomeda.DataFrame, but {} is cannot be found".format(name))
        self._measures_names = measures_names
    
    def get_measures_types(self):
        """Return the measures_types dict."""
        return self._measures_types.copy()
    
    def set_measures_types(self, measures_types : 'dict'):
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
                if not hasattr(self, '_discretized_measures') or self._discretized_measures is None:
                    self._discretized_measures = {}
                    self._discretized_measures_mapping = {}
                for measure in measures_types['continuous']:
                    if measure not in self._discretized_measures:
                        n_samples = self[measure].values.shape[0]
                        if n_samples >= 2:
                            self._discretized_measures[measure], \
                            self._discretized_measures_mapping[measure] = _to_discrete_values(self[measure].values)
                        elif n_samples == 1:
                            self._discretized_measures[measure] = [0]
                            self._discretized_measures_mapping[measure] = {
                                0: [[self[measure].values[0], self[measure].values[0]]]
                            }
                        else:
                            self._discretized_measures[measure] = []
                            self._discretized_measures_mapping[measure] = {}

    def get_discretized_measures(self):
        """Return discretized versions of continous measures."""
        return self._discretized_measures.copy()
    
    def set_discretized_measures(self, discretized_measures : 'dict'):
        """Set custom discretization for continous measures.
        
        Parameters
        ----------
        discretized_measures : dict
            Dict containing discrete values of each measure in the format {'measure_name': [0, 1, 1, ...]}. Array of values must have same shape as original measure had.
        """
        if discretized_measures is None:
            if not hasattr(self, '_discretized_measures'):
                self._discretized_measures = discretized_measures
            return
        
        if type(discretized_measures) != dict:
            raise TypeError("discretized_measures argument must be dict in the format {'measure_name': [0, 1, 1, ...]}")
            
        for measure_name in discretized_measures.keys():
            if len(discretized_measures[measure_name]) != len(self):
                raise TypeError("Values for discretized_measures for anomeda.DataFrame must have the same length as the underlying pandas.DataFrame data")
        
        self._discretized_measures = discretized_measures.copy()
    
    def get_index_name(self):
        """Return the name of an index column."""
        return self._index_name

    def set_index(self, *args, **kwargs):

        index_name = args[0] if len(args) > 0 else kwargs.get('index_name')
        curr_indeces = list(filter(lambda x: x is not None, self.index.names))

        if type(index_name) != list:
            index_name = [index_name]

        if index_name != curr_indeces:
            resp = super().reset_index()
            resp = super().set_index(*args, **kwargs)
            if resp is not None:
                try:
                    self.index = self.index.astype('str').astype('int64')
                except BaseException:
                    self.index = pd.to_datetime(self.index)
                return self.replace_df(resp, inplace=False)
            try:
                self.index = self.index.astype('str').astype('int64')
            except BaseException:
                self.index = pd.to_datetime(self.index)
            index_names = list(filter(lambda x: x is not None, self.index.names))
            if len(index_names) >= 1:
                self._index_name = index_names
            self.set_index_type()
        else:
            self._index_name = curr_indeces
            self.set_index_type()

    def reset_index(self, *args, **kwargs):
        resp = super().reset_index(*args, **kwargs)
        if resp is not None:
            return DataFrame(
                resp, 
                measures_names=self._measures_names, 
                measures_types=self._measures_types, 
                discretized_measures=None,
                index_name=None, 
                metric_name=self._metric_name, 
                agg_func=self._agg_func
            )
        self._index_name = None       

    def set_index_type(self):
        if len(self.index) == 0:
            self._index_is_numeric = None
            self._index_freq = None
            return
        
        try: 
            int(self.index[0])
            if type(self.index[0]) == np.datetime64:
                raise ValueError
            self._index_is_numeric = True
            self._index_freq = None
        except BaseException:
            try:
                pd.Timestamp(str(self.index[0]))
                self._index_is_numeric = False
                freq = _extract_freq(self.index)
                self._index_freq = freq.value['pandas_freq']
            except ValueError:
                raise ValueError('x values must be either convertable to Interger or compatible with pandas.Timestamp method')

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
        if metric_name is not None and metric_name not in self.columns:
            raise KeyError("metric_name must be present among columns of the underlying pandas.DataFrame object")
        
        if self.get_measures_names() is not None and metric_name in self.get_measures_names():
            raise KeyError("metric_name not be present among measures_names. Change measures_names first")
            
        self._metric_name = metric_name
    
    def get_agg_func(self):
        """Return the function used to aggregate the metric by measures."""
        return self._agg_func
    
    def set_agg_func(self, agg_func):
        """Set a function to aggregate the metric by measures.
        
        Parameters
        ----------
        agg_func: '"sum" | "avg" | "count" | callable' = 'sum'
            Can be "sum", "avg", "count" or callable compatible with pandas.DataFrame.groupby
        """
        if agg_func == 'sum': 
            self._agg_func = np.sum
        if agg_func == 'avg': 
            self._agg_func = np.mean
        if agg_func == 'count': 
            self._agg_func = len
        else:
            self._agg_func = agg_func