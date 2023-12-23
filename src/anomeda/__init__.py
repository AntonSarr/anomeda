# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("numpy", "pandas", "sklearn")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies


import anomeda.DataFrame
from anomeda.utils import *

__doc__ = """
anomeda - a Python library for exploratory analysis of anomalies and its causes
===============================================================================

The library helps to identify important metric changes and quickly find clusters in data which changed the trend of the metric or caused the anomaly.

Main features:
-- Identify dramatic change of a metric by looking at the changes of metric values over time
-- Find clusters in data which caused the anomalies 
-- Find clusters in data which changed the trend of the data
-- Compare arbitrary periods of time and look for clusters of data which changed the average values or variance of a metric the most
-- Find clusters in data which contribute an increasing or decreasing trend to an overall trend of a metric
-- Find clusters in data which have the highest variance of a metric
"""