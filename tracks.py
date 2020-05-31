from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import psycopg2
import numpy as np
from pandas import DataFrame
from pandas import *
from matplotlib import pyplot as plt
import time


def exclude_different_tracks(source_tracker_groups, dest_tracker_groups):
    unique_source_ids = source_tracker_groups[2].unique()
    unique_dest_ids = dest_tracker_groups[2].unique()
    diff_list = unique_source_ids - unique_dest_ids
    source_tracker_groups = source_tracker_groups[source_tracker_groups[2] != diff_list]
    dest_tracker_groups = source_tracker_groups[source_tracker_groups[2] != diff_list]
    return source_tracker_groups, dest_tracker_groups


def cluster_tracks(tracker_groups):
    for tracker_group in tracker_groups:
        X = np.array(to_datetime(tracker_group[1][1]).astype(int) / 10 ** 9).reshape(-1, 1)
        db = DBSCAN(eps=3600, min_samples=10).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        first_index = tracker_group[1].index[0]
        labels_dataframe = DataFrame(db.labels_)
        labels_dataframe.index = labels_dataframe.index + first_index
        points_with_cluster = tracker_group[1].merge(labels_dataframe, left_index=True, right_index=True)
        return points_with_cluster


conn = psycopg2.connect(dbname='tracker-server', user='postgres', password='postgres', host='192.168.23.165')
cursor = conn.cursor()
cursor.execute(
    'select  tracker_id, measurement_time, geo_point from points p where  St_intersects(geo_point, ST_Buffer( ST_GeomFromText(\'POINT(38.975996 45.040216)\')::geography, 5000,\'quad_segs=8\')) order by tracker_id, measurement_time')
df = DataFrame(cursor.fetchall())
cursor.close()
conn.close()

start = time.time()

cluster_tracks(df.groupby(0))
end = time.time()
print('Total clustering time is: %d ' % (end - start))
