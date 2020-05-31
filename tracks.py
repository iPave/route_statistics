from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import psycopg2
import numpy as np
from pandas import DataFrame
from pandas import *
from matplotlib import pyplot as plt

conn = psycopg2.connect(dbname='tracker-server', user='postgres', password='postgres', host='192.168.23.165')
cursor = conn.cursor()
cursor.execute(
    'select  tracker_id, measurement_time, geo_point from points p where  St_intersects(geo_point, ST_Buffer( ST_GeomFromText(\'POINT(38.975996 45.040216)\')::geography, 5000,\'quad_segs=8\')) order by tracker_id, measurement_time')
df = DataFrame(cursor.fetchall())
cursor.close()
conn.close()

df_grouped_by_id = df.groupby(0).get_group('18308412')
X = np.array(to_datetime(df_grouped_by_id[1]).astype(int) / 10 ** 9).reshape(-1, 1)
# clustering = OPTICS(max_eps=3600, min_samples=10).fit(time_series_array)
# print(clustering.labels_)

import time

start = time.time()

db = DBSCAN(eps=3600, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

end = time.time()
print(end - start)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

first_index = df_grouped_by_id.index[0]
labels_dataframe = DataFrame(db.labels_)
labels_dataframe.index = labels_dataframe.index + first_index
points_with_cluster = df_grouped_by_id.merge(labels_dataframe, left_index=True, right_index=True)

print(points_with_cluster)


# git remote add origin git@github.com:iPave/route_statistics.git
