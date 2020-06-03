import time
from collections import defaultdict

import numpy as np
from pandas import *
from sklearn.cluster import DBSCAN


def exclude_different_tracks(source_tracker_groups, dest_tracker_groups):
    unique_source_ids = source_tracker_groups['tracker_id'].unique()
    unique_dest_ids = dest_tracker_groups['tracker_id'].unique()
    common_list = np.intersect1d(unique_source_ids, unique_dest_ids)
    source_tracker_groups = source_tracker_groups[source_tracker_groups['tracker_id'].isin(common_list)]
    dest_tracker_groups = dest_tracker_groups[dest_tracker_groups['tracker_id'].isin(common_list)]
    return source_tracker_groups, dest_tracker_groups


def cluster_tracks(tracker_groups):
    points_with_clusters = DataFrame(columns=['tracker_id', 'date', 'geography', 'cluster_id'])
    for tracker_group in tracker_groups:
        X = np.array(to_datetime(tracker_group[1]['date']).astype(int) / 10 ** 9).reshape(-1, 1)
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
        points_with_cluster = tracker_group[1].merge(labels_dataframe, left_index=True, right_index=True).rename(
            columns={0: "cluster_id"})
        points_with_clusters = points_with_clusters.append(points_with_cluster)

    return points_with_clusters


def build_tracks(source_clustered_tracks, dest_clustered_tracks, track_length, daily_length):
    source_clustered_tracks = source_clustered_tracks[source_clustered_tracks['cluster_id'] != -1]
    dest_clustered_tracks = dest_clustered_tracks[dest_clustered_tracks['cluster_id'] != -1]
    source_clustered_tracks, dest_clustered_tracks = exclude_different_tracks(source_clustered_tracks,
                                                                              dest_clustered_tracks)
    min_time_in_trip = (track_length / daily_length) * 0.7 * 24
    max_time_in_trip = (track_length / daily_length) * 1.3 * 24
    possible_pairs = defaultdict()
    for source_cluster in source_clustered_tracks.groupby(['tracker_id', 'cluster_id']):
        source_cluster_last_time = source_cluster[1].iloc[-1]['date']
        source_cluster_tracker_id = source_cluster[1].iloc[0]['tracker_id']
        source_cluster_id = source_cluster[1].iloc[0]['cluster_id']
        dest_tracker_clusters = dest_clustered_tracks.groupby('tracker_id').get_group(
            source_cluster_tracker_id).groupby('cluster_id')
        for dest_tracker_cluster in dest_tracker_clusters:
            dest_cluster_id = dest_tracker_cluster[1].iloc[0]['cluster_id']
            dest_cluster_first_time = dest_tracker_cluster[1].iloc[0]['date']
            if source_cluster_last_time < dest_cluster_first_time:
                source_cluster_last_time_sec = to_datetime(source_cluster_last_time).timestamp()
                dest_cluster_first_time_sec = to_datetime(dest_cluster_first_time).timestamp()
                diff = (dest_cluster_first_time_sec - source_cluster_last_time_sec) / 3600
                if diff > min_time_in_trip and diff < max_time_in_trip:
                    possible_pairs.setdefault(source_cluster_tracker_id, []).append(
                        [source_cluster_last_time_sec, dest_cluster_first_time_sec])

    return possible_pairs


# conn = psycopg2.connect(dbname='tracker-server', user='postgres', password='postgres', host='192.168.23.165')
# cursor = conn.cursor()
# cursor.execute(
#     'select  tracker_id, measurement_time, geo_point from points p where  St_intersects(geo_point, ST_Buffer( ST_GeomFromText(\'POINT(38.975996 45.040216)\')::geography, 5000,\'quad_segs=8\')) order by tracker_id, measurement_time')
# source_tracker_groups = DataFrame(cursor.fetchall())
# cursor.execute(
#     'select  tracker_id, measurement_time, geo_point from points p where  St_intersects(geo_point, ST_Buffer( ST_GeomFromText(\'POINT(65.534328 57.153033)\')::geography, 5000,\'quad_segs=8\')) order by tracker_id, measurement_time')
# dest_tracker_groups = DataFrame(cursor.fetchall())
# cursor.close()
# conn.close()

source_tracker_groups = pandas.read_csv("source.csv")
dest_tracker_groups = pandas.read_csv("dest1.csv")
start = time.time()

source_tracker_groups, dest_tracker_groups = exclude_different_tracks(source_tracker_groups, dest_tracker_groups)

source_clustered_tracks = cluster_tracks(source_tracker_groups.groupby('tracker_id'))
dest_clustered_tracks = cluster_tracks(dest_tracker_groups.groupby('tracker_id'))

daily_length = 600
track_length = 2600

print(build_tracks(source_clustered_tracks, dest_clustered_tracks, track_length, daily_length))

end = time.time()
print('Total clustering time is: %d ' % (end - start))
