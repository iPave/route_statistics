import json
import time
import uuid
from collections import defaultdict
from datetime import datetime

import numpy as np
import psycopg2
from pandas import *
from psycopg2.extras import execute_values
from sklearn.cluster import DBSCAN


class TracksDetector:
    DAILY_RUN = 600

    def __init__(self, source_geo_zone: dict, destination_geo_zone: dict, track_length: int):
        self.source_geo_zone = source_geo_zone
        self.destination_geo_zone = destination_geo_zone
        self.track_length = track_length

    def detect(self):

        request_start = time.time()
        conn = psycopg2.connect(dbname='tracker-server', user='postgres', password='postgres', host='192.168.23.165')
        cursor = conn.cursor()
        # argslist = [('1', '2', str(datetime.utcfromtimestamp(1561092382)), str(datetime.utcfromtimestamp(1562092382))),
        #             ('1', '2', str(datetime.utcfromtimestamp(1561092282)), str(datetime.utcfromtimestamp(1561092312)))]
        # execute_values(cursor, "insert into geo_zone_tracks (id, tracker_id, time_from, time_to) values  %s", argslist)

        response = self.db_request(cursor, self.source_geo_zone, self.destination_geo_zone)
        request_end = time.time()
        print('Total request time is: %d ' % (request_end - request_start))
        start = time.time()
        source_tracker_groups = response[response['group_id'] == 1]
        dest_tracker_groups = response[response['group_id'] == 2]
        source_clustered_tracks = self.cluster_tracks(source_tracker_groups.groupby('tracker_id'))
        dest_clustered_tracks = self.cluster_tracks(dest_tracker_groups.groupby('tracker_id'))
        end = time.time()
        print('Total clustering time is: %d ' % (end - start))

        tracks = self.build_tracks(source_clustered_tracks, dest_clustered_tracks, self.track_length, self.DAILY_RUN)
        print(tracks)
        if not tracks:
            return ''
        tracks_uuid = uuid.uuid4()
        self.save_tracks(cursor, tracks_uuid, tracks)
        cursor.close()
        return tracks_uuid

    def exclude_different_tracks(self, source_tracker_groups: DataFrame, dest_tracker_groups: DataFrame):
        unique_source_ids = source_tracker_groups['tracker_id'].unique()
        unique_dest_ids = dest_tracker_groups['tracker_id'].unique()
        common_list = np.intersect1d(unique_source_ids, unique_dest_ids)
        source_tracker_groups = source_tracker_groups[source_tracker_groups['tracker_id'].isin(common_list)]
        dest_tracker_groups = dest_tracker_groups[dest_tracker_groups['tracker_id'].isin(common_list)]
        return source_tracker_groups, dest_tracker_groups

    def cluster_tracks(self, tracker_groups: DataFrame) -> DataFrame:
        points_with_clusters = DataFrame(columns=['tracker_id', 'measurement_time', 'point', 'cluster_id'])
        for tracker_group in tracker_groups:
            X = np.array(to_datetime(tracker_group[1]['measurement_time']).astype(int) / 10 ** 9).reshape(-1, 1)
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

    def cluster_tracks_threaded(self, proc_id: str, tracker_groups: DataFrame,
                                returned_clusters: DataFrame) -> DataFrame:
        points_with_clusters = DataFrame(columns=['tracker_id', 'measurement_time', 'point', 'cluster_id'])
        for tracker_group in tracker_groups:
            print(proc_id)
            X = np.array(to_datetime(tracker_group[1]['measurement_time']).astype(int) / 10 ** 9).reshape(-1, 1)
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

        returned_clusters[proc_id] = points_with_clusters

    def db_request(self, cursor, source_geo_zone, destination_geo_zone) -> DataFrame:
        cursor.execute(
            "with first_geozone_points as (select 1 as group_id, tracker_id, measurement_time, point from points p where St_intersects(point, ST_SetSRID(st_geomfromgeojson(%s::text), 4326)) order by tracker_id, measurement_time), second_geozone_points as (select 2 as group_id, tracker_id, measurement_time, point from points p where St_intersects(point, ST_SetSRID(st_geomfromgeojson(%s::text), 4326)) order by tracker_id, measurement_time) select * from first_geozone_points where tracker_id in (select tracker_id from  second_geozone_points) union (select * from second_geozone_points where tracker_id in (select tracker_id from  first_geozone_points))",
            [json.dumps(source_geo_zone), json.dumps(destination_geo_zone)])
        geo_zone_data_frame = DataFrame(cursor.fetchall(),
                                        columns=['group_id', 'tracker_id', 'measurement_time', 'point'])

        return geo_zone_data_frame

    def build_tracks(self, source_clustered_tracks: DataFrame, dest_clustered_tracks: DataFrame, track_length: int,
                     daily_length: int) -> defaultdict:
        source_clustered_tracks = source_clustered_tracks[source_clustered_tracks['cluster_id'] != -1]
        dest_clustered_tracks = dest_clustered_tracks[dest_clustered_tracks['cluster_id'] != -1]
        source_clustered_tracks, dest_clustered_tracks = self.exclude_different_tracks(source_clustered_tracks,
                                                                                       dest_clustered_tracks)
        min_time_in_trip = (track_length / daily_length) * 0.7 * 24
        max_time_in_trip = (track_length / daily_length) * 1.3 * 24
        possible_pairs = defaultdict()
        for source_cluster in source_clustered_tracks.groupby(['tracker_id', 'cluster_id']):
            source_cluster_last_time = source_cluster[1].iloc[-1]['measurement_time']
            source_cluster_tracker_id = source_cluster[1].iloc[0]['tracker_id']
            source_cluster_id = source_cluster[1].iloc[0]['cluster_id']
            dest_tracker_clusters = dest_clustered_tracks.groupby('tracker_id').get_group(
                source_cluster_tracker_id).groupby('cluster_id')
            for dest_tracker_cluster in dest_tracker_clusters:
                dest_cluster_id = dest_tracker_cluster[1].iloc[0]['cluster_id']
                dest_cluster_first_time = dest_tracker_cluster[1].iloc[0]['measurement_time']
                if source_cluster_last_time < dest_cluster_first_time:
                    source_cluster_last_time_sec = to_datetime(source_cluster_last_time).timestamp()
                    dest_cluster_first_time_sec = to_datetime(dest_cluster_first_time).timestamp()
                    diff = (dest_cluster_first_time_sec - source_cluster_last_time_sec) / 3600
                    if diff > min_time_in_trip and diff < max_time_in_trip:
                        possible_pairs.setdefault(source_cluster_tracker_id, []).append(
                            [source_cluster_last_time_sec, dest_cluster_first_time_sec])

        return possible_pairs

    def save_tracks(self, cursor, uuid, tracks):
        argslist = []
        for tracker_id, grouped_tracks in tracks.items():
            for track in grouped_tracks:
                argslist.append(
                    (str(uuid), tracker_id, str(datetime.utcfromtimestamp(track[0])),
                     str(datetime.utcfromtimestamp(track[1]))))

        print(execute_values(cursor, "insert into geo_zone_tracks (id, tracker_id, time_from, time_to) values  %s",
                             argslist))
