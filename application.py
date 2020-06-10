from flask import Flask, jsonify, request
from flask_expects_json import expects_json

from tracks import TracksDetector

app = Flask(__name__)

schema = {
    'type': 'object',
    'properties': {
        'track_length': {
            "title": "Track length in kilometers",
            'type': 'number'},
        'geozone1': {
            "title": "The source geozone polygon",
            "$ref": "http://geojson.org/schema/Polygon.json",
        },
        'geozone2': {
            "title": "The destination geozone polygon",
            "$ref": "http://geojson.org/schema/Polygon.json",
        }
    },
    'required': ['track_length', 'geozone1', 'geozone2']
}


@app.route('/tracks', methods=['POST'])
@expects_json(schema)
def register():
    try:
        geo_request = request.get_json()
        track_length = geo_request['track_length']
        source_geo_zone = geo_request['geozone1']
        destination_geo_zone = geo_request['geozone2']
        tracks_detector = TracksDetector(source_geo_zone, destination_geo_zone, track_length)
        detection_id = tracks_detector.detect()
    except Exception as e:
        return jsonify(dict(message=e)), 409

    return jsonify(dict(message=detection_id)), 200
