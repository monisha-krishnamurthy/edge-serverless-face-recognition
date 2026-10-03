import base64
import json
import os
from io import BytesIO

import boto3
import numpy as np
import torch
from PIL import Image as PILImage
from facenet_pytorch import InceptionResnetV1

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
RESPONSE_QUEUE_NAME = os.getenv("RESPONSE_QUEUE_NAME", "1233351056-resp-queue")
MODEL_WEIGHTS_PATH = "resnetV1_video_weights.pt"

sqs_client = boto3.client("sqs", region_name=AWS_REGION)

_queue_url_cache = {}
_recognizer = None


class FaceRecognition:
    def __init__(self, model_weights_path=MODEL_WEIGHTS_PATH):
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()
        saved_data = torch.load(model_weights_path, map_location="cpu")
        self.embedding_list = [torch.as_tensor(emb, dtype=torch.float32).view(-1) for emb in saved_data[0]]
        self.name_list = list(saved_data[1])

    @staticmethod
    def _tensor_from_bytes(face_bytes):
        with PILImage.open(BytesIO(face_bytes)) as face_pil:
            face_rgb = face_pil.convert("RGB")
            face_array = np.asarray(face_rgb, dtype=np.float32) / 255.0
        face_array = np.transpose(face_array, (2, 0, 1))
        return torch.from_numpy(face_array)

    def recognize_face(self, face_bytes):
        face_tensor = self._tensor_from_bytes(face_bytes).unsqueeze(0)
        with torch.no_grad():
            embedding = self.resnet(face_tensor).detach().squeeze(0)
        dist_list = []
        for emb_db in self.embedding_list:
            dist = torch.dist(embedding, emb_db).item()
            dist_list.append(dist)
        idx_min = dist_list.index(min(dist_list))
        return self.name_list[idx_min]


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceRecognition()
    return _recognizer


def get_queue_url(queue_name):
    if queue_name in _queue_url_cache:
        return _queue_url_cache[queue_name]
    resp = sqs_client.get_queue_url(QueueName=queue_name)
    url = resp["QueueUrl"]
    _queue_url_cache[queue_name] = url
    return url


def send_response_message(message_body):
    queue_url = get_queue_url(RESPONSE_QUEUE_NAME)
    sqs_client.send_message(QueueUrl=queue_url, MessageBody=json.dumps(message_body))


def _face_bytes_from_payload(face_payload):
    face_b64 = face_payload.get("face_base64")
    face_bytes = base64.b64decode(face_b64)
    return {
        "face_filename": face_payload.get("face_filename", "face.jpg"),
        "face_bytes": face_bytes,
    }


def _face_bytes_from_path(face_img_path):
    with open(face_img_path, "rb") as face_file:
        face_bytes = face_file.read()
    return {
        "face_filename": os.path.basename(face_img_path),
        "face_bytes": face_bytes,
    }


def _collect_faces(body):
    faces_payload = body.get("faces", [])
    if faces_payload:
        return [_face_bytes_from_payload(face_payload) for face_payload in faces_payload]
    if "face_img_path" in body:
        return [_face_bytes_from_path(body["face_img_path"])]
    return []


def handler_face_recognition(event, context):
    recognizer = get_recognizer()
    records = event.get("Records", [])
    processed = 0
    for record in records:
        body = record.get("body")
        if isinstance(body, str):
            body = json.loads(body)
        request_id = body.get("request_id")
        if not request_id:
            continue
        face_entries = _collect_faces(body)
        result_entries = []
        for face_entry in face_entries:
            match_name = recognizer.recognize_face(face_entry["face_bytes"])
            result_entries.append(
                {
                    "face_filename": face_entry["face_filename"],
                    "match_name": match_name,
                }
            )
        result_string = result_entries[0]["match_name"] if result_entries else ""
        response_payload = {"request_id": request_id, "result": result_string}
        if RESPONSE_QUEUE_NAME:
            send_response_message(response_payload)
        processed += 1
    return {
        "statusCode": 200,
        "body": json.dumps({"records_processed": processed}),
    }