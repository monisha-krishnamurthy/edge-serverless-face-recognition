import base64
import json
import os
import threading
import io

import boto3
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image as PILImage

try:
    import awsiot.greengrasscoreipc as gg_ipc
    from awsiot.greengrasscoreipc.model import (
        SubscribeToTopicRequest,
        SubscriptionResponseMessage,
    )
    SubscribeToTopicStreamHandler = None
    try:
        from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
    except ImportError:
        GreengrassCoreIPCClientV2 = None
    try:
        from awsiot.greengrasscoreipc.client import SubscribeToTopicStreamHandler
    except (ImportError, AttributeError):
        try:
            from awsiot.greengrasscoreipc.model import StreamHandler as SubscribeToTopicStreamHandler
        except (ImportError, AttributeError):
            SubscribeToTopicStreamHandler = None
except ImportError:
    gg_ipc = None
    SubscribeToTopicStreamHandler = None
    GreengrassCoreIPCClientV2 = None


AWS_REGION = "us-east-1"
REQUEST_QUEUE_NAME = "1233351056-req-queue"
RESPONSE_QUEUE_NAME = "1233351056-resp-queue"
MQTT_TOPIC = "clients/1233351056-IoTThing"


class FaceDetection:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    def detect_faces_from_bytes(self, image_bytes: bytes):
        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        img = np.array(img)
        img = PILImage.fromarray(img)
        face, prob = self.mtcnn(img, return_prob=True, save_path=None)
        if face is None:
            return []

        face_img = face - face.min()
        face_img = face_img / face_img.max()
        face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()
        face_pil = PILImage.fromarray(face_img, mode="RGB")
        return [face_pil]


_detector = None
_queue_url_cache = {}
_processed_requests = set()


def get_detector():
    global _detector
    if _detector is None:
        _detector = FaceDetection()
    return _detector


def get_sqs_client():
    return boto3.client("sqs", region_name=AWS_REGION)


def get_queue_url(queue_name):
    if queue_name in _queue_url_cache:
        return _queue_url_cache[queue_name]
    sqs_client = get_sqs_client()
    resp = sqs_client.get_queue_url(QueueName=queue_name)
    url = resp["QueueUrl"]
    _queue_url_cache[queue_name] = url
    return url


def send_faces_to_sqs(request_id, original_filename, faces):
    queue_url = get_queue_url(REQUEST_QUEUE_NAME)

    faces_payload = []
    for idx, face_pil in enumerate(faces):
        buf = io.BytesIO()
        face_pil.save(buf, format="JPEG")
        buf.seek(0)
        face_bytes = buf.read()
        face_b64 = base64.b64encode(face_bytes).decode("utf-8")
        face_filename = f"{os.path.splitext(original_filename)[0]}_face{idx}.jpg"
        faces_payload.append(
            {"face_filename": face_filename, "face_base64": face_b64}
        )

    message = {
        "request_id": request_id,
        "original_filename": original_filename,
        "faces_count": len(faces_payload),
        "faces": faces_payload,
    }

    sqs_client = get_sqs_client()
    sqs_client.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(message),
    )


def send_response_to_sqs(request_id, result):
    queue_url = get_queue_url(RESPONSE_QUEUE_NAME)
    message = {
        "request_id": request_id,
        "result": result,
    }

    sqs_client = get_sqs_client()
    sqs_client.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(message),
    )


def process_message(payload_bytes, detector):
    global _processed_requests
    try:
        payload_str = payload_bytes.decode("utf-8")
        body = json.loads(payload_str)

        request_id = body.get("request_id")
        encoded = body.get("encoded")
        filename = body.get("filename", "input.jpg")

        if request_id in _processed_requests:
            print(f"Skipping duplicate request_id={request_id}")
            return
        _processed_requests.add(request_id)
        if len(_processed_requests) > 1000:
            _processed_requests.clear()

        print(f"Received message: request_id={request_id}, filename={filename}")

        if not request_id or not encoded:
            print("Missing request_id or encoded data")
            return

        image_bytes = base64.b64decode(encoded)
        print(f"Decoded image: {len(image_bytes)} bytes")
        
        faces = detector.detect_faces_from_bytes(image_bytes)
        print(f"Detected {len(faces)} face(s)")

        if not faces:
            send_response_to_sqs(request_id, "No-Face")
            print(f"Sent No-Face response for {request_id}")
            return

        send_faces_to_sqs(request_id, filename, faces)
        print(f"Sent {len(faces)} face(s) to request queue for {request_id}")
    except Exception as e:
        print(f"Error processing message: {e}")
        import traceback
        traceback.print_exc()


_base_class = object
if SubscribeToTopicStreamHandler is not None:
    try:
        _base_class = SubscribeToTopicStreamHandler
    except (AttributeError, TypeError):
        _base_class = object

class PubSubMessageHandler(_base_class):
    def __init__(self):
        if _base_class != object:
            super().__init__()
        self.detector = get_detector()

    def on_stream_event(self, event):
        try:
            if hasattr(event, 'binary_message') and event.binary_message:
                payload_bytes = event.binary_message.message
            elif hasattr(event, 'json_message') and event.json_message:
                payload_bytes = json.dumps(event.json_message.message).encode('utf-8')
            elif hasattr(event, 'message'):
                payload_bytes = event.message.payload if hasattr(event.message, 'payload') else event.message
            else:
                print(f"Unknown event format: {type(event)}")
                return
            
            process_message(payload_bytes, self.detector)
        except Exception as e:
            print(f"Error in on_stream_event: {e}")
            import traceback
            traceback.print_exc()

    def on_stream_error(self, error):
        print(f"Stream error: {error}")
        return True

    def on_stream_closed(self):
        print("Stream closed")


def run():
    print(f"Starting FaceDetection component...")
    print(f"Subscribing to topic: {MQTT_TOPIC}")
    
    detector = get_detector()
    print("MTCNN detector initialized")
    
    try:
        if GreengrassCoreIPCClientV2 is not None:
            print("Using GreengrassCoreIPCClientV2")
            ipc_client = GreengrassCoreIPCClientV2()
            
            def on_message(event):
                try:
                    if hasattr(event, 'binary_message') and event.binary_message:
                        payload_bytes = event.binary_message.message
                    elif hasattr(event, 'json_message') and event.json_message:
                        payload_bytes = json.dumps(event.json_message.message).encode('utf-8')
                    else:
                        print(f"Unknown event format: {type(event)}")
                        return
                    process_message(payload_bytes, detector)
                except Exception as e:
                    print(f"Error processing: {e}")
            
            _, operation = ipc_client.subscribe_to_topic(
                topic=MQTT_TOPIC,
                on_stream_event=on_message
            )
            print(f"Subscribed to {MQTT_TOPIC} successfully")
        else:
            print("Using GreengrassCoreIPC v1 client")
            ipc_client = gg_ipc.connect()
            
            request = SubscribeToTopicRequest()
            request.topic = MQTT_TOPIC
            
            handler = PubSubMessageHandler()
            operation = ipc_client.new_subscribe_to_topic(handler)
            operation.activate(request).result()
            print(f"Subscribed to {MQTT_TOPIC} successfully")
        
        print("Waiting for messages...")
        threading.Event().wait()
    except Exception as e:
        print(f"Error in run: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run()
