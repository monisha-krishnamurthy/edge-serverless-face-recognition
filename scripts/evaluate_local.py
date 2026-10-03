"""Evaluate the local detection/recognition pipeline without AWS calls."""
import argparse
import csv
import importlib.util
import io
from pathlib import Path
from unittest.mock import patch
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True, help='Course image ZIP')
    parser.add_argument('--labels', type=Path, required=True, help='Course Image,Results CSV')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation-results.csv')
    parser.add_argument('--limit', type=int, default=0, help='0 evaluates all labeled images')
    args = parser.parse_args()
    with args.labels.open(newline='', encoding='utf-8-sig') as source:
        rows = list(csv.DictReader(source))
    if not rows or not {'Image', 'Results'} <= rows[0].keys():
        parser.error('Labels must contain Image and Results columns and at least one row')
    if args.limit:
        rows = rows[:args.limit]

    # The recognition module creates a client at import. Replace it, and block
    # all client construction during inference; no queue operations are tested.
    with patch('boto3.client'):
        recognition = load_module('local_recognition', ROOT / 'face-recognition/fr_lambda.py')
    with patch('boto3.client', side_effect=RuntimeError('AWS disabled for local evaluation')):
        detection = load_module('local_detection', ROOT / 'face-detection/fd_component.py')
        detector = detection.FaceDetection()
        original_load = torch.load

        def safe_load(*a, **kw):
            kw['weights_only'] = True
            return original_load(*a, **kw)

        with patch('torch.load', side_effect=safe_load):
            recognizer = recognition.FaceRecognition(str(ROOT / 'face-recognition/resnetV1_video_weights.pt'))
        correct = errors = 0
        with zipfile.ZipFile(args.images) as archive, args.output.open('w', newline='') as out:
            members = {Path(n).name: n for n in archive.namelist() if not n.endswith('/')}
            writer = csv.DictWriter(out, fieldnames=['image', 'expected', 'predicted', 'correct', 'error'])
            writer.writeheader()
            for index, row in enumerate(rows, 1):
                name, expected = row['Image'].strip(), row['Results'].strip()
                predicted = error = ''
                try:
                    faces = detector.detect_faces_from_bytes(archive.read(members[name]))
                    if faces:
                        buffer = io.BytesIO()
                        faces[0].save(buffer, format='JPEG')
                        predicted = str(recognizer.recognize_face(buffer.getvalue()))
                    else:
                        predicted = 'No-Face'
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    errors += 1
                match = not error and predicted == expected
                correct += int(match)
                writer.writerow(dict(image=name, expected=expected, predicted=predicted, correct=match, error=error))
                print(f'[{index}/{len(rows)}] {name}: expected={expected}, predicted={predicted or "ERROR"}, match={match}', flush=True)
        print(f'\nExact-label matches: {correct}/{len(rows)} ({correct / len(rows):.1%}); errors: {errors}')
        print(f'Results: {args.output}')
        print('Local inference only; AWS delivery and deployment were not tested.')


if __name__ == '__main__':
    main()
