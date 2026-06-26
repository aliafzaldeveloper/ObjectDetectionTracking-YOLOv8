"""
detect_track.py
Real-time object detection and tracking using a pre-trained YOLOv8 model
(ultralytics) with its built-in ByteTrack tracker (assigns a persistent ID
to each detected object across frames - the same role SORT/DeepSORT play).

Usage:
    Webcam:            python detect_track.py --source 0
    Video file:        python detect_track.py --source path/to/video.mp4
    Save output video: python detect_track.py --source video.mp4 --output out.mp4
    Single image:      python detect_track.py --source path/to/image.jpg --image

Press 'q' to quit the live preview window.
"""
import argparse

import cv2
from ultralytics import YOLO


def run_on_image(model, source, output, conf):
    results = model(source, conf=conf, verbose=False)
    annotated = results[0].plot()
    out_path = output or 'output_detected.jpg'
    cv2.imwrite(out_path, annotated)
    print(f"Saved annotated image to {out_path}")
    print("Detections:")
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls)]
        conf_score = float(box.conf)
        print(f"  {cls_name}: {conf_score:.2f}")


def run_on_video(model, source, output, conf, show):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # model.track() runs detection + assigns persistent tracking IDs
            results = model.track(frame, persist=True, conf=conf, verbose=False)
            annotated = results[0].plot()

            if show:
                cv2.imshow('Object Detection and Tracking (press q to quit)', annotated)

            if writer:
                writer.write(annotated)

            if show and (cv2.waitKey(1) & 0xFF == ord('q')):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames.")
    if output:
        print(f"Saved annotated video to {output}")


def main():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking with YOLOv8.")
    parser.add_argument('--source', default='0', help="0 for webcam, or path to a video/image file")
    parser.add_argument('--model', default='yolov8n.pt', help="YOLOv8 weights (auto-downloaded on first run)")
    parser.add_argument('--output', default=None, help="Path to save annotated output (video or image)")
    parser.add_argument('--conf', type=float, default=0.4, help="Confidence threshold")
    parser.add_argument('--image', action='store_true', help="Treat --source as a single image instead of video")
    parser.add_argument('--no-show', action='store_true', help="Don't open a live preview window (headless mode)")
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.image:
        run_on_image(model, args.source, args.output, args.conf)
        return

    source = int(args.source) if args.source.isdigit() else args.source
    run_on_video(model, source, args.output, args.conf, show=not args.no_show)


if __name__ == '__main__':
    main()
