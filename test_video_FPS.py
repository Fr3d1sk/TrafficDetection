# process a test video and prints FPS
import math
import random
import os
import cv2
import numpy as np
import time
import argparse  
import configparser
from ultralytics import YOLO
import draw_boxes as Boxes
from collections import defaultdict

# video_file = 'vid.mp4'
# video_file_out = 'vid_res.mp4'
# model = YOLO('best_m.pt')


def main(model_file, mode: str, create_output: bool = False, tracker: str = "bytetrack.yaml", 
         video_file: str = "vid.mp4", video_file_out: str = "vid_res.mp4"):
    """
    Main function to run the model on the video
    args:
        model: YOLO model
        mode: Mode to run the model in, either 'predict' or 'track'
        create_output: Whether to write to an output or not
        tracker: Tracker file to use for tracking
    """
    if os.path.exists(model_file):
        model = YOLO(model_file)
    else:
        if os.path.exists("best_m.pt"):
            print("Model not found, exporting the model")
            model = YOLO(model_file.split(".")[0] + ".pt")

            if not model_file.endswith(".pt"):
                model.export(format=model_file.split(".")[-1])
            model = YOLO(model_file)
        else:
            Exception("We need the pt model to derive the engine or onnx model")


    tracking = True if mode == "track" else False
    frame = 0
    average_fps = 0.0
    track_history = defaultdict(lambda: [])

    vid = cv2.VideoCapture(video_file)
    fps = int(round(vid.get(cv2.CAP_PROP_FPS)))
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
    out_w = w
    out_h = h  
    out = cv2.VideoWriter(video_file_out, fourcc, fps, (out_w, out_h), True)

    print( "image size: " , w, " x ", h)

    np.set_printoptions(suppress=True)    

    while True:
        ret, img = vid.read()
        if not ret:
            break
            
        start = time.time()
        
        if tracking:
            results = model.track(img, persist=True, tracker=tracker)
        else:
            results = model.predict(img, conf=0.25)
        end = time.time()
        
        frame = frame + 1
        fps = 1.0/(end-start)
        print("%.1f" % fps)
        
        average_fps += fps
        
        if create_output:
            if not tracking:
                Boxes.drawBoxes(img, results[0].boxes.data, score=True)
            else:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                img = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                out.write(img)
    
    print("Average FPS: %.1f" % (average_fps / frame) )

    vid.release()
    out.release()

    return {
        "FPS": average_fps / frame,
        "model": model,
        "mode": mode,
        "tracker": tracker,
        "create_output": create_output,
    }


if __name__ == "__main__":
    models = ["best_m.engine", "best_m.onnx", "best_m.pt"]
    trackers = ["bytetrack.yaml", "botsort.yaml"]

    results = []
    for model in models:
        for tracker in trackers:
            for mode in ["predict", "track"]:
                for create_output in [True, False]:
                    print(f"Running {model} with {tracker} in {mode} mode with output {create_output}")
                    file_str = f"{model.split('.')[-1]}_{tracker.split('.')[-1]}_{mode}.mp4"
                    results.append(main(model, mode, create_output, tracker, video_file_out=file_str))

    print(results)

    with open("results.txt", "w") as f:
        f.write(str(results))
