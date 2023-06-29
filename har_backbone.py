import cv2
import threading
import time
import os
import argparse
from har_utils import process_video

video_count = 0

def analyze_video(video_file):
    # Implement your logic to analyze the video here
    print(f'Analyzing video: {video_file}')
    process_video(video_file)
    print(f'Finished analyzing video: {video_file}')

    # Remove the video file after analyzing
    try:
        os.remove(video_file)
        print(f'Successfully deleted: {video_file}')
    except Exception as e:
        print(f'Error deleting file: {video_file}, {e}')

def main(args, har_args):
    global video_count

    if args.video_file:
        cap = cv2.VideoCapture(args.video_file)
    else:
        cap = cv2.VideoCapture(0)

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if not os.path.exists(args.buffer_dir):
        os.makedirs(args.buffer_dir)

    #i=0
    #while i<1:
    while True:
        # Get the default resolution of the camera
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_file = os.path.join(args.buffer_dir, f'output_{video_count}.mp4')
        out = cv2.VideoWriter(video_file, fourcc, args.fps_keep, (frame_width, frame_height))

        start_time = time.time()
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % round(input_fps / args.fps_keep) == 0:
                out.write(frame)

            frame_count += 1

            if time.time() - start_time > args.video_duration:
                break

        out.release()

        threading.Thread(target=analyze_video, args=(video_file,)).start()

        video_count += 1

        # Break loop if input is a video file
        if args.video_file:
            break
        #i+=1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture video and analyze it concurrently.')
    parser.add_argument('--buffer-dir', default='./', help='Output directory for raw videos to be proccesed')
    parser.add_argument('--fps-keep', type=float, default=10.0, help='Frames per second to keep')
    parser.add_argument('--video-duration', type=int, default=30, help='Video duration in seconds')
    parser.add_argument('--video-file', default=None, help='Pre-recorded video file to be processed')


    general_args, har_args = parser.parse_known_args()

    main(general_args, har_args)
