"""Test pre-trained RGB model on a single video.

Date: 01/15/18
Authors: Bolei Zhou and Alex Andonian

This script accepts an mp4 video as the command line argument --video_file
and averages ResNet50 (trained on Moments) predictions on num_segment equally
spaced frames (extracted using ffmpeg).

Alternatively, one may instead provide the path to a directory containing
video frames saved as jpgs, which are sorted and forwarded through the model.

ResNet50 trained on Moments is used to predict the action for each frame,
and these class probabilities are average to produce a video-level predction.

Optionally, one can generate a new video --rendered_output from the frames
used to make the prediction with the predicted category in the top-left corner.

"""

import os
import re
import cv2
import argparse
import functools
import subprocess
import sys
import json
import numpy as np
from PIL import Image
import moviepy.editor as mpy

import torch.optim
import torch.nn.parallel
from torch.nn import functional as F
from torch.autograd import Variable

from test_model import load_model, load_categories, load_transform


def extract_frames(video_file, num_frames=8):
    """Return a list of PIL image frames uniformly sampled from an mp4 video."""
    try:
        os.makedirs(os.path.join(os.getcwd(), 'frames'))
    except OSError:
        pass

    output = subprocess.Popen(['ffmpeg', '-i', video_file],
                              stderr=subprocess.PIPE).communicate()
    # Search and parse 'Duration: 00:05:24.13,' from ffmpeg stderr.
    re_duration = re.compile('Duration: (.*?)\.')
    duration = re_duration.search(str(output[1])).groups()[0]

    seconds = functools.reduce(lambda x, y: x * 60 + y,
                               map(int, duration.split(':')))
    rate = num_frames / float(seconds)

    output = subprocess.Popen(['ffmpeg', '-i', video_file,
                               '-vf', 'fps={}'.format(rate),
                               '-vframes', str(num_frames),
                               '-loglevel', 'panic',
                               'frames/%d.jpg']).communicate()
    frame_paths = sorted([os.path.join('frames', frame)
                          for frame in os.listdir('frames')])

    frames = load_frames(frame_paths)
    subprocess.call(['rm', '-rf', 'frames'])
    return frames


def load_frames(frame_paths, num_frames=8):
    """Load PIL images from a list of file paths."""
    frames = [Image.open(frame).convert('RGB') for frame in frame_paths]
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise ValueError('Video must have at least {} frames'.format(num_frames))

def process_scene(scene):
    frames = extract_frames(scene, args.num_segments)

    # Prepare input tensor
    data = torch.stack([transform(frame) for frame in frames])
    input_var = Variable(data.view(-1, 3, data.size(2), data.size(3)),
                        volatile=True)

    # Make video prediction
    logits = model(input_var)
    h_x = F.softmax(logits, 1).mean(dim=0).data
    probs, idx = h_x.sort(0, True)

    # Output the prediction.

    result = {}
    for i in range(0, 5):
        result[str(categories[idx[i]])] = float(probs[i])
    return result


    # Render output frames with prediction text.
    if args.rendered_output is not None:
        prediction = categories[idx[0]]
        rendered_frames = render_frames(frames, prediction)
        clip = mpy.ImageSequenceClip(rendered_frames, fps=4)
        clip.write_videofile(args.rendered_output)


def render_frames(frames, prediction):
    """Write the predicted category in the top-left corner of each frame."""
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames


if __name__ == "__main__":
    
    # options
    parser = argparse.ArgumentParser(description="test TRN on a single video")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video_file', type=str, default=None)
    group.add_argument('--frame_folder', type=str, default=None)
    parser.add_argument('--rendered_output', type=str, default=None)
    parser.add_argument('--num_segments', type=int, default=8)

    group.add_argument('--videos_dir',type=str, default=None)
    args = parser.parse_args()

    # Get dataset categories
    categories = load_categories()

    # Load RGB model
    model_id = 1
    model = load_model(model_id, categories)

    # Load the video frame transform
    transform = load_transform()

    all_scenes = list(map(os.path.abspath, map(lambda d: os.path.join(args.videos_dir, d), os.listdir(args.videos_dir))))
    all_scenes = list(filter(lambda f: f.endswith('mp4'),all_scenes))
    out_dir = os.path.abspath(os.path.join(args.videos_dir, '../scores_final'))
    os.makedirs(out_dir, exist_ok=True)

    # Obtain video frames
    for scene in all_scenes:
        print(scene.split('/')[-1])
        file_name = scene.split('/')[-1].split('.')[0]
        try:
            result = process_scene(scene)
            if(len(result.keys()) > 0):
                out_file = os.path.join(out_dir, file_name+'.json')
                json.dump(result, open(out_file, 'w'))
                print(result)
        except Exception as e:
            print(e)

