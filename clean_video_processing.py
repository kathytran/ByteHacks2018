# ---  Very Basic Instructions ---
# 1 - place a video clip in a bucket on your Google Cloud Storage and set permission to public
# 2 - run the code from the GCP cloud VM
# 3 - run the requirements.txt file (pip install -r requirements.txt)
# 4 - run video_processing.py clip_name bucket_name at the command prompt
#     this will create tmp folder and under a series of folders including faces_found and text_found
#     where it will store what it learned from your clip
# 5 - Don't forget to delete the clip (or remove public permission at the very least) and turn
#     you VM off!

# If you have ffmpeg issues try this:
# sudo apt-get install ffmpeg
from __future__ import absolute_import

import glob, os, sys, io, skvideo.io, argparse, math, datetime, ffmpy, shutil, wikipedia
from google.cloud import videointelligence
from google.cloud import vision
from google.cloud import storage
from google.cloud.vision import types
from PIL import Image, ImageDraw
import numpy as np

def init():
    # clean out directory structure
    os.system('rm -r tmp')

def analyze_labels(movie_to_process, bucket_name):
    path = 'gs://' + bucket_name + '/' + movie_to_process
    print(path)
    """ Detects labels given a GCS path. """
    video_client = videointelligence.VideoIntelligenceServiceClient()
    #result = video_client.annotate_video

    features = [videointelligence.enums.Feature.LABEL_DETECTION]
    print(features)

    mode = videointelligence.enums.LabelDetectionMode.SHOT_AND_FRAME_MODE
    config = videointelligence.types.LabelDetectionConfig(
        label_detection_mode=mode)
    context = videointelligence.types.VideoContext(
        label_detection_config=config)


    # #print(context)

    operation = video_client.annotate_video(
        path, features=features, video_context=context)
    print('\nProcessing video for label annotations:')

    result = operation.result(timeout=90)
    print('\nFinished processing.')

    frame_offsets = []

    # Process frame level label annotations
    frame_labels = result.annotation_results[0].frame_label_annotations
    for i, frame_label in enumerate(frame_labels):
        #if (frame_label.entity.description == 'person'):
        print('Frame label description: {}'.format(
            frame_label.entity.description))
        for category_entity in frame_label.category_entities:
            if (category_entity.description == 'person'):
                print('\tLabel category description: {}'.format(
                    category_entity.description))
                print(frame_label)
                # Each frame_label_annotation has many frames,
                # here we print information only about the first frame.
                #for frame in frame_label.frames:
                frame = frame_label.frames[0]
                time_offset = (frame.time_offset.seconds +
                               frame.time_offset.nanos / 1e9)
                print('\tFirst frame time offset: {}s'.format(time_offset))
                print('\tFirst frame confidence: {}'.format(frame.confidence))
                print('\n')
                frame_offsets.append(time_offset)
    return(sorted(set(frame_offsets)))


def extract_image_from_video(video_input, name_output, time_stamp):
    ret = "Error"
    try:
        ret = os.system("ffmpeg -i " + video_input + " -ss " + time_stamp + " -frames:v 1 " + name_output)
        # if all goes well FFMPEG will return 0
        return ret
    except ValueError:
        return("Oops! error...")

def crop_image(input_image, output_image, start_x, start_y, width, height):
    """Pass input name image, output name image, x coordinate to start croping, y coordinate to start croping, width to crop, height to crop """
    input_img = Image.open(input_image)
    # give the image some buffer space
    start_with_buffer_x = int(start_x - np.ceil(width/2))
    start_with_buffer_y = int(start_y - np.ceil(height/2))
    width_with_buffer = int(start_x + width  + np.ceil(width/2))
    height_with_buffer = int(start_y + height  + np.ceil(height/2))

    box = (start_with_buffer_x, start_with_buffer_y, width_with_buffer, height_with_buffer)
    output_img = input_img.crop(box)
    output_img.save(output_image +".png")
    return (output_image +".png")

def detect_face(face_file, max_results=4):
    # can you find a face and return coordinates
    client = vision.ImageAnnotatorClient()
    content = face_file.read()
    image = types.Image(content=content)

    # return coords of face
    return client.face_detection(image=image).face_annotations

def highlight_faces(image, faces):
    # Draws a polygon around the faces, then saves to output_filename.
    faces_boxes = []
    im = Image.open(image)
    draw = ImageDraw.Draw(im)

    for face in faces:
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        faces_boxes.append([box[0][0], box[0][1], box[1][0] - box[0][0], box[3][1] - box[0][1]])
    return (faces_boxes)

def annotate(path):
    """Returns web annotations given the path to an image."""
    client = vision.ImageAnnotatorClient()

    if path.startswith('http') or path.startswith('gs:'):
        image = types.Image()
        image.source.image_uri = path
    else:
        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

    web_detection = client.web_detection(image=image).web_detection

    return web_detection

def report(annotations, max_report=5):
    """Prints detected features in the provided web annotations."""
    names =  []
    if annotations.web_entities:
        print ('\n{} Web entities found: '.format(
            len(annotations.web_entities)))
        count = 0
        for entity in annotations.web_entities:
            print('Score      : {}'.format(entity.score))
            print('Description: {}'.format(entity.description))
            names.append(entity.description)
            count += 1
            if count >=max_report:
                break;
    return names

def get_stills(movie_to_process, bucket_name, timestamps_to_pull):
    video_location = 'https://storage.googleapis.com/' + bucket_name + '/' + movie_to_process
    storage_client = storage.Client()
    max_results = 3

    timestamps_to_pull_tmp = timestamps_to_pull + [x + 0.15 for x in timestamps_to_pull[:-1]] + [x - 0.15 for x in timestamps_to_pull[1:]]

    # clear out stills folder
    if len(timestamps_to_pull_tmp) > 0:
        # create directory structure
        os.system('mkdir tmp')
        os.system('mkdir tmp/faces_found')
        os.system('mkdir tmp/text_found')
        os.system('mkdir tmp/face_images')

        filepath = 'tmp/'

        # make stills
        cnt_ = 0
        for ttp in timestamps_to_pull_tmp:
            # get the still image at that timestamp
            time_stamp = str(datetime.timedelta(seconds=ttp))
            file = "still_"  + str(cnt_) + ".png"
            filePathAndName =  filepath + file
            print('filename: ' + time_stamp)
            ret = extract_image_from_video(video_input = video_location, name_output = filePathAndName, time_stamp = time_stamp)
            cnt_ += 1

            # find face on still image
            with open(filePathAndName, 'rb') as image:
                faces = detect_face(image, max_results)
                print('Found {} face{}'.format(
                    len(faces), '' if len(faces) == 1 else 's'))

                print('Looking for a face {}'.format(filePathAndName))
                # Reset the file pointer, so we can read the file again
                image.seek(0)
                faces_boxes = highlight_faces(filePathAndName, faces) #, filePathAndName)
                print('faces_boxes:', faces_boxes)

                if len(faces_boxes) > 0:
                    # image had a face

                    count = 0
                    for face_box in faces_boxes:
                        # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
                        saved_name = crop_image(filePathAndName, "tmp/face_images/" + file.split('.')[0] + str(count) + '_faces', face_box[0], face_box[1], face_box[2], face_box[3])
                        count += 1

                        # get actors name
                        potential_names = report(annotate(saved_name),2)
                        print('potential_names: ', potential_names)
                        # does the first have two words -  as in first and last name?
                        if (len(potential_names[0].split()) == 2):
                            # we have a winner
                            new_name = 'tmp/faces_found/' + potential_names[0] + '.png'
                            shutil.copy(saved_name,new_name)

                            # extract wiki bio
                            rez = wikipedia.page(potential_names[0]).content
                            # keep only intro paragraph
                            with open('tmp/text_found/' + potential_names[0] + ".txt", "w") as text_file:
                                text_file.write(rez.split('\n\n')[0] + " (Source: Wikipedia.com)")

BUCKET_NAME = ''
MOVIE_TO_PROCESS = ''

if __name__ == "__main__":
    if len(sys.argv) == 3:
        MOVIE_TO_PROCESS = sys.argv[1]
        BUCKET_NAME = sys.argv[2]

        # start things off clean
        print('Cleaning up...')
        init()
        print('Finding people...')
        # use video intelligence to find high probability of people being visible
        timestamps_to_pull = analyze_labels(MOVIE_TO_PROCESS, BUCKET_NAME)

        print('Processing people...')
        get_stills(MOVIE_TO_PROCESS, BUCKET_NAME, timestamps_to_pull)
        print('All done...')
    else:
        print('Wrong argument inputs')
