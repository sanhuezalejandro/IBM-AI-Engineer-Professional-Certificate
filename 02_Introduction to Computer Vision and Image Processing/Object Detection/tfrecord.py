import tensorflow as tf
import os 
import io
import PIL.Image
from object_detection.utils import dataset_util, label_map_util, config_util
from object_detection.utils.label_map_util import get_label_map_dict

def create_tf_record(images, annotations, label_map, image_path, output):
  # Create a train.record TFRecord file.
  with tf.python_io.TFRecordWriter(output) as writer:
    # Loop through all the training examples.
    for image_name in images:
      try:
        # Make sure the image is actually a file
        img_path = os.path.join(image_path, image_name)   
        if not os.path.isfile(img_path):
          continue
          
        # Read in the image.
        with tf.gfile.GFile(img_path, 'rb') as fid:
          encoded_jpg = fid.read()

        # Open the image with PIL so we can check that it's a jpeg and get the image
        # dimensions.
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
          raise ValueError('Image format not JPEG')

        width, height = image.size

        # Initialize all the arrays.
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for a in annotations[image_name]:
          if ("x" in a and "x2" in a and "y" in a and "y2" in a):
            label = a['label']
            xmins.append(a["x"])
            xmaxs.append(a["x2"])
            ymins.append(a["y"])
            ymaxs.append(a["y2"])
            classes_text.append(label.encode("utf8"))
            classes.append(label_map[label])
       
        # Create the TFExample.
        tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(image_name.encode('utf8')),
          'image/source_id': dataset_util.bytes_feature(image_name.encode('utf8')),
          'image/encoded': dataset_util.bytes_feature(encoded_jpg),
          'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        if tf_example:
          # Write the TFExample to the TFRecord.
          writer.write(tf_example.SerializeToString())
      except ValueError:
        print('Invalid example, ignoring.')
        pass
      except IOError:
        print("Can't read example, ignoring.")
        pass
    
    
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as PImage
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
DATA_PATH       = os.path.join(os.getcwd(),'content/data')
LABEL_MAP_PATH    = os.path.join(DATA_PATH, 'label_map.pbtxt')
EXPORTED_PATH   = os.path.join(os.getcwd(),'content/exported')
def displaydetectedobject(image):
# Load the labels
    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

# Load the model
    path_to_frozen_graph = os.path.join(EXPORTED_PATH, 'frozen_inference_graph.pb')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        #image = PImage.open('nayef.jpeg')
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=(12, 8))
        return plt.imshow(image_np),print('Wears'+' '+  category_index[np.squeeze(classes).astype(np.int32)[0]]['name']+ ' '),scores[0][1]