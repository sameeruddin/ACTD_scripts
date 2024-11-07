import os
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.detection import run_detector

# Label mapping for MegaDetector
DEFAULT_DETECTOR_LABEL_MAP = ['empty', 'animal', 'person', 'vehicle']

# paths for mega detector model and test image
current_path = os.getcwd()
md_model_path = os.path.join(current_path, 'md_model', 'md_v5b.0.0.pt')
test_image_path = os.path.join(current_path, 'Cat_Selfie.JPG')

# Inference testing with a camera trap image
try:
    image = vis_utils.load_image(test_image_path)
    model = run_detector.load_detector(md_model_path)
    result = model.generate_detections_one_image(image)
    detections_above_threshold = [d for d in result['detections'] if d['conf'] > 0.2]
    print('Found {} detections above threshold'.format(len(detections_above_threshold)))
    print('SUCCESS')
except OSError as oer:
    print(oer)
    raise


# Select the dictionary with the highest 'conf' value
highest_conf_detection = max(result['detections'], key=lambda x: x['conf'])
print(highest_conf_detection)

# verify the cropping part, if needed
#crop_image_list = vis_utils.crop_image(detections_above_threshold, image, confidence_threshold=0.2, expansion=0)
#crop_image_list[0].save('cropped.jpg')