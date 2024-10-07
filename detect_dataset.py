"""
---> Generate a dataset from sorted and reviewed species data to train YOLO based object detection model
        1. Load the list of input directory for reading filenames of the images and their labels/classes
        2. locate the image path and pass it to the MegaDetector for identifying the position of the animal in the image
        3. Build the output pipeline to write the bounding box and category information in the TXT file for each image in the output directory.
        4. The image and txt files are moved to the output directory and they are suitable to train a YOLO model
"""

# importing system libraries
import os
from datetime import datetime
import torch
import multiprocessing as mp
import numpy as np
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.utils import ct_utils as ct_utils
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox
import humanfriendly
from tqdm import tqdm
import shutil

# Declaring variables
current_path = os.getcwd() # working directory path
input_dir_path = os.path.join(current_path, 'input') # iput directory path
output_dir_path = os.path.join(current_path, 'output') # output directory path
tbd_dir_path = os.path.join(output_dir_path, 'TBD') # images not detected by megadetector - to be reviewed by Human
#model_path = os.path.join(current_path, 'md_model', 'md_v5a.0.0.pt') # mega detector v5a model directory path
model_path = os.path.join(current_path, 'md_model', 'md_v5b.0.0.pt') # mega detector v5b model directory path
BATCH_SIZE = 256 # number of images used to load and process in a single job request

# Label mapping for MegaDetector
DEFAULT_DETECTOR_LABEL_MAP = ['animal', 'person', 'vehicle']

# Variables used by the megadetector model
IMAGE_SIZE = 1280  # image size used in training
STRIDE = 64
DETECTION_THRESHOLD = 0.2


def load_dir_list(data_dir_path):
    """ Function to load the directory contents into the lists """
    total_files_count = 0
    tmp_img_path_list = []
    tmp_labels_list = []
    dataset_list = os.listdir(data_dir_path)
    if dataset_list:
        for category in dataset_list:
            #print(category)
            category_dir_path = os.path.join(data_dir_path, category)
            category_dir_list = os.listdir(category_dir_path)
            if category_dir_list:
                for image in category_dir_list:
                    #print(image)
                    image_path = os.path.join(category_dir_path, image)
                    tmp_img_path_list.append(image_path)
                    tmp_labels_list.append(category)
                    total_files_count = total_files_count + 1
            else:
                print(f'{category_dir_path} is an empty directory !!')
    else:
        print(f'{input_dir_path} is empty!!')
        raise
    return tmp_img_path_list, tmp_labels_list, total_files_count


def create_directory(dir_path):
    """ Function to create a directory, if does not exists """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print(f'{os.path.basename(dir_path)}: {dir_path} directory created')


def create_classes(spec_labels_list, output_dir_path):
    """ create the classes.txt and the category directories in the output directory based on the list of species"""
    categories_list = set(spec_labels_list)
    class_file_path = os.path.join(output_dir_path, 'classes.txt')
    with open(class_file_path, 'w') as class_file:
        for category in categories_list:
            class_file.write(category)
            create_directory(os.path.join(output_dir_path, category))
            class_file.write('\n')
    return list(categories_list)


def create_job_batches(img_path_lists, num_gpus):
    """ Function to generate the job batches based on the number of gpus """
    jobs_list = []
    procs_list = []
    total_number = len(img_path_lists)
    if (num_gpus == 0) or (num_gpus == 1):
        initial_value = 0
        counter = total_number
        while counter>=0:
            current_value = initial_value + BATCH_SIZE
            if current_value > total_number:
                #print(f'{initial_value}:')
                jobs_list.append(f'{initial_value}:')
            else:
                #print(f'{initial_value}:{current_value}')
                jobs_list.append(f'{initial_value}:{current_value}')
            procs_list.append('0')
            initial_value = current_value
            counter = counter - BATCH_SIZE
        #print(jobs_list, procs_list)
        return jobs_list, procs_list
    else:
        gpu_value = num_gpus - 1
        initial_value = 0
        counter = total_number
        while counter>=0:
            current_value = initial_value + BATCH_SIZE
            if current_value > total_number:
                #print(f'{initial_value}:')
                jobs_list.append(f'{initial_value}:')
            else:
                #print(f'{initial_value}:{current_value}')
                jobs_list.append(f'{initial_value}:{current_value}')
            if gpu_value == 0:
                procs_list.append(gpu_value)
                gpu_value = num_gpus - 1
            else:
                procs_list.append(gpu_value)
                gpu_value = gpu_value - 1
            initial_value = current_value
            counter = counter - BATCH_SIZE
        #print(jobs_list, procs_list)
        return jobs_list, procs_list


def create_label_file(cat_output_dir_path, item, output, category):
    """ Create TXT file for saving the category and the bounding box values """
    head,tail = (os.path.basename(item)).split('.')
    output_txt_file = os.path.join(cat_output_dir_path, f'{head}.txt')
    with open(output_txt_file, 'w') as txt_file:
        txt_file.write(output)
    try:
        shutil.copy(item, os.path.join(cat_output_dir_path, f'{head}.JPG'))
    except shutil.Error as ser:
        print(ser)
        raise


def generate_dataset(img_path_lists, spec_labels_list, device_id, output_dir_path, categories_set):
    """ Function to find the animals in the batch of images using mega detector, cropping the animal and saving it to the respective output directory """
    execution_started = datetime.now()
    print(f'[4.1]: Loading megadetector model for analysis.. ')
    try:
        model_checkpoint = torch.load(model_path, weights_only=False)
        model = model_checkpoint['model'].float().fuse().eval()
    except OSError as oer:
        print(oer)
    model.to(device_id)
    for item in tqdm(img_path_lists):
        # print(f'[4.2]: Pre-processing the images to tensors.. ')
        img_data = vis_utils.load_image(item) # Load the image file
        img_array = np.asarray(img_data) # Convert the image data into a numpy array
        img_resize = letterbox(img_array, new_shape=IMAGE_SIZE, stride=STRIDE, auto=True)[0] # Resize the image shape as per the megadetector model
        img_transpose = img_resize.transpose((2, 0, 1))  # HWC to CHW; PIL Image is RGB already
        img = np.ascontiguousarray(img_transpose)
        img = torch.from_numpy(img)
        img = img.to(device_id)
        img = img.float()
        img /= 255
        if len(img.shape) == 3:  # always true for now, TODO add inference using larger batch size
            img = torch.unsqueeze(img, 0)
        # print(f'[4.3]: Processing the tensors to detect objects using megadetector.. ')
        tot_detections = []
        labels = []
        bbox = []
        scores = []
        max_conf = 0.0
        final_output = {'input_file' : f'{item}'}
        # print(f'{final_output} processing on the {device_id}')
        with torch.no_grad():
            result = model(img)[0]
        result_nms = non_max_suppression(prediction=result, conf_thres=DETECTION_THRESHOLD) # applying non-maximum supression logic to the results
        normalization = torch.tensor(img_array.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for detection in result_nms:
            if len(detection):
                detection[:, :4] = scale_coords(img.shape[2:], detection[:, :4], img_array.shape).round()
                for *xyxy, conf, cls in reversed(detection):
                    # normalized center-x, center-y, width and height
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / normalization).view(-1).tolist()
                    api_box = ct_utils.convert_yolo_to_xywh(xywh)
                    conf = ct_utils.truncate_float(conf.tolist(), precision=3)
                    # MegaDetector output format's categories start at 1, but this model's start at 0
                    cls = int(cls.tolist())
                    labels.append(DEFAULT_DETECTOR_LABEL_MAP[cls])
                    scores.append(conf)
                    bbox.append(ct_utils.truncate_float_array(api_box, precision=4))
                    tot_detections.append({ 'category': DEFAULT_DETECTOR_LABEL_MAP[cls], 'conf': conf, 'bbox': ct_utils.truncate_float_array(api_box, precision=4)})
                    max_conf = max(max_conf, conf)
        final_output['max_detection_conf'] = max_conf
        final_output['detections'] = tot_detections
        if final_output:
            item_index = img_path_lists.index(item)
            category = spec_labels_list[item_index]
            cat_value = categories_set.index(category)
            category_output_dir_path = os.path.join(output_dir_path, category)
            # if detections have multiple results then, the highest confidence score object details will be selected
            # print(final_output)
            if final_output['detections']:
                for detection_values in final_output['detections']:
                    if detection_values['conf'] == final_output['max_detection_conf']:
                        bbox_values = detection_values['bbox']
                        # print(f'{bbox_values} be used to generate the txt file.. ')
                        # output = f'{cat_value} {" ".join(map(str,bbox_values))}' # the original bbox co-ordinates need to be modified for the YOLO label file
                        x1 = float(bbox_values[0]) + float(bbox_values[2])/2
                        y1 = float(bbox_values[1]) + float(bbox_values[3])/2
                        x2 = bbox_values[2]
                        y2 = bbox_values[3]
                        output = f'{cat_value} {x1} {y1} {x2} {y2}'
                        # print(output)
                        create_label_file(category_output_dir_path, item, output, category)
            else:
                print(f'No animal detected in the image: {item}')
                try:
                    create_directory(tbd_dir_path)
                    tbd_category_dir_path = os.path.join(tbd_dir_path, category)
                    create_directory(tbd_category_dir_path)
                    des_path = os.path.join(tbd_category_dir_path, os.path.basename(item))
                    try:
                        shutil.copy(item, des_path)
                    except shutil.Error as ser:
                        print(ser)
                except OSError as oer:
                    raise oer
    execution_completed = datetime.now() - execution_started
    torch.cuda.empty_cache()
    print(f'Time taken by MegaDetector to detect objects in the {len(img_path_lists)} images - {humanfriendly.format_timespan(execution_completed)}')




if __name__ == "__main__":
    start_time = datetime.now()
    print(f'[1]: Loading {input_dir_path} ... ')
    if os.path.isdir(input_dir_path) and os.listdir(input_dir_path):
        img_path_lists, spec_labels_list, total_files_count = load_dir_list(input_dir_path)
        if img_path_lists and spec_labels_list:
            print(f'[2]: Create the classes.txt file for the categories in the output folder.. ')
            categories_set = create_classes(spec_labels_list, output_dir_path)
            # verify the gpu availability in the system for mega detctor processing
            if torch.cuda.is_available():
                #num_gpus = 1 # hard-coded value used during the development of GPU and jobs allocation logic
                num_gpus = torch.cuda.device_count() # number of GPUs available in the system
                if num_gpus > 1:
                    print(f'[INFO]: PyTorch - CUDA has located {num_gpus} GPUs for processing')
                    print('[2]: Creating job batches for multi-processing with the multiple GPUs... ')
                    job_list, procs_list = create_job_batches(img_path_lists, num_gpus)
                    device_list = []
                    for proc in procs_list:
                        device_list.append(f'cuda:{int(proc)}')
                    # print(job_list)
                    # print(device_list)
                    print('[3]: Generate dataset for the batch of images using Mega Detector.. ')
                    job_counter = 0
                    first_job_counter = 0
                    while job_counter < len(job_list):
                        second_job_counter = first_job_counter + 1
                        if first_job_counter == 0:
                            print(first_job_counter, second_job_counter)
                            p1_head, p1_end = job_list[first_job_counter].split(':')
                            p2_head, p2_end = job_list[second_job_counter].split(':')
                            if not p1_end:
                                p1_head = int(p1_head)
                                p1 = mp.Process(target=generate_dataset, args=(img_path_lists[p1_head:], spec_labels_list[p1_head:], device_list[first_job_counter], output_dir_path, categories_set, ))
                            else:
                                p1_head = int(p1_head)
                                p1_end = int(p1_end)
                                p1 = mp.Process(target=generate_dataset, args=(img_path_lists[p1_head:p1_end], spec_labels_list[p1_head:p1_end], device_list[first_job_counter], output_dir_path, categories_set,))
                            if not p2_end:
                                p2_head = int(p2_head)
                                p2 = mp.Process(target=generate_dataset, args=(img_path_lists[p2_head:], spec_labels_list[p2_head:], device_list[second_job_counter], output_dir_path, categories_set,)) 
                            else:
                                p2_head = int(p2_head)
                                p2_end = int(p2_end)
                                p2 = mp.Process(target=generate_dataset, args=(img_path_lists[p2_head:p2_end], spec_labels_list[p2_head:p2_end], device_list[second_job_counter], output_dir_path, categories_set,))
                            p1.start()
                            p2.start()
                            p1.join()
                            p2.join()
                        else:
                            print(first_job_counter, second_job_counter)
                            if first_job_counter >= len(job_list):
                                break
                            elif second_job_counter >= len(job_list):
                                print(job_list[first_job_counter])
                                p1_head, p1_end = job_list[first_job_counter].split(':')
                                if not p1_end:
                                    p1_head = int(p1_head)
                                    p1 = mp.Process(target=generate_dataset, args=(img_path_lists[p1_head:], spec_labels_list[p1_head:], device_list[first_job_counter], output_dir_path, categories_set,))
                                else:
                                    p1_head = int(p1_head)
                                    p1_end = int(p1_end)
                                    p1 = mp.Process(target=generate_dataset, args=(img_path_lists[p1_head:p1_end], spec_labels_list[p1_head:p1_end], device_list[first_job_counter], output_dir_path, categories_set,))
                                p1.start()
                                p1.join()
                            else:
                                print(job_list[first_job_counter], job_list[second_job_counter])
                                p1_head, p1_end = job_list[first_job_counter].split(':')
                                p2_head, p2_end = job_list[second_job_counter].split(':')
                                if not p1_end:
                                    p1_head = int(p1_head)
                                    p1 = mp.Process(target=generate_dataset, args=(img_path_lists[p1_head:], spec_labels_list[p1_head:], device_list[first_job_counter], output_dir_path, categories_set,))
                                else:
                                    p1_head = int(p1_head)
                                    p1_end = int(p1_end)
                                    p1 = mp.Process(target=generate_dataset, args=(img_path_lists[p1_head:p1_end], spec_labels_list[p1_head:p1_end], device_list[first_job_counter], output_dir_path, categories_set,))
                                if not p2_end:
                                    p2_head = int(p2_head)
                                    p2 = mp.Process(target=generate_dataset, args=(img_path_lists[p2_head:], spec_labels_list[p2_head:], device_list[second_job_counter], output_dir_path, categories_set,)) 
                                else:
                                    p2_head = int(p2_head)
                                    p2_end = int(p2_end)
                                    p2 = mp.Process(target=generate_dataset, args=(img_path_lists[p2_head:p2_end], spec_labels_list[p2_head:p2_end], device_list[second_job_counter], output_dir_path, categories_set,))
                                p1.start()
                                p2.start()
                                p1.join()
                                p2.join()
                        first_job_counter = second_job_counter + 1
                        job_counter = job_counter + 1
                else:
                    print('[INFO]: PyTorch - CUDA has located 1 GPU for processing')
                    print('[3]: Creating job batches to process the images with the single GPU... ')
                    device_id = 'cuda:0'
                    job_list, procs_list = create_job_batches(img_path_lists, num_gpus)
                    print('[4]: Generate dataset for the batch of images using Mega Detector.. ')
                    job_counter = 0
                    while job_counter < len(job_list):
                        p_head, p_tail = job_list[job_counter].split(':')
                        if not p_tail:
                            p_head = int(p_head)
                            generate_dataset(img_path_lists[p_head:], spec_labels_list[p_head:], device_id, output_dir_path, categories_set)
                        else:
                            p_head = int(p_head) 
                            p_tail = int(p_tail)
                            generate_dataset(img_path_lists[p_head:p_tail], spec_labels_list[p_head:p_tail], device_id, output_dir_path, categories_set)
                        job_counter = job_counter + 1
            else:
                print('[INFO]: PyTorch - CUDA cannot find any GPU for processing')
                print('[2]: Creating job batches to process the images with the CPU... ')
                device_id = ['cpu:0']
                job_list, procs_list = create_job_batches(img_path_lists, 0)
        else:
            print(f'[Error]: Image paths and labels loaded from the {input_dir_path} are empty')
        end_time = datetime.now() - start_time
        print(f'Time taken to generate the object detection dataset for {total_files_count} images - {humanfriendly.format_timespan(end_time)}')
    else:
        print(f'[Error]: Cannot load the {input_dir_path} directory!!')