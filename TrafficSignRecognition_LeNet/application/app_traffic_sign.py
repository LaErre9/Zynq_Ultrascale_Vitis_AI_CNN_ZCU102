from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse

from resource import getrusage, RUSAGE_SELF


_divider = '-------------------------------'


def preprocess_fn(image_path, fix_scale):
    '''
    Image pre-processing.
    Opens image as grayscale, adds channel dimension, normalizes to range 0:1
    and then scales by input quantization scaling factor
    input arg: path of image file
    return: numpy array
    '''
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    image = image.reshape(32,32,3)
    image = image * (1/255.0) * fix_scale
    image = image.astype(np.int8)
    return image


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]



def runDPU(id,start,dpu,img):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    # we can avoid output scaling if use argmax instead of softmax
    #output_fixpos = outputTensors[0].get_attr("fix_point")
    #output_scale = 1 / (2**output_fixpos)

    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    ids=[]
    ids_max = 10
    outputData = []
    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])
        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData[len(ids)])
        ids.append((job_id,runSize,start+count))
        count = count + runSize 
        if count<n_of_images:
            if len(ids) < ids_max-1:
                continue
        for index in range(len(ids)):
            dpu.wait(ids[index][0])
            write_index = ids[index][2]
            '''store output vectors '''
            for j in range(ids[index][1]):
                # we can avoid output scaling if use argmax instead of softmax
                # out_q[write_index] = np.argmax(outputData[0][j] * output_scale)
                out_q[write_index] = np.argmax(outputData[index][0][j])
                write_index += 1
        ids=[]



def app(image_dir,threads,model,eval):

    listimage=os.listdir(image_dir)
    runTotal = len(listimage)

    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos

    ''' preprocess images '''
    print (_divider)
    print('Pre-processing',runTotal,'images...')
    img = []
    for i in range(runTotal):
        path = os.path.join(image_dir,listimage[i])
        img.append(preprocess_fn(path, input_scale))

    '''run threads '''
    print (_divider)
    print('Starting',threads,'threads...')
    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print (_divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))


    ''' post-processing '''
    classes = ['0','1','2','3','4','5','6','7','8','9',
             '10','11','12','13','14','15','16','17','18','19',
             '20','21','22','23','24','25','26','27','28','29',
             '30','31','32','33','34','35','36','37','38','39',
             '40','41','42']
    correct = 0
    wrong = 0
    for i in range(len(out_q)):
        prediction = classes[out_q[i]]
        ground_truth, _ = listimage[i].split('_',1)
        if (ground_truth==prediction):
            correct += 1
        else:
            wrong += 1
    accuracy = correct/len(out_q)
    print('Correct:%d, Wrong:%d, Accuracy:%.4f' %(correct,wrong,accuracy))
    print (_divider)

    if (eval == 'prediction'):
        class_names = ["Speed limit (20km/h)", "Speed limit (30km/h)",
                       "Speed limit (50km/h)", "Speed limit (60km/h)",
                       "Speed limit (70km/h)", "Speed limit (80km/h)",
                       "End of speed limit (80km/h)", "Speed limit (100km/h)",
                       "Speed limit (120km/h)", "No passing",
                       "No passing for vehicles over 3.5 metric tons",
                       "Right-of-way at the next intersection",
                       "Priority road", "Yield", "Stop", "No vehicles",
                       "Vehicles over 3.5 metric tons prohibited",
                       "No entry", "General caution", "Dangerous curve to the left",
                       "Dangerous curve to the right", "Double curve",
                       "Bumpy road", "Slippery road", "Road narrows on the right",
                       "Road work", "Traffic signals", "Pedestrians",
                       "Children crossing", "Bicycles crossing",
                       "Beware of ice/snow", "Wild animals crossing",
                       "End of all speed and passing limits",
                       "Turn right ahead", "Turn left ahead",
                       "Ahead only", "Go straight or right",
                       "Go straight or left", "Keep right", "Keep left",
                       "Roundabout mandatory", "End of no passing",
                       "End of no passing by vehicles over 3.5 metric tons"]
        for i in range(len(out_q)):
            sign = class_names[out_q[i]]
            print('Traffic Sign Recognized: %s' %(sign))

    memory_usage_kb = getrusage(RUSAGE_SELF).ru_maxrss / 1024
    print("Memory:  %.6f kb" %(memory_usage_kb))
    return



def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='images', help='Path to folder of images. Default is images')  
  ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='CNN_zcu102.xmodel', help='Path of xmodel. Default is CNN_zcu102.xmodel')
  ap.add_argument('-e', '--eval',      type=str, default='evaluation', help='Mode of execution. Default is evaluation')
  args = ap.parse_args()  
  
  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)
  print (' --eval      : ', args.eval)

  app(args.image_dir,args.threads,args.model,args.eval)

if __name__ == '__main__':
  main()
