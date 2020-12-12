import os
import json

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

from utils.image import pad_img

class VinoModel:
    def __init__(self, config, num_proc):
        self.config = config
        self.ie = IECore()
        self.ie.set_config({'CPU_THREADS_NUM': str(num_proc), 'CPU_BIND_THREAD': 'NO'}, 'CPU')
        self.model_xml = config['model_xml']
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        self.net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)
        self.net.reshape({'image': (1, 3, config['shape'][1], config['shape'][0])})
        self.net.batch_size = 1
        self.input_blob = next(iter(self.net.inputs))
        self.exec_net = self.ie.load_network(network=self.net, num_requests=1, device_name='CPU')
        self.batch = np.zeros([1, 3, self.config['shape'][1], self.config['shape'][0]], dtype=np.float32)

    def start_async_infer(self, batch, request_id):
        """Starts async inference with certain request_id"""
        self.exec_net.start_async(request_id=request_id, inputs={self.input_blob: batch})

    def get_outputs_wait(self, request_id):
        """Waits for async inference to be over and returns result"""
        if self.exec_net.requests[request_id].wait(-1) == 0:
            return self.exec_net.requests[request_id].outputs
        else:
            return 0

    def preprocess(self, image):
        to_shape = tuple(self.config['shape'])
        if self.config['pad_image']:
            image = pad_img(image, to_shape)
        else:
            image = cv2.resize(image, to_shape)
        self.batch[0, :, :, :] = image.transpose(2, 0, 1)    

    def predict(self, image):
        self.preprocess(image)
        self.start_async_infer(self.batch[0, :, :, :], request_id=0)
        results = self.get_outputs_wait(request_id=0)
        out_name = self.config.get('out_name')
        return results[out_name] if out_name else results


def load_model_config(path):
    with open(path) as in_file:
        config = json.load(in_file)
        config['model_xml'] = os.path.join(
            os.path.split(path)[0],
            config['model_xml']
        )
    return config