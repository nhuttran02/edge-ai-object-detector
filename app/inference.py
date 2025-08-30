import onnxruntime as ort
import numpy as np
import cv2

class ONNXDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img_resized = cv2.resize(img, (640, 640))
        img = img_resized.transpose(2,0,1).astype(np.float32)/255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, img):
        input_tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs