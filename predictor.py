import cv2
from string import ascii_letters, digits

from nopaddle.tools.infer.utility import parse_args
from nopaddle.predict_rec import TextRecognizer


class NoPaddleOnnxPredictor:
    def __init__(self, onnx, dict_, use_gpu=True):
        self.onnx = None
        self.dict = ascii_letters + digits

        args = parse_args()
        args.use_onnx = True
        args.use_gpu = use_gpu
        args.rec_model_dir = onnx
        args.rec_char_dict_path = dict_

        self.predictor = TextRecognizer(args)

    def __call__(self, image_path):
        image = cv2.imread(image_path)
        pred = self.predictor([image])[0][0][0]
        return pred



if __name__ == '__main__':
    ONNX_MODEL_PATH = './data/model.onnx'
    DICT_PATH = './data/dict.txt'
    IMAGE_PATH = './data/test.png'

    predictor = NoPaddleOnnxPredictor(ONNX_MODEL_PATH, DICT_PATH)
    res = predictor(IMAGE_PATH)
    print(res)


