import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face

from flask import Flask
from flask import request
import json
from flask import make_response
from flask import render_template
import flask

import base64
import numpy as np

def base64_to_image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.fromstring(img_data,np.uint8)
    img =cv2.imdecode(img_array,cv2.COLOR_RGB2BGR)
    return img

def image_to_base64(image_np):
    image = cv2.imencode('.jpg',image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code


app = Flask(__name__)

@app.route('/predict',methods=["POST","GET"])
def index():
    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                        r_model_path="./original_model/rnet_epoch.pt",
                                        o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)


    #get_json=flask.request.get_json(force=True)

    print(request.data)
    base_data = request.json['image']

    img = base64_to_image(base_data)
    bboxs, landmarks = mtcnn_detector.detect_face(img)

    #初始化一下json
    res = {}
    faces = {}
    if bboxs.shape[0] < 1:
        res["success"] = False
        res["faces_detected"] = faces
        return flask.jsonify(res)
    else:
        res["success"] = True

    #这里开始处理一幅图中有多个人脸的情况
    for i in range(bboxs.shape[0]):
        x1=int(bboxs[i][0])
        x2=int(bboxs[i][2])
        y1=int(bboxs[i][1])
        y2=int(bboxs[i][3])
        face = img[y1:y2, x1:x2]

        face_name = "face_"+str(i)
        return_base64 = image_to_base64(face)
        faces[face_name]=return_base64

    res["faces_detected"]=faces
    return flask.jsonify(res)

if __name__ == '__main__':
    app.run("0.0.0.0",5002,debug=True)


# if __name__ == '__main__':
#
#     pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
#     mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)
#
#     img = cv2.imread("./6.jpeg")
#     img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #b, g, r = cv2.split(img)
#     #img2 = cv2.merge([r, g, b])
#
#     bboxs, landmarks = mtcnn_detector.detect_face(img)
#     # print box_align
#     # print(bboxs)
#     # x1=int(bboxs[0][0])
#     # x2=int(bboxs[0][2])
#     # y1=int(bboxs[0][1])
#     # y2=int(bboxs[0][3])
#     #
#     # face = img[y1:y2,x1:x2]
#     # cv2.imwrite('face2.jpg',face)
#
#     save_name = 'r_444.jpg'
#     vis_face(img_bg,bboxs,landmarks, save_name)
