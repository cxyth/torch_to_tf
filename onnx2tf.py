# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import onnx
from tensorflow import keras
import onnx2keras
from onnx2keras import onnx_to_keras
from onnx_tf.backend import prepare


'''
测试环境：
tensorflow-gpu      2.5.0
onnx                1.10.1
onnx2keras          0.0.24
'''

def onnx_to_h5(onnx_path, h5_path):
    # Load the ONNX file
    onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(model)
    print(onnx_model.graph.output)
    k_model = onnx_to_keras(onnx_model, ['input'])
    keras.models.save_model(k_model, h5_path, overwrite=True, include_optimizer=True)


def onnx_to_pb(onnx_path, pb_path):
    # Load the ONNX file
    onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(model)
    print(onnx_model.graph.output)
    k_model = onnx_to_keras(onnx_model, ['input'])
    keras.models.save_model(k_model, pb_path, save_format="tf")


def onnx_to_pb2(onnx_path, pb_path):
    # Load the ONNX file
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(model.graph.output)
    # Import the ONNX model to Tensorflow
    tf_rep = prepare(model, device='CUDA', strict=True, logging_level='INFO')
    print('inputs:', tf_rep.inputs)
    print('outputs:', tf_rep.outputs)
    print('tensor_dict:')
    print(tf_rep.onnx_op_list)
    tf_rep.export_graph(pb_path)  # export the model

    # print('run:')
    # raw_img, input = load_img('img0.8m/2.png')
    # output = tf_rep.run(input)[0]
    # print("output:", output.shape)
    # pred = np.argmax(output, axis=1).reshape((256, 256))
    # draw(raw_img, pred, '2_pb.png')



if __name__=="__main__":
    onnx_path = 'model.onnx'
    h5_path = 'model.h5'
    pb_path = 'tf_model'
    onnx_to_h5(onnx_path, h5_path)
    onnx_to_pb(onnx_path, pb_path)
    # onnx_to_pb2(onnx_path, 'tf_model2')
