import sys
import torch
from torch2trt import torch2trt
import importlib.util
import os.path as osp
import argparse

# return a class point to class_name from file_path
def import_class_from_file(file_path, class_name):
  spec = importlib.util.spec_from_file_location(osp.basename(file_path)[:-3], file_path)
  foo = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(foo)
  return getattr(foo, class_name)

# convert and save model to cwd
def convert_model(Model, ckpt_path, input_shape): 
  model = Model()
  model.load_state_dict(torch.load(ckpt_path))
  model.eval().cuda()
  model_trt = torch2trt(model, [torch.ones(tuple(input_shape))])
  model_trt_path = 'trt' + osp.basename(file_path)
  torch.save(model_trt.state_dict(), model_trt_path)
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert PyTorch model to trt model.')
  parser.add_argument('-m', '--model_path', type=str, help='path to file that contains the model class') 
  parser.add_argument('-c', '--ckpt_path', type=str, help='path to the checkpoint file') 
  parser.add_argument('-n', '--model_name', type=str, help='model class name')
  parser.add_argument('-s', '--input_shape', type=str, help='model input shape')
  args = parser.parse_args()
  Model = import_class_from_file(args.model_path, args.model_name)
  convert_model(Model, args.ckpt_path, args.input_shape.split(','))


