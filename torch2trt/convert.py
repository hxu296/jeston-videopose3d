import sys

# a hacky way to import the torch2trt package
dist_packages = '/usr/lib/python3.6/dist-packages/'
torch2trt_egg = '/usr/lib/python3.6/dist-packages/torch2trt-0.3.0-py3.6.egg/'
if dist_packages not in sys.path:
    sys.path.append(dist_packages)
if torch2trt_egg not in sys.path:
    sys.path.append(torch2trt_egg)
    
import torch
from torch2trt import torch2trt
import importlib.util
import os.path as osp
from config import parse_args


def import_class_from_file(file_path, class_name):
  """
  return a class pointing to class_name from file_path
  """
  spec = importlib.util.spec_from_file_location(osp.basename(file_path)[:-3], file_path)
  foo = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(foo)
  return getattr(foo, class_name)

def convert_model(Model, ckpt_path, input_shape, model_args): 
  """
  convert and save model to cwd
  """
  print('Initializing model using args:')
  print(model_args)
  model = Model(**model_args)
  print('Loading checkpoint...')
  model.load_state_dict(torch.load(ckpt_path)['model_pos'])
  model.eval()
  model = model.cuda()
  print('Converting model...')
  model_trt = torch2trt(model, [torch.ones(tuple(input_shape)).cuda()])
  print('Saving model...')
  ckpt_trt_path = 'trt_' + osp.basename(ckpt_path)
  torch.save(model_trt.state_dict(), ckpt_trt_path)
  print('Done! Converted checkpoint saved in:', ckpt_trt_path)
  

if __name__ == '__main__':
  args = parse_args()
  Model = import_class_from_file(args.common.model_path, args.common.model_name)
  convert_model(Model, args.common.ckpt_path, args.common.input_shape, dict(args.model))


