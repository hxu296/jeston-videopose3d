# generate a sample (smaller) dataset by clipping the npy file to only 1 subject

import argparse


def gen_sample(path_3d, path_2d, max_subjects):
  # load and select sample data
  original_3d = np.load(path_3d, allow_pickle=True)['positions_3d'].item()
  subjects = original_3d.keys()
  target_subjects = subjects[:max_subjects]
  sample_3d = {key:original_3d[key] for key in target_subjects}
  original_2d = np.load(path_2d, allow_pickle=True)
  # TODO: dump sample data to npz file
  assert False, 'not implemented'
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate a sample dataset')   parer.add_argument('-3d', '--dataset_3d', type=str, help='path to the original 3d dataset')
  parser.add_argument('-2d', '--dataset_2d', type=str, help='path to the original 2d dataset')
  parser.add_argument('-s', '--subjects', default=1, type=int, help='number of subjects in the sample dataset')
  args = parser.parse_args()
  gen_sample(args.dataset_3d, args.dataset_2d, args.subjects)
