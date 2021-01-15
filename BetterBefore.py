import IA_comptacloud as IACC
import sys

datasetPath = '.dataset.csv'

conv1_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 32
conv1_kernelsize = int(sys.argv[2]) if len(sys.argv) > 2 else 3
conv1_activation = str(sys.argv[3]) if len(sys.argv) > 3 else 'relu'
conv2_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 64
conv2_kernelsize = int(sys.argv[5]) if len(sys.argv) > 5 else 3
conv2_activation = str(sys.argv[6]) if len(sys.argv) > 6 else 'relu'
dropout1 = float(sys.argv[7]) if len(sys.argv) > 7 else 0.25
dense_units = int(sys.argv[8]) if len(sys.argv) > 8 else 128
dense_activation = str(sys.argv[9]) if len(sys.argv) > 9 else 'relu'
dropout2 = float(sys.argv[10]) if len(sys.argv) > 10 else 0.5

score = IACC.train_model(datasetPath,'BBModel', conv1_layers, conv1_kernelsize, conv1_activation, conv2_layers, conv2_kernelsize, conv2_activation, dropout1, dense_units, dense_activation, dropout2)