#!/usr/bin/env bash
echo $PATH
DATAFILE=$1   # $1: first commandline argument
WORKERS=48
EPOCHS=10000

# 15530： 365*6*7  # with 0 reflectivity
# 13140： 365*6*6
# 7854 : 187*6*7  # spring-summer days
# 7476 : 178*6*7
# 6732： 187*6*6  # spring-summer days
# 6408： 178*6*6

# use > (overwrite) or >> (append) the cmd prompt output (e.g., print()) to the file
time pipenv run python /amethyst/s0/fbx5002/PythonWorkingDir/DeepLearning/train.py \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --data "$DATAFILE" \
    --input total \
    --label trans > total_trans.log \
    --datalength 13140 \
    --matrixsize 256 7 \
    --out_dir /home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage2_BHData/trainModel

time pipenv run python /amethyst/s0/fbx5002/PythonWorkingDir/DeepLearning/train.py \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --data "$DATAFILE" \
    --input total \
    --label down > total_down.log \
    --datalength 13140 \
    --matrixsize 256 7 \
    --out_dir /home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage2_BHData/trainModel

time pipenv run python /amethyst/s0/fbx5002/PythonWorkingDir/DeepLearning/train.py \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --data "$DATAFILE" \
    --input total \
    --label up1_emission > total_up1.log \
    --datalength 13140 \
    --matrixsize 256 7 \
    --out_dir /home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage2_BHData/trainModel

time pipenv run python /amethyst/s0/fbx5002/PythonWorkingDir/DeepLearning/train.py \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --data "$DATAFILE" \
    --input total \
    --label up2_scatter > total_up2.log \
    --datalength 13140 \
    --matrixsize 256 7 \
    --out_dir /home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage2_BHData/trainModel