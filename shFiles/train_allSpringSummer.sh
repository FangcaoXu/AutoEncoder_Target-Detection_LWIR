#$test.sh a b c d

# when you run this bash commands, the first input argument "a" you put in the commandline is the datafile
# here the datafile is the data1_256x8.npz
#!/usr/bin/env bash
DATAFILE=$1
WORKERS=48
EPOCHS=10000

time pipenv run python /amethyst/s0/fbx5002/PythonWorkingDir/machineLearning/train.py \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --data "$DATAFILE" \
    --input total \
    --label trans > total_trans.log \
    --datalength 7854

time pipenv run python /amethyst/s0/fbx5002/PythonWorkingDir/machineLearning/train.py \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --data "$DATAFILE" \
    --input total \
    --label grnd > total_grnd.log \
    --datalength 7854

time pipenv run python /amethyst/s0/fbx5002/PythonWorkingDir/machineLearning/train.py \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --data "$DATAFILE" \
    --input total_no0reflect \
    --label down > total_down.log \
    --datalength 6732

time pipenv run python /amethyst/s0/fbx5002/PythonWorkingDir/machineLearning/train.py \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --data "$DATAFILE" \
    --input total \
    --label surf > total_surf.log \
    --datalength 7854

time pipenv run python /amethyst/s0/fbx5002/PythonWorkingDir/machineLearning/train.py \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --data "$DATAFILE" \
    --input total \
    --label up1_emission > total_up1_emission.log \
    --datalength 7854

time pipenv run python /amethyst/s0/fbx5002/PythonWorkingDir/machineLearning/train.py \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --data "$DATAFILE" \
    --input total \
    --label up2_scatter > total_up2_scatter.log \
    --datalength 7854


time pipenv run python /amethyst/s0/fbx5002/PythonWorkingDir/machineLearning/training.py \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --data "$DATAFILE" \
    --input total \
    --label total > total_total.log \
    --datalength 7854