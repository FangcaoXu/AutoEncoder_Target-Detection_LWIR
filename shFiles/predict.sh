#!/usr/bin/env bash

# ${variable_name}/$variable_name to perform the variable expansion
# Radiance_100_10_0_total.csv
# "#" operates on the left side of a parameter, and "%" operates on the right
# * is the part to be removed
#filename=Radiance_100_10_0_total.csv
#${filename#*_}  # 100_10_0_total.csv
#${filename##*_} # total.csv
#${filename%_*}  # Radiance_100_10_0
#${filename##*[![:digit:]]}


for file in /home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/MidLatitude/TotalRadianceCSV/*
do
  filename=$(basename "$file")  # no space before and after =, space is used as command separator
  day="${filename#*_}"     # 100_10_0_total.csv
  day="${day%%_*}"         # 100, extract days from the filename
  reflec="${filename%_*}"  # Radiance_100_10_0
  reflec="${reflec##*_}"   # 0, extract reflectivity

  # extract the filename before _component.csv
  outputfilename_trans="${filename%_*}_trans.csv"
  outputfilename_grnd="${filename%_*}_grnd.csv"
  outputfilename_down="${filename%_*}_down.csv"
  outputfilename_up1="${filename%_*}_up1.csv"
  outputfilename_up2="${filename%_*}_up2.csv"
  outputfilename_surf="${filename%_*}_surf.csv"

  # assign the output file path for each component
  outfilepath_trans="/home/graduate/fbx5002/disk10TB/DARPA/predictedCSV/MidLatitude/TransmissionCSV/$outputfilename_trans"
  outfilepath_grnd="/home/graduate/fbx5002/disk10TB/DARPA/predictedCSV/MidLatitude/GroundReflectedCSV/$outputfilename_grnd"
  outfilepath_down="/home/graduate/fbx5002/disk10TB/DARPA/predictedCSV/MidLatitude/DownwellingCSV/$outputfilename_down"
  outfilepath_up1="/home/graduate/fbx5002/disk10TB/DARPA/predictedCSV/MidLatitude/PathThermalEmissionCSV/$outputfilename_up1"
  outfilepath_up2="/home/graduate/fbx5002/disk10TB/DARPA/predictedCSV/MidLatitude/PathThermalScatteringCSV/$outputfilename_up2"
  outfilepath_surf="/home/graduate/fbx5002/disk10TB/DARPA/predictedCSV/MidLatitude/SurfaceEmissionCSV/$outputfilename_surf"

  # separate one line command to several lines no white space after \
  #-eq: equal to
  #-ne: not equal to
  #-lt: less than
  #-le: less than or equal to
  #-gt: greater than
  #-ge: greater than or equal to
  if [ "$day" -ge 79 -a "$day" -lt 266 ] # space after [ and before ] in if statement
  then modelpath="/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsSpringSummer"
  else modelpath="/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsAutumnWinter"
  fi
  #predict the transmission
  echo $outfilepath_trans
  pipenv run python predict.py \
  --input "$file" \
  --model "$modelpath/l-mse_batch128_filters64_vector16_intotal_outtrans/ae_latest.tar" \
  --output $outfilepath_trans
  #predict the ground reflected component
  echo $outfilepath_grnd
  pipenv run python predict.py \
  --input "$file" \
  --model "$modelpath/l-mse_batch128_filters64_vector16_intotal_outgrnd/ae_latest.tar" \
  --output $outfilepath_grnd
  # when the reflectance is not 0, predict the downwelling component
  if [ "$reflec" -ne 0 ]
  then
  echo $outfilepath_down
  pipenv run python predict.py \
  --input "$file" \
  --model "$modelpath/l-mse_batch128_filters64_vector16_intotal_no0reflect_outdown/ae_latest.tar" \
  --output $outfilepath_down
  fi
  # predict the path thermal emission component
  echo $outfilepath_up1
  pipenv run python predict.py \
  --input "$file" \
  --model "$modelpath/l-mse_batch128_filters64_vector16_intotal_outup1_emission/ae_latest.tar" \
  --output $outfilepath_up1
  # predict the path thermal scattering component
  echo $outfilepath_up2
  pipenv run python predict.py \
  --input "$file" \
  --model  "$modelpath/l-mse_batch128_filters64_vector16_intotal_outup2_scatter/ae_latest.tar" \
  --output $outfilepath_up2
  # predict the surface emission component
  echo $outfilepath_surf
  pipenv run python predict.py \
  --input "$file" \
  --model "$modelpath/l-mse_batch128_filters64_vector16_intotal_outsurf/ae_latest.tar" \
  --output $outfilepath_surf
done


#pipenv run python predict.py \
#  --input "/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/MidLatitude/TotalRadianceCSV/Radiance_107_14_10_total.csv" \
#  --model "/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/trainModel_July30th/trainModelsAutumnWinter/l-mse_batch128_filters64_vector16_intotal_outtrans/ae_latest.tar" \
#  --output "/home/graduate/fbx5002/disk10TB/DARPA/predictedCSV/MidLatitude/Radiance_107_14_10_trans_autumnwintermodel.csv"
#

## run predict.py from pycharm terminal using within the virtual enviroment
#PIPENV_VERBOSITY=-1
#pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage1_7.5_12um/DifferentMaterials/Acetone/Radiance_107_14_total.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsSpringSummer/l-mse_batch128_filters64_vector16_intotal_outgrnd/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/DifferentMaterials/Acetone/predictedCSV/Radiance_107_14_grnd.csv'
#pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage1_7.5_12um/DifferentMaterials/Acetone/Radiance_107_14_total.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsSpringSummer/l-mse_batch128_filters64_vector16_intotal_outup1_emission/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/DifferentMaterials/Acetone/predictedCSV/Radiance_107_14_up1.csv'
#pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage1_7.5_12um/DifferentMaterials/Acetone/Radiance_107_14_total.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsSpringSummer/l-mse_batch128_filters64_vector16_intotal_outup2_scatter/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/DifferentMaterials/Acetone/predictedCSV/Radiance_107_14_up2.csv'
#pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage1_7.5_12um/DifferentMaterials/Acetone/Radiance_107_14_total.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsSpringSummer/l-mse_batch128_filters64_vector16_intotal_outsurf/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/DifferentMaterials/Acetone/predictedCSV/Radiance_107_14_surf.csv'
#pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage1_7.5_12um/DifferentMaterials/Acetone/Radiance_107_14_total.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsSpringSummer/l-mse_batch128_filters64_vector16_intotal_no0reflect_outdown/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/DifferentMaterials/Acetone/predictedCSV/Radiance_107_14_down.csv'
#pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage1_7.5_12um/DifferentMaterials/Acetone/Radiance_107_14_total.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsSpringSummer/l-mse_batch128_filters64_vector16_intotal_outtrans/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/DifferentMaterials/Acetone/predictedCSV/Radiance_107_14_trans.csv'
#
#
#pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/Stage1_7.5_12um/DifferentTemp/originalCSV/360K/Radiance_107_14_total.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsSpringSummer/l-mse_batch128_filters64_vector16_intotal_outgrnd/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/DifferentTemp/predictedCSV/360K/Radiance_107_14_grnd.csv'
#pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/Stage1_7.5_12um/DifferentTemp/originalCSV/360K/Radiance_107_14_total.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsSpringSummer/l-mse_batch128_filters64_vector16_intotal_outup1_emission/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/DifferentTemp/predictedCSV/360K/Radiance_107_14_up1.csv'
#pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/Stage1_7.5_12um/DifferentTemp/originalCSV/360K/Radiance_107_14_total.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsSpringSummer/l-mse_batch128_filters64_vector16_intotal_outup2_scatter/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/DifferentTemp/predictedCSV/360K/Radiance_107_14_up2.csv'
#pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/Stage1_7.5_12um/DifferentTemp/originalCSV/360K/Radiance_107_14_total.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsSpringSummer/l-mse_batch128_filters64_vector16_intotal_outsurf/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/DifferentTemp/predictedCSV/360K/Radiance_107_14_surf.csv'
#pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/Stage1_7.5_12um/DifferentTemp/originalCSV/360K/Radiance_107_14_total.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage1_7.5_12um/trainModel_July30th/trainModelsSpringSummer/l-mse_batch128_filters64_vector16_intotal_no0reflect_outdown/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/DifferentTemp/predictedCSV/360K/Radiance_107_14_down.csv'


# remember to set header=1 when the parse_csv function is called in predict.py
pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage2_BHData/BHExtracted/1_BHnetwork/BHExtracted.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage2_BHData/trainModel/l-mse_batch128_filters64_vector16_intotal_outtrans/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage2_BHData/BHExtracted/1_BHnetwork/Radiance_trans.csv'
pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage2_BHData/BHExtracted/1_BHnetwork/BHExtracted.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage2_BHData/trainModel/l-mse_batch128_filters64_vector16_intotal_outdown/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage2_BHData/BHExtracted/1_BHnetwork/Radiance_down.csv'
pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage2_BHData/BHExtracted/1_BHnetwork/BHExtracted.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage2_BHData/trainModel/l-mse_batch128_filters64_vector16_intotal_outup1_emission/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage2_BHData/BHExtracted/1_BHnetwork/Radiance_up1.csv'
pipenv run python predict.py --input '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage2_BHData/BHExtracted/1_BHnetwork/BHExtracted.csv' --model '/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage2_BHData/trainModel/l-mse_batch128_filters64_vector16_intotal_outup2_scatter/ae_latest.tar' --output '/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/Stage2_BHData/BHExtracted/1_BHnetwork/Radiance_up2.csv'


