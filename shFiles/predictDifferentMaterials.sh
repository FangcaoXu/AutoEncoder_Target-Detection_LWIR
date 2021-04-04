#!/usr/bin/env bash
# ${variable_name}/$variable_name to perform the variable expansion
# Radiance_100_10_0_total.csv
# "#" operates on the left side of a parameter, and "%" operates on the right
# * is the part to be removed
#filename=Radiance_100_10_total.csv
#${filename#*_}  # 100_10_0_total.csv
#${filename##*_} # total.csv
#${filename%_*}  # Radiance_100_10_0
#${filename##*[![:digit:]]}

# `` : execute the command inside; for folder in `ls .`
inputfolder="/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/DifferentMaterials/exoscan/originalCSV"
outputfolder="/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/DifferentMaterials/exoscan/predictedCSV"
# /*/: only loop through subdirectories
for folder in $inputfolder/*/
do
  echo $folder
  foldername=$(basename $folder)
  # find the total radiance which is used as the input
  totalradiance=$(find $folder -type f -name "*total.csv")
  echo $totalradiance
  # extract the day from the input file basename
  filename=$(basename $totalradiance)  # no space before and after =, space is used as command separator
  day=${filename#*_}     # 100_10_0_total.csv
  day=${day%%_*}         # 100, extract days from the filename
  # extract the filename before _total.csv
  outputfilename_trans=${filename%_*}_trans.csv
  outputfilename_grnd=${filename%_*}_grnd.csv
  outputfilename_down=${filename%_*}_down.csv
  outputfilename_up1=${filename%_*}_up1.csv
  outputfilename_up2=${filename%_*}_up2.csv
  outputfilename_surf=${filename%_*}_surf.csv
  # assign the output file path for each component
  outfilepath_trans=$outputfolder/$foldername/$outputfilename_trans
  outfilepath_grnd=$outputfolder/$foldername/$outputfilename_grnd
  outfilepath_down=$outputfolder/$foldername/$outputfilename_down
  outfilepath_up1=$outputfolder/$foldername/$outputfilename_up1
  outfilepath_up2=$outputfolder/$foldername/$outputfilename_up2
  outfilepath_surf=$outputfolder/$foldername/$outputfilename_surf
  # separate one line command to several lines no white space after \
  # -eq: equal to
  # -ne: not equal to
  # -lt: less than
  # -le: less than or equal to
  # -gt: greater than
  # -ge: greater than or equal to
  if [ "$day" -ge 79 -a "$day" -lt 266 ] # space after [ and before ] in if statement
  then modelpath="/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/trainModel_July30th/trainModelsSpringSummer"
  else modelpath="/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/trainModel_July30th/trainModelsAutumnWinter"
  fi
  #predict the transmission
  echo $outfilepath_trans
  pipenv run python predict.py \
  --input $totalradiance \
  --model $modelpath/l-mse_batch128_filters64_vector16_intotal_outtrans/ae_latest.tar \
  --output $outfilepath_trans
  #predict the ground reflected component
  echo $outfilepath_grnd
  pipenv run python predict.py \
  --input $totalradiance \
  --model $modelpath/l-mse_batch128_filters64_vector16_intotal_outgrnd/ae_latest.tar\
  --output $outfilepath_grnd
  # predict the downwelling component
  echo $outfilepath_down
  pipenv run python predict.py \
  --input $totalradiance \
  --model $modelpath/l-mse_batch128_filters64_vector16_intotal_no0reflect_outdown/ae_latest.tar \
  --output $outfilepath_down
  # predict the path thermal emission component
  echo $outfilepath_up1
  pipenv run python predict.py \
  --input $totalradiance \
  --model $modelpath/l-mse_batch128_filters64_vector16_intotal_outup1_emission/ae_latest.tar \
  --output $outfilepath_up1
  # predict the path thermal scattering component
  echo $outfilepath_up2
  pipenv run python predict.py \
  --input $totalradiance \
  --model $modelpath/l-mse_batch128_filters64_vector16_intotal_outup2_scatter/ae_latest.tar \
  --output $outfilepath_up2
  # predict the surface emission component
  echo $outfilepath_surf
  pipenv run python predict.py \
  --input $totalradiance \
  --model $modelpath/l-mse_batch128_filters64_vector16_intotal_outsurf/ae_latest.tar \
  --output $outfilepath_surf
done