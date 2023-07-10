#!/bin/bash
echo "## Docker  'Build Docker containing the ML Model'"

BASE_NAME="explore_ai_demo"
echo -e "Build started on `date`"
echo -e "\nBuilding the Docker image ..."
docker-compose -f docker/docker-compose.yml build

echo -e "\nCreate container for style analysis ..."
docker-compose -f docker/docker-compose.yml run $BASE_NAME ./scripts/style_analysis.sh

echo -e "\nCreate container for static analysis ..."
docker-compose -f docker/docker-compose.yml run $BASE_NAME ./scripts/static_analysis.sh
export PYTHONUNBUFFERED=TRUE

echo -e "\nCreate container to preprocess data ..."
docker-compose -f docker/docker-compose.yml run $BASE_NAME python3 ./explore_ai_demo/preprocess.py

echo -e "\nGet the latest container id ..."
CONTAINER_ID=$(echo $(docker ps -a | awk '{print $1}') | awk '{print $2}')

echo -e "\nCopy the data trained in the container to the host ..."
mkdir -p ./data && docker cp $CONTAINER_ID:/opt/ml/processing/train/train_featuresMBPTP.csv ./data \
&& docker cp $CONTAINER_ID:/opt/ml/processing/train/train_labelsMBPTP.csv ./data \
&& docker cp $CONTAINER_ID:/opt/ml/processing/train/train_weightMBPTP.csv ./data

echo -e "\nCreate container to train data ..."
docker-compose -f docker/docker-compose.yml run -v "//$(PWD)/data:/opt/ml/processing/train" $BASE_NAME python3 ./explore_ai_demo/train.py

echo -e "\nGet the latest container id ..."
CONTAINER_ID=$(echo $(docker ps -a | awk '{print $1}') | awk '{print $2}')

echo -e "\nCopy the model trained in the container to the host ..."
mkdir -p ./models && docker cp $CONTAINER_ID:/opt/ml/model/model.joblib ./models

echo -e "\nCreate container for unit test and test coverage ..."
docker-compose -f docker/docker-compose.yml run -v "//$(PWD)/models:/opt/ml/model" $BASE_NAME ./scripts/coverage.sh

echo -e "\nRun container to serve on port 8080"
IMAGE_NAME=$(echo $(docker ps -a | awk '{print $2}') | awk '{print $2}')
docker run --rm -v //$(pwd)/models:/opt/ml/model -p 8080:8080 $IMAGE_NAME python3 ./serve