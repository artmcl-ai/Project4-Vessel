The Docker Image for the GNN approach can be accessed from: https://hub.docker.com/r/laith123/vessel-seg

Run the command below:
docker run --gpus all -v %cd%\input2:/input -v %cd%\output2:/output vessel-seg
where '%cd%\input2' represents the directory with your inputs & %cd%\output2 represents the directory where you want the prediction
masks to be stored
