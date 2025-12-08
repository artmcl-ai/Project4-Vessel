The Docker Image for the GNN method can be pulled from: https://hub.docker.com/r/laith123/vessel-seg

The command to run this image is: docker run --gpus all -v %cd%\input2:/input -v %cd%\output2:/output vessel-seg
where %cd%\input2 repesents the folder on your device with the input images and %cd%\output2 represents the folder that you want to output the predictions to
