The Docker Image for the GNN approach can be accessed from: https://hub.docker.com/r/laith123/vessel-seg

Run this command:
**docker run --gpus all -v %cd%\input2:/input -v %cd%\output2:/output vessel-seg**


where '%cd%\input2' represents the directory with your inputs & %cd%\output2 represents the directory where you want the prediction
masks to be stored

ENSURE THAT THESE DIRECTORIES EXIST if you do not create the directories and store your inputs in the input folder then the docker will not run!
The inputs should be .nii.gz images of non-contrast CT images.

The models are not stored in this repository, but they are contained in the image.
