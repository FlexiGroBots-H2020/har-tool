# For more information, please refer to https://aka.ms/vscode-docker-python
FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-pytorch-cuda-full:v1.5.0

USER root

RUN apt-get update && apt-get install -y python3-opencv wget g++

WORKDIR /wd

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
COPY mmaction2/requirements.txt /wd
COPY mmaction2/requirements/ /wd/requirements/
RUN pip install -r requirements.txt

COPY har_utils.py /wd
COPY har_backbone.py /wd
COPY mmaction2/ /wd/mmaction2/
COPY models/ /wd/models/

WORKDIR /wd/mmaction2
RUN mkdir -p /wd/mmaction2/data
ENV FORCE_CUDA="1"
RUN pip install cython --no-cache-dir
RUN pip install --no-cache-dir -e .
RUN pip install paho-mqtt

RUN pip install openmim
RUN mim install mmcv-full==1.7.0
RUN mim install mmdet==2.28.2  
RUN mim install mmpose==0.29.0
RUN pip install mmengine==0.7.4

WORKDIR /wd
RUN chmod -R 777 /wd

USER jovyan

ENTRYPOINT ["python", "-u", "har_backbone.py"]
