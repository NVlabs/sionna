FROM gitlab-master.nvidia.com:5005/nvresearch-gcml/sionna/tensorflow:2.8.4-gpu-jupyter

ENV NVIDIA_DRIVER_CAPABILITIES=all

ADD requirements-2.8.4.txt .
ADD requirements_test.txt .
RUN pip3 --proxy http://repox.mi.nvidia.com:3128 install -r requirements-2.8.4.txt && \
  pip3 --proxy http://repox.mi.nvidia.com:3128 install -r requirements_test.txt && \
  rm requirements_test.txt requirements-2.8.4.txt
