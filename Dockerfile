FROM gcr.io/kaggle-gpu-images/python:v122

RUN pip install black && \
    pip install jupyter-contrib-nbextensions && \
    jupyter contrib nbextension install && \
    jupyter nbextensions_configurator enable && \
    jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip && \
    jupyter nbextension enable jupyter-black-master/jupyter-black && \
    polars==0.16.8 && \

