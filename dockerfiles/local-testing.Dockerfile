FROM ubuntu:16.04

# RUN apt-get update && apt-get install -y --no-install-recommends \
#         git-all
#         && \
#     apt-get clean && \

ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} install --upgrade \
    pip \
    setuptools \
    keras \
    ipython \
    notebook \
    matplotlib \
    google-cloud-storage

ARG TF_PACKAGE=tensorflow
RUN ${PIP} install ${TF_PACKAGE}
