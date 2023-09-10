FROM 817929577935.dkr.ecr.ap-southeast-1.amazonaws.com/de-mlauto-master-base:krisml_python3_torch_1_10_gpu

LABEL maintainer="ITD_DA_DE <ITD_DA_DE@singaporeair.com.sg>"
ARG BUILD_ARG_HTTP_PROXY=HTTP_PROXY
ARG BUILD_ARG_NO_PROXY=NO_PROXY

ARG SIA_ENV
ENV SIA_ENV=$SIA_ENV

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update

RUN apt install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa 

RUN apt-get update && apt-get install -y --no-install-recommends \
     build-essential \
     curl wget \
     unzip \
     nginx \
     ca-certificates \
     make gcc g++ \
     git \
     python3.9-distutils \
     python3.9-dev \
     && \
     rm -rf /var/lib/apt/list/* && \
     rm -f /usr/bin/python && \
     ln -s /usr/bin/python3.9 /usr/bin/python

RUN python --version

RUN wget https://bootstrap.pypa.io/get-pip.py --no-check-certificate \
     && python get-pip.py --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org

COPY requirements.txt /opt/requirements.txt

RUN python -m pip install --upgrade pip
RUN python -m pip install -r /opt/requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV PYTHONIOENCODING='utf-8'
# Azure Key
ENV GPT_API_KEY=
ENV HAYSTACK_TELEMETRY_ENABLED=False


COPY search /opt/program
WORKDIR /opt/program

#RUN pip install gunicorn[tornado,eventlet]
RUN pip install gunicorn[tornado]

ENV HTTP_PROXY ${BUILD_ARG_HTTP_PROXY}
ENV HTTPS_PROXY ${BUILD_ARG_HTTP_PROXY}
ENV http_proxy ${BUILD_ARG_HTTP_PROXY}
ENV https_proxy ${BUILD_ARG_HTTP_PROXY}
ENV no_proxy=${BUILD_ARG_NO_PROXY}
RUN echo $HTTPS_PROXY

RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve

