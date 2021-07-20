FROM ubuntu:20.04

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

# install curl
RUN echo 8 | apt-get install -y tzdata
# remove python 3.9 if was ever there
RUN apt-get remove --purge python3 
# install python3.8
RUN apt-get install -y curl python3.8 python3-distutils php php-zip php-curl

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --force-reinstall && \
    rm get-pip.py

WORKDIR /gp
COPY . /gp
# Install pynsgp python package
RUN python3 -m pip install .
RUN ln -s /usr/bin/python3 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip
# start localhost
CMD php -S 0.0.0.0:4200 -t /gp/interface/

