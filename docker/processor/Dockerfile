FROM ubuntu

RUN apt-get -y update
RUN apt-get -y install python3 python3-pip
RUN pip3 install dtoolbioimage>=0.1.8 scikit-image imageio
RUN pip3 install simpleITK
RUN pip3 install dtool_utils
#RUN pip3 install dtool-azure dtool-ecs
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
