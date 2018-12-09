FROM anapsix/alpine-java
RUN apk update

RUN mkdir -p /opt/tools && cd /opt/tools && \
    wget http://downloads.openmicroscopy.org/bio-formats/5.6.0/artifacts/bftools.zip && \
    unzip bftools.zip && \
    rm bftools.zip

ENV PATH "/opt/tools/bftools:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/jdk/bin"

RUN apk add gcc gfortran python3 python3-dev build-base freetype-dev libpng-dev openblas-dev libffi-dev

RUN apk add jpeg-dev libpng-dev

RUN pip3 install -U pip

RUN pip3 install dtoolbioimage dtool-azure dtool-http

RUN rm -rf /var/cache/apk/*
