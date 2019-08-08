FROM anapsix/alpine-java
RUN apk update

RUN mkdir -p /opt/tools && cd /opt/tools && \
    wget http://downloads.openmicroscopy.org/bio-formats/5.6.0/artifacts/bftools.zip && \
    unzip bftools.zip && \
    rm bftools.zip

ENV PATH "/opt/tools/bftools:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/jdk/bin"

RUN apk add gcc gfortran python3 python3-dev build-base freetype-dev libpng-dev openblas-dev libffi-dev zeromq-dev linux-headers

RUN apk add jpeg-dev libpng-dev

RUN pip3 install -U pip

# Force partial rebuild, we'll fix this later
RUN pip3 install scipy numpy
RUN pip3 install dtoolbioimage dtool-http
RUN apk add openssl-dev
RUN pip3 install dtool-azure dtool-ecs
RUN rm -rf /var/cache/apk/*

ENTRYPOINT ["convert_image_dataset"]
