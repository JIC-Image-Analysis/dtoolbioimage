dtoolbioimage
=============

This repository provides bioimaging integration for dtool. In particular it contains:

1. A tool to convert a dtool dataset containining raw microscope files into a dataset with organised PNG images and microscope metadata.

2. A library and related tools for loading and interacting with images from the dataset.

Using the converter container
-----------------------------

..code-block:: bash

    docker run --env-file convert-env.list jicscicomp/dbiconverter \
    azure://jicinformaticsrawdata/b86c43ee-dd76-44cd-977b-07e426a5ac43 \
    azure://imagedatasets \
    jingyi_protoplast_measurement_feb_2019_ids
