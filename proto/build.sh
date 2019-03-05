#!/usr/bin/env bash

protoc -I . comp.proto --go_out=plugins=grpc:. --python_out=.
echo "========= instructions for python ============="
echo "copy python file to virtualenv site package for it to be imported"
echo "for instance here: ~/.venv/lib/python3.6/site-packages"
echo "or ~/.conda/envs/comp/lib/python2.7/site-packages"
echo "where <comp> is the name of conda environment"
echo "or copy it to the folder where other python file is trying to import it"
echo "or copy where ever site packages are stored"
