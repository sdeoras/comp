#!/usr/bin/env bash
set -ex
dir=`pwd`
pkg=`basename ${dir}`
#python model.py
mo u -k package:${pkg}
goimports -w -l .