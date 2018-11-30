#!/usr/bin/env bash
if [[ ! -d "${GOPATH}/src/github.com/tensorflow/tensorflow/" ]]; then
    go get -d -v github.com/tensorflow/tensorflow/...
fi

cd ${GOPATH}/src/github.com/tensorflow/tensorflow/tensorflow/core/protobuf
for d in `ls -1 *.proto`; do
    echo ${d}
    protoc -I . -I ../../../ -I ${GOPATH}/src/github.com/gogo/protobuf/protobuf \
        --go_out=${GOPATH}/src/github.com/sdeoras/go-scicomp/tensorflow ${d}
done

cd ${GOPATH}/src/github.com/tensorflow/tensorflow/tensorflow/core/framework
for d in `ls -1 *.proto`; do
    echo ${d}
    protoc -I . -I ../../../ -I ${GOPATH}/src/github.com/gogo/protobuf/protobuf \
        --go_out=${GOPATH}/src/github.com/sdeoras/go-scicomp/tensorflow ${d}
done

cd ${GOPATH}/src/github.com/tensorflow/tensorflow/tensorflow/core/lib/core
for d in `ls -1 *.proto`; do
    echo ${d}
    protoc -I . -I ../../../../ -I ${GOPATH}/src/github.com/gogo/protobuf/protobuf \
        --go_out=${GOPATH}/src/github.com/sdeoras/go-scicomp/tensorflow ${d}
done

cd ${GOPATH}/src/github.com/sdeoras/go-scicomp/tensorflow
mkdir -p ../vendor
rm -rf ../vendor/*
mkdir -p ./eager
mv github.com ../vendor
mv ./eager_service.pb.go ./eager
