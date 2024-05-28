#!/bin/bash

sudo docker buildx build --platform linux/arm64/v8 -t client . --load --file dockerfile2
sudo docker tag client victorhidalgo/client
sudo docker push victorhidalgo/client
