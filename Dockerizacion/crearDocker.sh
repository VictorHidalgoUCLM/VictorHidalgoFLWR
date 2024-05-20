#!/bin/bash

sudo docker buildx build --platform linux/arm64/v8 -t client . --load --file dockerfile
sudo docker tag client victorhidalgo/client
sudo docker push victorhidalgo/client
