IMAGE_NAME := agatan/python-science
DOCKER_FILES := $(shell find dockerfiles -type f)

.PHONY: build
build: dockerfiles/Dockerfile.cpu dockerfiles/Dockerfile.gpu
	docker build . -f dockerfiles/Dockerfile.cpu -t $(IMAGE_NAME):cpu
	docker build . -f dockerfiles/Dockerfile.gpu -t $(IMAGE_NAME):gpu
