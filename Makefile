IMAGE_NAME := agatan/python-science

.PHONY: build
build: Dockerfile
	docker build . -t $(IMAGE_NAME)
