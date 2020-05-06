# Video Analytics OpenCV


## Note: This is a work in progress!

## Getting Started

### Prerequisites
This was build on linux using Python 3.5.6, but any python > 3.5.6 will do.

See requirements.txt for libraries dependencies.

### CV Models Endpoints
We use both Azure's Computer Vision Cognitive Service "Read" API, and other open-source model on container.

## Deployment

0. Install needed resources

1. Clone the repo

2. Configure needed enviroment variables (see main.py) in launch.json-type file for development or pass to k8s pod via yaml file for deployment

3. build your docker image 

```
docker build --rm -f ".Dockerfile" -t videoanalyticsopencv:vx.x .
```

4. run your docker image
```
docker run --rm -it  videoanalyticsopencv:vx.x
```
