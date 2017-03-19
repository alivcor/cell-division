FROM bvlc/caffe:cpu
MAINTAINER Abhinandan Dubey <adubey@cs.stonybrook.edu>

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt-get update \
	&& apt-get install -y wget make g++ vim python-pip python-dev \
	&& apt-get install -y build-essential

ADD . /caffeapp
WORKDIR /caffeapp

EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["python-scripts/crfasrnn_demo.py"]