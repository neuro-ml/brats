FROM debian:stretch
# ENV LAST_UPDATED 21-07-2017
RUN apt-get update && apt-get install -y \
	git \
	wget \
        bzip2

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh 
RUN ./Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3
RUN chmod -R 777 /miniconda3
RUN rm ./Miniconda3-latest-Linux-x86_64.sh
ENV PATH "/app":"/miniconda3/bin:${PATH}":$PATH

RUN conda install scipy==1.0.0 
RUN conda install pytorch-cpu==0.3.0 -c pytorch

RUN pip install nibabel==2.1.0

RUN mkdir app && git clone https://github.com/neuro-ml/deep_pipe.git /app/deep_pipe && cd /app/deep_pipe/ && git checkout 41da0997d21090e9e7ec348aacb037303590c451 && ln -s /app/deep_pipe/dpipe /miniconda3/lib/python3.6/site-packages

COPY ./root/app /app
COPY predict.py /app/
