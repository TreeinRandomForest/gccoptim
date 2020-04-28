FROM fedora

COPY ["requirements.txt", "/requirements.txt"]
RUN dnf -y install which python3-pip git emacs
RUN pip3 install -r requirements.txt

#RUN useradd optim
#RUN mkdir -