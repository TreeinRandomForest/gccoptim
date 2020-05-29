FROM fedora

COPY ["requirements.txt", "/requirements.txt"]
RUN dnf -y install which python3-pip git emacs gcc-9.2.1 gcc gcc-c++
RUN pip3 install -r requirements.txt

ARG USER_ID
ARG GROUP_ID

RUN groupadd --gid $GROUP_ID user
RUN useradd --uid $USER_ID --gid $GROUP_ID user

USER user