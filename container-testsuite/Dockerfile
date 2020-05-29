FROM fedora

RUN dnf -y install git php-cli php-xml php-json findutils gcc-9.2.1 gcc gcc-c++ make autoconf automake patch expat-devel unzip bzip2

RUN chmod 777 /etc
RUN chmod ugo+w /var

ARG USER_ID
ARG GROUP_ID

RUN groupadd --gid $GROUP_ID user
RUN useradd --uid $USER_ID --gid $GROUP_ID user

USER user

RUN git clone https://github.com/phoronix-test-suite/phoronix-test-suite.git /home/user/phoronix-test-suite
