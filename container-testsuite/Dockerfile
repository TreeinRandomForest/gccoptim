FROM fedora

RUN dnf -y install git php-cli php-xml php-json findutils gcc gcc-c++ make autoconf automake patch expat-devel unzip bzip2

RUN chmod 777 /etc
RUN chmod ugo+w /var

RUN useradd user
RUN mkdir /home/user/store && chown user:user /home/user/store

RUN git clone https://github.com/phoronix-test-suite/phoronix-test-suite.git /home/user/phoronix-test-suite
VOLUME /home/user/store

