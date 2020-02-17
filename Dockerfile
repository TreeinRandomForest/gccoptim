FROM fedora

RUN dnf -y install git php-cli php-xml php-json findutils gcc-9.2.1 gcc gcc-c++ make autoconf automake patch expat-devel unzip bzip2
RUN git clone https://github.com/phoronix-test-suite/phoronix-test-suite.git
