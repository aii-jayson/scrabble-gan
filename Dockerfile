# Need devel version cause we need /usr/include/cudnn.h
# for compiling libctc_decoder_with_kenlm.so
FROM tensorflow/tensorflow:latest

ENV DEBCONF_NOWARNINGS yes

RUN apt-get update
# for opencv
RUN apt-get install -y --no-install-recommends libsm6 libxext6 libxrender-dev

# for mysqlclient
RUN apt-get install -y --no-install-recommends libmysqlclient-dev libssl-dev

# for utility
RUN apt-get install -y --no-install-recommends git vim
RUN apt-get install -y screen less
RUN apt-get install -y --no-install-recommends  net-tools  iptables htop

# for ssh connection
RUN apt-get install -y openssh-server
RUN apt-get install -y openssh-client
RUN apt-get install -y autossh

# for ja_JP.utf8
RUN apt-get install -y locales

# install python library
RUN pip3 install -U pip
RUN pip3 --no-cache-dir install awscli
RUN pip3 --no-cache-dir install pydot imageio matplotlib graphviz
RUN pip3 --no-cache-dir install opencv-python
RUN pip3 --no-cache-dir install gin-config


# for as you need
RUN pip3 --no-cache-dir install gpustat

# https://qiita.com/YumaInaura/items/7509061e4b27e03ea538
RUN mkdir /var/run/sshd
RUN mkdir /root/.ssh
COPY id_rsa.pub /root/.ssh/authorized_keys
RUN echo 'root:password' | chpasswd
RUN chmod 600 /root/.ssh/authorized_keys

RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/^#PermitRootLogin yes/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN echo "X11Forwarding yes" >> /etc/ssh/sshd_config
RUN echo "UsePAM yes" >> /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
RUN rm -rf /var/lib/apt/lists/*

# for inheritance from host proxy settings
RUN touch /root/.bash_profile
RUN echo "export http_proxy=$http_proxy" >> /root/.bash_profile
RUN echo "export https_proxy=$https_proxy" >> /root/.bash_profile
RUN echo ". ~/.bashrc" >> /root/.bash_profile

# for ja_JP.utf8
# https://qiita.com/kazuyoshikakihara/items/0cf74c11d273b0064c83
RUN echo "ja_JP UTF-8" > /etc/locale.gen
RUN locale-gen
RUN echo "export LANG=ja_JP.UTF-8" >> /root/.bash_profile

# for scp
# https://qiita.com/montblanc18/items/b93fa4082e3bc2702a7f
# https://qiita.com/U_ikki/items/b86ce318cb7c086bb6c1
# This script is provisional depending on /etc/profile
RUN sed -i '1s/^/if [[ $- != *i* ]]; then return; fi\n/' /etc/bash.bashrc

# for github
COPY config /root/.ssh/config
COPY id_rsa /root/.ssh/id_rsa

#COPY /raid/workspace/jayson/scrabble-gan /root/scrabble-gan/
EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]

