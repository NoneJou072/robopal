FROM python:3.9
# 拉取一个基础镜像，基于python3.8
WORKDIR /code/roboimi
# 设置工作目录，也就是下面执行 ENTRYPOINT 后面命令的路径
ENV LANG C.UTF-8
# 设置语言为utf-8
RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone 
# 设置容器时间，有的容器时区与我们的时区不同，可能会带来麻烦
ADD . /code/roboimi/
# 将你的项目文件放到docker容器中的/code/bdtools文件夹，这里code是在根目录的，与/root /opt等在一个目录
# 这里的路径，可以自定义设置，主要是为了方便对项目进行管理
RUN wget -P /code https://cmake.org/files/v3.26/cmake-3.26.4-linux-x86_64.tar.gz \
    && tar -zxvf /code/cmake-3.26.4-linux-x86_64.tar.gz -C /code \
    && mv /code/cmake-3.26.4-linux-x86_64 /code/cmake-3.26.4 \
    && ln -sf /code/cmake-3.26.4/bin/* /usr/bin
# 安装 cmake
RUN sh /code/roboimi/kdl_install.sh
# 安装 PyKDL
RUN /usr/local/bin/pip3 install -e .
# 根据requirement.txt下载好依赖包
