# Install dependencies
apt-get -y update
apt-get -y install libeigen3-dev libcppunit-dev
# Build from source
git clone https://github.com/orocos/orocos_kinematics_dynamics.git
cd orocos_kinematics_dynamics/orocos_kdl/
mkdir build && cd build
cmake ..
make && make install
# Install python-bindings
cd ../..
git submodule update --init
cd python_orocos_kdl/ && mkdir build && cd build
cmake ..
make && make install

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.9/dist-packages
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib" >> /root/.bashrc
ldconfig
