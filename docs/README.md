# OMPL
This doc notices globally the installation about ompl with python binding(open motion planning library)
## Site of git: https://github.com/ompl/ompl
## Tutorial: https://ompl.kavrakilab.org/tutorials.html
## Contributions: https://ompl.kavrakilab.org/thirdparty.html

## Set up steps:
```
git clone --recursive https://github.com/ompl/ompl.git
mkdir build && cd build
cmake ..
make -j4
make py_ompl
make -j4 update_bindings (long time to wait)
sudo make install
```

### Notation: the directory 'ompl' is compiled by lzd's personal PC env, the root path should be changed !

