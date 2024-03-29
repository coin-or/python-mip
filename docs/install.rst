.. _chapInstall:

Installation
============

Python-MIP requires Python 3.5 or newer. Since Python-MIP is included in the `Python Package Index <https://pypi.org>`_, once you have a `Python installation <https://www.python.org/downloads>`_, installing it is as easy as entering in the command prompt:

.. code-block:: sh

    pip install mip

If the command fails, it may be due to lack of permission to install globally available Python modules. In this case, use:

.. code-block:: sh

    pip install mip --user

The default installation includes pre-compiled libraries of the MIP Solver `CBC <https://projects.coin-or.org/Cbc>`_ for Windows, Linux and MacOS.
If you have the commercial solver `Gurobi <http://gurobi.com>`_ installed in your computer, Python-MIP will automatically use it as long as it finds the Gurobi dynamic loadable library. Gurobi is free for academic use and has an outstanding performance for solving MIPs. Instructions to make it accessible on different operating systems are included bellow.


Gurobi Installation and Configuration (optional)
------------------------------------------------

For the installation of Gurobi you can look at the `Quickstart guide <https://www.gurobi.com/documentation/quickstart.html>`_ for your operating system. Python-MIP will automatically find your Gurobi installation as long as you define the :code:`GUROBI_HOME` environment variable indicating where Gurobi was installed.

Pypy installation (optional)
----------------------------

Python-MIP is compatible with the just-in-time Python compiler `Pypy <https://pypy.org>`_.
Generally, Python code executes much faster in Pypy.
Pypy is also more memory efficient.
To install Python-MIP as a Pypy package, just call (add :code:`--user` may be necessary also):

.. code-block:: sh

    pypy3 -m pip install mip

Using your own CBC binaries (optional)
--------------------------------------

Python-MIP provides CBC binaries for 64 bits versions of MacOS, Linux and Windows that run on Intel hardware. These binaries may not be suitable for you in some cases:

    a) if you plan to use Python-MIP in another platform, such as the Raspberry Pi, a 32 bits operating system or FreeBSD, for example;

    b) if you want to build CBC binaries with special optimizations for your hardware, i.e., using the :code:`-march=native` option in GCC, you may also want to enable some optimizations for CLP, such as the use of the parallel :code:`AVX2` instructions, available in modern hardware;

    c) if you want use CBC binaries built with debug information, to help elucidating some bug. 

In the `CBC page <https://github.com/coin-or/Cbc>`_ page there are instructions on how to build CBC from source on Unix like platforms and on Windows. `Coinbrew <https://github.com/coin-or/coinbrew>`_ is a script that makes it easier the task of downloading and building CBC and its dependencies. The commands bellow can be used to download and build CBC on Ubuntu Linux, slightly different packages names may be used in different distributions. Comments are included describing some possible customizations.

.. code-block:: sh
    
    # install dependencies to build
    sudo apt-get install gcc g++ gfortran libgfortran-9-dev liblapack-dev libamd2 libcholmod3 libmetis-dev libsuitesparse-dev libnauty2-dev git
    # directory to download and compile CBC
    mkdir -p ~/build ; cd ~/build
    # download latest version of coinbrew
    wget -nH https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
    # download CBC and its dependencies with coinbrew
    bash coinbrew fetch Cbc@master --no-prompt
    # build, replace prefix with your install directory,  add --enable-debug if necessary
    bash coinbrew build Cbc@master --no-prompt --prefix=/home/haroldo/prog/ --tests=none --enable-cbc-parallel --enable-relocatable


Python-MIP uses the :code:`CbcSolver` shared library to communicate with CBC. In Linux, this file is named :code:`libCbcSolver.so`, in Windows and MacOS the extension should be :code:`.dll` and :code:`.dylp`, respectively. To force Python-MIP to use your freshly compiled CBC binaries, you can set the :code:`PMIP_CBC_LIBRARY` environment variable, indicating the full path to this shared library. In Linux, for example, if you installed your CBC binaries in :code:`/home/haroldo/prog/`, you could use:

.. code-block:: sh

    export PMIP_CBC_LIBRARY="/home/haroldo/prog/lib/libCbcSolver.so"

Please note that CBC uses multiple libraries which are installed in the same directory. You may also need to set one additional environment variable specifying that this directory also contains shared libraries that should be accessible. In Linux and MacOS this variable is :code:`LD_LIBRARY_PATH`, on Windows the :code:`PATH` environment variable should be set.

.. code-block:: sh

    export LD_LIBRARY_PATH="/home/haroldo/prog/lib/":$LD_LIBRARY_PATH

In Linux, to make these changes persistent, you may also want to add the :code:`export` lines to your :code:`.bashrc`.

Docker installation (optional)
------------------------------

It is also possible to containerize the above build process using Docker. The following dockerfile shows how to build CBC for Python-MIP for an linux/arm/v6 platform (i.e., a Raspberry Pi 2 B). The dockerfile starts from Alpine Linux, which requires slightly different libraries than the Debian libraries above. Depending on your :code:`requirements.txt`, you may need to install additional libraries in the :code:`apk add` command. The dockerfile does not include the optional dependencies of CBC (:code:`libamd2 libcholmod3 libmetis-dev libsuitesparse-dev libnauty2-dev`).

.. code-block:: sh

    # syntax=docker/dockerfile:1
    FROM arm32v6/python:3.7-alpine3.15 AS builder
    RUN apk add --no-cache \
        bash \
        gcc \ 
        gfortran \
        git \
        g++ \  
        libffi-dev \ 
        libgfortran \
        lapack-dev \
        make \ 
        patch
    RUN wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
    RUN chmod u+x coinbrew
    RUN ./coinbrew fetch Cbc@master
    RUN ./coinbrew build Cbc@master --prefix=/home/haroldo/prog/ --tests=none --enable-cbc-parallel --enable-relocatable
    COPY requirements.txt requirements.txt
    RUN mkdir /pip-install && pip3 install --prefix=/pip-install -r requirements.txt

    FROM arm32v6/python:3.7-alpine3.15
    RUN apk add --no-cache \
        libffi-dev \ 
        libgfortran \
        lapack-dev \
        libstdc++6
    COPY --from=builder /home/haroldo/prog /home/haroldo/prog/
    COPY --from=builder /pip-install /usr/local
    COPY . .
    ENV PMIP_CBC_LIBRARY="/home/haroldo/prog/lib/libCbc.so"
    ENV PATH=$PATH:/home/haroldo/prog/bin
    RUN chmod u+x ./entrypoint.sh
    ENTRYPOINT ["./entrypoint.sh"]
    
There are two ways to build this dockerfile. The first option is to build on the same device as where your run the code. In case of the Raspberry Pi, you need a lot of patience (more than 12 hours) to build using the following command:

.. code-block:: sh

    docker build -t <tag> . 

The second option is to build on a fast device and deploy on another. Most likely, your development machine does not have the linux/arm/v6 architecture, and you require cross-compilation with :code:`buildx`. This option requires an account at `Docker Hub <https://hub.docker.com/>`_. You can run the following to build the code:

.. code-block:: sh

    docker buildx create --name mybuilder
    docker buildx use mybuilder
    docker buildx inspect --bootstrap
    docker login
    docker buildx build --platform linux/arm/v6 -t <your-docker-hub-username>/<reponame> . --push
    

