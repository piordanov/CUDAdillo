QT += core
QT -= gui

CONFIG += c++14
QMAKE_MACOSX_DEPLOYMENT_TARGET = 10.11

TARGET = BenchMarkcuBLAS
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

HEADERS += \
    helper_cuda.h \
    armacudawrapper.h

LIBS += -L/usr/local/lib/ -lbenchmark -larmadillo

INCLUDEPATH += /usr/local/include
DEPENDPATH += /usr/local/include

macx: QMAKE_CXXFLAGS += -O3 -std=c++14 -stdlib=libc++ -fno-exceptions -fno-rtti -Wall -pedantic -Werror

DISTFILES += \
    cudautilities.cu \
    cudautilities.cuh

CUDA_SOURCES += cudautilities.cu

CUDA_DIR = "/Developer/NVIDIA/CUDA-7.5"


SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_30           # (tested with sm_30 on my comp) Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math


# include paths
INCLUDEPATH += $$CUDA_DIR/include

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/

CUDA_OBJECTS_DIR = ./


# Add the necessary libraries
CUDA_LIBS = -lcudart -lcublas # <-- changed this

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so') <-- didn't need this
LIBS += $$CUDA_LIBS # <-- needed this


# SPECIFY THE R PATH FOR NVCC (this caused me a lot of trouble before)
QMAKE_LFLAGS += -Wl,-rpath,$$CUDA_DIR/lib # <-- added this
NVCCFLAGS = -Xlinker -rpath,$$CUDA_DIR/lib # <-- and thisß

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}




