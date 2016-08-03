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
    json.hpp

LIBS += -L/usr/local/lib/ -lbenchmark -larmadillo -L../CUDAdillo -lCUDAdillo -L/Developer/NVIDIA/CUDA-7.5/lib -lcudart -lcublas

INCLUDEPATH += /usr/local/include \
                ../Armuda \
               /Developer/NVIDIA/CUDA-7.5/include
DEPENDPATH += /usr/local/include \
                ../Armuda \
               /Developer/NVIDIA/CUDA-7.5/include

macx: QMAKE_CXXFLAGS += -O3 -std=c++14 -stdlib=libc++ -fno-rtti -Wall -pedantic -Werror
