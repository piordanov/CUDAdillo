QT += core
QT -= gui

CONFIG += c++11

TARGET = BenchMarkcuBLAS
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

LIBS += -L/usr/local/lib/ -lbenchmark -larmadillo -pthread

INCLUDEPATH += /usr/local/include
DEPENDPATH += /usr/local/include

macx: QMAKE_CXXFLAGS += -O3 -std=c++14 -stdlib=libc++ -fno-exceptions -fno-rtti -Wall -pedantic -Werror -pthreads
