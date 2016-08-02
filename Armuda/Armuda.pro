#-------------------------------------------------
#
# Project created by QtCreator 2016-08-02T09:48:23
#
#-------------------------------------------------

QT       -= core gui

TARGET = Armuda
TEMPLATE = lib
CONFIG += staticlib


HEADERS += armuda.h \
unix { \
    cudautilities.cuh
    target.path = /usr/lib
    INSTALLS += target
}

DISTFILES += \
    cudautilities.cu


