TEMPLATE = subdirs

SUBDIRS += \
    Test \
    CUDAdillo

Test.depends = CUDAdillo
