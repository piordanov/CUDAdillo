TEMPLATE = subdirs

SUBDIRS += \
    Test \
    Armuda

Test.depends = Armuda
