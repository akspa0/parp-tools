SOURCES += \
    src/main.cpp \
    src/window.cpp \
    src/settingswindow.cpp \
    src/model.cpp \
    src/model_object.cpp \
    src/material.cpp \
    src/binary_io.cpp \
    src/WMO_exporter.cpp \
    src/bsptreegenerator.cpp

HEADERS += \
    include/window.h \
    include/settingswindow.h \
    include/model.h \
    include/model_object.h \
    include/material.h \
    include/material-inl.h \
    include/model_object-inl.h \
    include/binary_io.h \
    include/3D_types.h \
    include/WMO_exporter.h \
    include/bsptreegenerator.h

INCLUDEPATH += include/

RESOURCES += \
    ressources/ressources.qrc

QMAKE_CXXFLAGS += -Wall

RC_FILE = ressources/mm.rc
