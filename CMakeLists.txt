cmake_minimum_required(VERSION 3.19)
project(test LANGUAGES CXX)

if(WIN32)
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    set(TIFF_INCLUDE_DIR "C:/vcpkg/installed/x64-windows/include")
    set(ZLIB_INCLUDE_DIR "C:/vcpkg/installed/x64-windows/include")
    set(LibArchive_INCLUDE_DIR "C:/vcpkg/installed/x64-windows/include")
    set(TIFF_LIBRARY "C:/vcpkg/installed/x64-windows/lib/tiff.lib")
    set(ZLIB_LIBRARY "C:/vcpkg/installed/x64-windows/lib/zlib.lib")
    set(LibArchive_LIBRARY "C:/vcpkg/installed/x64-windows/lib/archive.lib")
    find_package(Qt6 6.5 REQUIRED COMPONENTS Core)
    find_package(Qt6 REQUIRED COMPONENTS
        Widgets
        3DCore
        3DRender
        3DInput
        3DLogic
        3DExtras
        3DAnimation
        Multimedia
        MultimediaWidgets
    )
    find_package(portaudio CONFIG REQUIRED)
    find_package(Opus CONFIG REQUIRED)
    list(APPEND CMAKE_MODULE_PATH "C:/vcpkg/installed/x64-windows/share/ffmpeg")
    find_package(FFMPEG REQUIRED)
    find_package(TIFF REQUIRED)
    find_package(ZLIB REQUIRED)
    find_package(LibArchive REQUIRED)
    find_package(OpenCV REQUIRED)
elseif(UNIX)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
    find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
    find_package(Qt6 REQUIRED COMPONENTS
        Core
        Widgets
        3DCore
        3DRender
        3DExtras
        3DInput
        3DLogic
        3DAnimation
        Multimedia
        MultimediaWidgets
    )
    find_package(nlohmann_json 3.2.0 REQUIRED)
    find_package(PkgConfig REQUIRED)
    find_package(Threads REQUIRED)
    pkg_check_modules(OpenCV REQUIRED opencv4)
    pkg_check_modules(OPUS REQUIRED opus)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
    pkg_check_modules(ZBAR REQUIRED zbar)
    pkg_check_modules(SDL2 REQUIRED sdl2)

    set(VOSK_ROOT "/home/arepo/vosk/vosk-linux-x86_64-0.3.45" CACHE PATH "Path to Vosk installation directory")
    add_library(Vosk::vosk SHARED IMPORTED)
    set_target_properties(Vosk::vosk PROPERTIES
        IMPORTED_LOCATION "${VOSK_ROOT}/libvosk.so"
        INTERFACE_INCLUDE_DIRECTORIES "${VOSK_ROOT}"
    )
endif()

if(WIN32)
    qt_standard_project_setup()
    qt_add_executable(test
        WIN32
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
    )
    target_link_libraries(test PRIVATE
        Qt6::Widgets
        Qt6::3DCore
        Qt6::3DRender
        Qt6::3DLogic
        Qt6::3DExtras
        Qt6::3DAnimation
        Qt6::Multimedia
        Qt6::MultimediaWidgets
        ws2_32
        ${OpenCV_LIBS}
        portaudio
        Opus::opus
        Xinput
        LibArchive::LibArchive
        ZLIB::ZLIB
        TIFF::TIFF
    )
elseif(UNIX)
    set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
    )
    qt_add_executable(test
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        mainwindow.h
        mainwindow.cpp
    )
    target_link_libraries(test PRIVATE
        Qt6::Widgets
        Qt6::3DCore
        Qt6::3DRender
        Qt6::3DLogic
        Qt6::3DExtras
        Qt6::3DAnimation
        Qt6::3DInput
        Qt6::Multimedia
        Qt6::MultimediaWidgets
        ${OpenCV_LIBRARIES}
        ${OPUS_LIBRARIES}
        ${PORTAUDIO_LIBRARIES}
        ${ZBAR_LIBRARIES}
        Threads::Threads
        Vosk::vosk
        ${CMAKE_DL_LIBS}
        pthread
        nlohmann_json::nlohmann_json
        ${SDL2_LIBRARIES}
    )
    target_include_directories(test PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${OPUS_INCLUDE_DIRS}
        ${PORTAUDIO_INCLUDE_DIRS}
        ${ZBAR_INCLUDE_DIRS}
        ${SDL2_INCLUDE_DIRS}
    )
    set_target_properties(test PROPERTIES
        INSTALL_RPATH "${VOSK_ROOT}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

include(GNUInstallDirs)
install(TARGETS test
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}|
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(WIN32)
    qt_generate_deploy_app_script(
        TARGET test
        OUTPUT_SCRIPT deploy_script
        NO_UNSUPPORTED_PLATFORM_ERROR
    )
    install(SCRIPT ${deploy_script})
elseif(UNIX)
    qt_finalize_executable(test)
endif()
