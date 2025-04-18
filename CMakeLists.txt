cmake_minimum_required(VERSION 3.19)
project(bruh LANGUAGES CXX)

set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
#set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "")

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
add_definitions(-DHAVE_OPENCV_CUDA=1)
#find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    imgcodecs
    videoio
    objdetect
    cudaarithm
    cudafilters
    cudaimgproc
)

#find_package(OpenCV REQUIRED)

qt_standard_project_setup()

qt_add_executable(bruh
    WIN32
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
)

target_link_libraries(bruh PRIVATE
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

include(GNUInstallDirs)

install(TARGETS bruh
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET bruh
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
