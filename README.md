# Rescue GUI

This repository contains a Qt6 project for a C++ based Graphical User-Interface. It serves as a unified review and processing center for all data coming from the robot's relay program through a [custom implementation](https://github.com/arepo90/ROTAS) of the RTP protocol. 

### Features

- Automatic relay connection handling.
- 4x independent video viewers with CV filters and settings.
- Robot digital twin 3D viewer.
- Sensor dashboard.
- Speech-to-text implementation.
- Automatic Linux/Windows differentiation on CMakeLists.
- In-depth [wiki](wiki.md).

## Dependencies

- Qt6 (6.5 or newer):
  - Core
  - 3D
  - Widgets
- Windows (Vcpkg):
  - OpenCV 
  - PortAudio
  - Opus
  - FFmpeg
  - ZLib
  - Companion program
- Linux:
  - OpenCV core (4.10.x or newer)
  - PortAudio 
  - Opus
  - ZBar
 
## Installation

> WIP

## Usage

Once a connection to the relay has been established, the main window will appear and all widgets will load automatically.

When you wish to terminate the program, simply close the window normally and the shutdown procedure will begin.

The main window is split into three main sections: 
- Video viewers
- 3D viewer
- Sensor dashboard

### Video viewers

These are used to monitor the robot's camera feeds in real time:
- Each viewer is fully independent, and corresponds to a separate channel.
- The two dropdowns at the top allow you to change the current camera feed and CV filter. The default feed is "No Camera", and a valid one needs to be selected in order to change the CV filter.
- For concurrent use with filters, the same camera feed can be used on different viewers at the same time.
- Once a feed is selected, you may click on the viewer itself to enlarge it so it covers the full 2x2 section. To return to the original arrangement, simply click again.
- Some filters may require adjustments on the go, so when selected, a small dashboard will appear underneath the viewer (all changes are applied immediately and automatically).
- The filters include:
  - QR code detection
  - Hazmat image detection
  - Victim crate task shape detection
  - Thermal sensor overlay

### 3D viewer

This is a digital representation of the robot's state in real time, including:
- Robot's orientation in the x, y and z axis.
- Flippers' angles.
- Arm's joints' angles and positions.

For ease of analysis, you may zoom in or out, pan, and rotate the view.

### Sensor dashboard

This is where the numerical sensor data is displayed in real time, including:
- Gas sensor data (in ppm).
- Speech-to-text output (resets on every phrase).
- Magnetometer data in the x, y and z axis (in µT).

At the bottom of the dashboard, two buttons can be found:
- **Toggle audio**: Notifies the relay to start or stop the audio stream, as well as the speech-to-text process.
- **Clear data**: Erases all dashboard data.

## Logs

The program regularly outputs console messages regarding the state of execution. They generally mark the beginning or end of a particularly relevant piece of logic, and are followed by `...` when a blocking function is awaiting a response.

They are, in increasing order of severity:
- **Info `[i]`**: General information; only appear during the startup and shutdown processes, or when the connection status changes.
- **Warning `[w]`**: Data warnings; generally appear due to unexpected or corrupt data from external sources.
- **Error `[e]`**: Fatal errors that end the program's execution or edge cases due to improper shutdown.

## Notes

- Due to dependency issues, the Windows version cannot perform all data processing tasks locally and needs a python _companion_ program running concurrently.
- The first 4 socket ports from `START_PORT` (inclusive) are always used, with increasing pairs proportional to the number of video sources.
- Logs offer a look into the program's execution, but anything other than an `[i] info` message should be treated as an error.
- The 3D viewer may take a few seconds to load properly, time during which the window may appear to hang.
