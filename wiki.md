# Wiki

The following is a full explanation of the Graphical User Interface program.

## Syntax 

In general, different conventions are used to refer to different concepts:

- **ALL_CAPS**: Constant or permanent declarations, generally stated as preprocessor definitions (i.e. `const` or `#define`).
- **snake_case**: Variables of all kinds, pointers and object instances.
- **camelCase**: Class methods and functions of any kind.
- **PascalCase**: Struct, class, and object names and declarations.

> If you find an instance where this is not true (like a float using camelCase or a function using snake_case), that's a mistake and should be corrected soon.

## Terminology

- A **setting** is the initial state of a variable that alters the how a function or the program as a whole work (it's not modified during the program's execution). 
- A **channel** is an instance of `SocketStruct`, and refers to an object capable of handling two-way data transmission through ROTAS.
- A **payload** is the relevant data or information sent inside a packet.
- A **packet** is an entire data structure meant for transmission. Contains a header and one or more payloads.
- A **header** is a data structure that contains metadata about the rest of the payload, generally placed at the beginning of the packet.
- **Fragmentation** is the process of separating particularly large payloads into sections (fragments) and handling each one as a sub-payload in a different packet.
- A **network port** is a virtual point where a connection starts and/or ends. A socket needs to point to a particular port on any given IP address.
- A **USB port** is a virtual index refering to a physical device connected through USB.
- A **declaration** is the act of creating a variable, object, etc., but not giving it any initial state. 
- An **initialization** is the act of assigning an initial state to a variable, object, etc.
- A **flag** is a boolean or _std::atomic&lt;bool&gt;_ variable that allows a processing loop to be executed (activated = set as _true_, deactivated = set as _false_). The `is_running` flags are activated once on startup and deactivated on shutdown to end all threads, while the `is_active` flags may change throughout the program's execution and serve as a _pause_ or _resume_ indicator.
- **send** refers to all variables, objects, processes, and threads that handle communication from the GUI to the relay.
- **recv** refers to all variables, objects, processes, and threads that handle communication from the relay to the GUI.
- A **filter** is a computer vision processing task applied to a frame (QR code detection, hazmat image detection, thermal image overlay, etc.).
- A **video frame** is an image captured from a video source, streamed through a video channel at standard resolution.
- A **thermal frame** is an image constructed from the 8x8 thermal sensor, streamed through the base channel.
- A **subsection** is an instance of the `SubsectionWidget` video feed viewer.
- **Concurrency control** is a series of techniques used to prevent race conditions and data corruption caused by simultaneous get/set operations on data from different threads (generally marked by the use of _std::atomic_ variables or _std::lock_guard_ clauses). 

## File structure

For convenience (and to follow Qt's templates), the program is split into different files:
- `CMakeLists.txt`: Where the compiler looks first. Contains all dependency links and automatically differences between Windows and Linux.
- `mainwindow.h`: Header file for libraries, classes, and function declarations.
- `mainwindow.cpp`: Source file for libraries, classes, and function initializations (most of the code is here).
- `main.cpp`: Main source file for the project. Links `mainwindow.h` and makes simple function calls (`main()` is called here).

> [!NOTE]
> Addidional `mainwindow.x` files exist with the `_linux` suffix added, but contain the same code logic. 

Some helper directories are also needed:
- `asseets`: Contains all relevant 3D models (`.obj` files) and placeholder images.
- `net`: Hazmat DNN model weights and settings.

## Logs

The program regularly outputs console messages regarding the state of execution. They generally mark the beginning or end of a particularly relevant piece of logic, and are followed by `...` when a blocking function is awaiting a response.

They are, in increasing order of severity:
- **Info `[i]`**: General information; only appear during the startup and shutdown processes, or when the connection status changes.
- **Warning `[w]`**: Data warnings; generally appear due to unexpected or corrupt data from external sources.
- **Error `[e]`**: Fatal errors that end the program's execution or edge cases due to improper shutdown.

> [!TIP]
> The mosts important log messages to watch out for are `[i] Awaiting response...` (program started correctly and is awaiting a connection), `[i] Connection established` (connection to the relay has been established and streams should be able to start), `[i] Closing program...` (window has been closed and shutdown is underway).

## Overview

The GUI is a Qt application, and as such, it is built using Qt's libraries and dependencies. It serves as a review and data processing center for the robot's data, making use of the superior computing power compared to the robot's devices.

The code logic is sepparated into classes that serve as _Widgets_, and each widget is represented by a `QWidget` _container_. Since all Qt operations occur on the _UI thread_, concurrency controls are needed to access and modify shared variables.

The GUI is designed to send a notification through the base channel once when it starts and once again when it shuts down, but the relay itself is not supposed to shut down under normal circumstances (unless a fatal error appears). As such, the GUI can be closed and opened as many times as you want and the connection will be handled automatically so long as the relay is running.

> [!NOTE]
> With the exception of network sockets and controller input, all functions and libraries are cross-platform. The only differences between `mainwindow.x` and `mainwindow_linux.x` are the `#include` headers, the socket implementations inside the _RTPStreamHandler_ class, and the _Controller_ class. The OS is automatically detected by CMake during build, and `main.cpp` will always link the appropiate files. 

On startup, the program performs the following actions:
1. Start the `AppHandler` class.
2. Start the base and audio channels.
3. Await a connection to the relay.
4. Start the video channels.
5. Start the base, audio and video threads.
6. Create and show the main window

It should be noted that audio and video streaming do **not** start immediately after a connection is established; only when a video feed is selected or the audio toggled can the corresponding channel send the `is_active` flag and start receiving data.

A general `cam_map` is saved and updated in real time, containing each of the video feed viewer ID's and their selected video feed ID. When requesting a video feed, the video channels look at the second elements of the map; when forwarding the received frames to the requesting `SubsectionWidget`, they look for the viewer ID on the first elements. If a subsection is not requesting any video feeds, its requested video feed ID value will be `-1`. This is hard to explain, so here's an example:

| Viewer ID (first element in map)    | Video feed ID (second element in map)  |
|-------------------------------------|----------------------------------------|
| 0                                   | 0                                      |
| 1                                   | -1                                     |
| 2                                   | 0                                      |
| 3                                   | 1                                      |

**Interpretation**: Viewers 0 and 2 are requesting images from video feed 0, so a single request is forwarded to `video_channels[0]`; viewer 1 is inactive and viewer 3 is requesting images from video feed 1, so the request is forwarded to `video_channels[1]`. Once an image is received, `video_channels[0]` sees that the requests are coming from viewers 0 and 2, so it forwards the frames to them; `video_channels[1]` sees the request coming from viewer 3, so it forwards the frames to it. If there are other `video_channels` set up, they looked at the map and determined that no requests were being made, so they stay in standby.

Due to the ROTAS implementation, the `recv()` function must be able to handle **any** data type natively. As such, the `RTPStreamHandler` class has a callback function pointer that serves as an extension of the recv function itself, which is why the recv threads only contain a call to `recv()` (the callbacks ensure the correct data type is always cast to and handled appropiately).

### Base channel

- send function (not a loop):
  1. Send `0` or `-1` on startup or shutdown, respectively. 
- recv thread:
  1. Await a packet.
  2. Parse `BasePacket`, video feed names and thermal sensor data.
  3. Update window state and internal buffers.
 
### Audio channel

- send thread:
  1. Create `is_active` flag according to toggled status.
  2. Send to relay.
- recv thread:
  1. Await a packet
  2. Decode audio sample with Opus.
  3. Forward data to speech-to-text model.
  4. Play sample back with PortAudio.
 
### Video channels

- send thread:
  1. Create `is_active` flag according to dropdown status.
  2. Send to relay.
- recv thread:
  1. Await a packet.
  2. Decode frame using OpenCV.
  3. Save frame to internal buffer.
  4. Update frame in corresponding viewers.
 
> [!NOTE]
> To save on bandwith, the flag updates are sent every 500 ms. 

## Initial settings

### ROTAS

These preprocessor definitions directly modify the behavior of the ROTAS stream. They **must**:

- `CLIENT_IP`: point to the relay's IP address (should be set up as static).
- `FRAGMENTATION_FLAG`: use no more than 4 bytes (works as a reference value; either is or isn't present).
- `MAX_UDP_PACKET_SIZE`: be equal or less than the theoretical maximum size of a UDP datagram, without UDP and RTP headers.
- `SAMPLE_RATE`: be a supported value for Opus and PortAudio (for audio capture and playback; in hertz).
- `AUDIO_BUFFER_SIZE`: be a supported value for Opus and PortAudio (for audio fragmentation; in bytes).
- `CFG_PATH`: point to the hazmat model's `.cfg` file (relative to the executable).
- `WEIGHTS_PATH`: point to the hazmat model's `.weights` file (relative to the executable).
- `LABELS_PATH`: point to the hazmat model's `.names` file (relative to the executable).
- `INPUT_SIZE`: be a valid _cv::Size_ (for hazmat detection; in pixels).
- `CONF_THRESH`: be a valid confidence threshold (for hazmat detection; from `0.0f` to `1.0f`).
- `NMS_THRESH`: be a valid non-maximum-suppression threshold (for hazmat detection; from `0.0f` to `1.0f`).

> [!IMPORTANT]
> Excluding the IP address, **all** values must match those found on the relay.

## Helpers

- `BasePacket`: Carries all sensor information (that can be stored on the stack) as floats:
  - Robot orientation in degrees (x, y, z angles)
  - Flipper angles in degrees (left, right)
  - Articulation joint angles in degrees (1-4)
  - Track velocities in meters per second (left, right)
  - Magnetometer readings in micro Teslas (x, y, z axis)
  - Gas sensor data in ppm
- `RTPHeader`: Carries all stream metadata needed for transmission. The first five bitfields (`cc`, `x`, `p`, `version`, `pt`) are not actually used but serve as padding. The other values are:
  - `m`: Marker. Contains the total number of fragments for a given payload (same in all fragments).
  - `seq`: Sequence number. Identifies the current fragment for reassembly.
  - `timestamp`: Not actually a timestamp. Sometimes used as a byte marker similar to `m`.
  - `ssrc`: Synchronization source identifier. Random number unique to each payload (same in all fragments).
- `PayloadType`: Identifies the stream data types as a byte. Not actually used in the current implementation.
- `SocketStruct`: A ROTAS channel. Contains the necessary flags, buffers, threads, and helpers to handle an independent two-way connection:
  - `target_socket`: A pointer to `RTPStreamHandler` that handles the send and recv operations directly.
  - `_thread`: Send and recv processing threads.
  - `is_active`: Pause/Resume stream flag.
  - `is_running`: Startup/Shutdown stream flag.
  - `_data`: Internal data buffers.
  - `data_mutex`: Concurrency control for internal buffers used to handle get/set operations in different threads.

The `nMap()` function maps a value in an input range to its corresponding value in an output range. 

# Class descriptions

## ConsoleWindow

A secondary window meant for the release version of the app, in which QtCreator's _Application Output_ tab is not available. When active, appears on startup and bypases the regular console by showing all logs there instead.

This class inherits `QMainWindow`.

## Controller

XBox controller handler using XInput (Windows) or SDL (Linux).

### Internal variables

- `dead_zone`: An integer indicating the x and y axis central deadzone for joysticks (assuming an overall range from -32768 to 32768).

### Constructor

The class constructor. The dead zone is passed as a parameter and assigned to its internal variable.

### readState()

Returns the current state of the controller as a _std::vector&lt;int&gt;_ in the following order:
1. Left joystick x axis (analog from -255 to 255).
2. Left joystick y axis (analog from -255 to 255).
3. Right joystick x axis (analog from -255 to 255).
4. Right joystick y axis (analog from -255 to 255).
5. Left trigger button (analog from 0 to 255).
6. Right trigger button (analog from 0 to 255).
7. DPad up button (digital).
8. DPad down button (digital).
9. DPad left button (digital).
10. DPad right button (digital).
11. Start button (digital).
12. Back button (digital).
13. Left thumb button (digital).
14. Right thumb button (digital).
15. Left shoulder button (digital).
16. Right shoulder button (digital).
17. A button (digital).
18. B button (digital).
19. X button (digital).
20. Y button (digital).

> [!IMPORTANT]
> Although XInput registers the joystick values in the [-32768, 32768] range, they are mapped to the [-255, 255] range using `nMap()`.

## ModelWidget

The digital twin 3D viewer. Positions and angles are updated automatically with information from `BasePacket`, and the viewer allows for zooming, rotating, and panning.

This class inherits `QWidget`.

### Internal variables

- `root`: Local root entity; the viewport, camera, and all meshes are bound to it as children.
- `container`: Class container `QWidget`.
- `viewport`: 3D window.
- `parts`: _std::vector&lt;Qt3DCore::QEntity*&gt;_ of all object entity pointers.
- `pivots`: _std::vector&lt;Qt3DCore::QTransform*&gt;_ of all pivot transform pointers; used to update part positions and rotations after setup.
- `band_colors`: _std::vector&lt;Qt3DExtras::QPhongMaterial*&gt;_ of track material pointers; used to update velocity indicators after setup.

### Constructor

A parent `QWidget` is passed as an argument, but defaults to `nullptr` if unspeficied.

1. Initializes a local root entity.
2. Declares and initializes a 3D viewport and container.
3. Calls `loadModels()` method.
4. Declares and initializes a camera entity and camera controller.

### Destructor

Deletes the local root entity, also destroying all other entities by inheritance.

### loadModels()

In order to handle the rotation of objects with non-ideal origins, each mesh is bound to an invisible _pivot_ entity. After all initial transformations are applied, any modification is made to the pivots and propagated throughout the model by inheritance. 

1. Declares and initializes a light entity.
2. Prepares mesh setup (file addresses, initial mesh rotations and positions, initial pivot rotations and positions).
3. Declares and initializes pivots and mesh entities
4. Declares and initializes x, y, and z axis indicators

### updatePivot()

Updates a particular pivot's rotation. The pivot index, rotation axis, and angle (in degrees) are passed as parameters.

1. Checks for out-of-bounds pivot index.
2. Updates the selected pivot's rotation in the selected axis.

### updateState()

Updates the model's state and rotation externally. A `BasePacket` instance is passed as an argument.

1. Updates the base x, y, and z orientation angles.
2. Updates the flipper arms pivots using `updatePivot()`.
3. Updates the arm pivots using `updatePivot()`.
4. Maps the track velocities to a `QColor` using `nMap()` (forwards = green, backwards = red; higher velocity = brighter color).
5. Updates the track material colors.

## RTPStreamHandler

The ROTAS stream. Serves as a universal handler for two-way communication of **any** data types and sizes. 

> [!NOTE]
> The original implementation of ROTAS expected a variety of data types on the recv function. In order to comply with this, a callback was set up for each data type (_floatCallback_, _intCallback_, _charCallback_, etc.), so that a single call to `recv()` could handle just the desired data type. This proved to be over-engineered for the GUI, so the current implementation only uses _ucharCallback_ and casts the byte array to whatever data type is needed.

### Internal variables

- `Stream`: A helper struct that stores the stream information. Not actually used in the current implementation.
- Sockets:
  - `SOCKET`: Send and recv socket objects.
  - `sockaddr_in`: Send and recv socket addresses. Bound to the GUI's ip address and two contiguous ports.
  - `socket_address_size`: Helper variable for `sendto()` and `recvfrom()` socket methods.
- `floatCallback`: Continuation of `recv()` that accepts a _std::vector&lt;float&gt;_. Not actually used in the current implementation.
- `ucharCallback`: Continuation of `recv()` that accepts a _std::vector&lt;unsigned char&gt;_. Called only if set up.

### Constructor

An initial port, the target ip address, and the payload type are passed as parameters. 

1. Declares and initializes the `Stream` object (unused).
2. Initializes the `send_socket` object as UDP, and binds `send_socket_address` to the GUI on `port`.
3. Initializes the `recv_socket` object as UDP, starts a temporary 1 MB buffer, and binds the `recv_socket_address` to the GUI on `port+1`.
4. Finishes with an information (`[i]`) message (_Channel created, bound to ports (p, p+1)_).

### Destructor

1. Anounces the call with an `[i]` message.
2. Forcibly shuts down `recv_socket` to prevent a blocking condition.
3. Closes `send_socket` and `recv_socket`.

### destroy()

The actual destructor isn't (and shouldn't) be called as a method. This function serves as an intermediary.

### setUCharCallback()

A function pointer accepting a _std::vector&lt;unsigned char&gt;_ is passed as a parameter. This function will be called at the end of every `recv()` call.

### sendPacket()

Accepts a _std::vector_ of **any** data type as a parameter, which will be encoded as bytes and sent to the GUI.

1. Determines if the payload needs to be fragmented or can be sent in full.
2. Assings a random (not actually but good enough) ssrc to the payload.
3. For every fragment, an `RTPHeader` instance is created and set up. If there is more than one fragment, the `FRAGMENTATION_FLAG` is encoded in the last four bytes of `RTPHeader.seq`.
4. The packet is created as a _std::vector&lt;char&gt;_. The `RTPHeader` is copied first, followed by the current payload fragment.
5. Sends the packet and returns to point 3 if there are multiple packets.

### recvPacket()

Blocking function that awaits a packet and processes in a callback function.

1. Declares a temporary packet buffer.
2. Awaits a packet (blocking function).
3. Parses RTP header and saves relevant metadata.
4. For every fragment, the ssrc is stored and a payload buffer is initialized.
5. If the packet is fragmented, store current fragment in payload buffer and continue; otherwise, set payload as complete.
8. Payload is passed to `ucharCallback`.

## SubsectionWidget

The video feed viewer. Four of these are used in the left section of the GUI in a 2x2 arrangement. Each _subsection_ is independent of the others and can handle any video feed and CV filters by itself.

This class inherits `QWidget`.

To ensure different feeds can access the same video feed at the same time, each subsection carries a particular subsection `id` and a video feed `cam_id`. The ROTAS stream automatically forwards received frames to the requesting subsections.

> [!IMPORTANT]
> Qt really **does not like it** when you update UI elements from anywhere that is not the UI thread. You _can_ do it and it _will_ work for a while, but at some point or another it will lead to segmentation faults and/or internal crashes. Moreover, updating the state of a UI element with a variable (even as a copy) and then derreferencing said variable can have unpredictable effects. For this reason, all frame updates are passed through several internal buffers and effectuated through a Qt _signal_, which greatly decreases the framerate (but there is no other alternative without implementing a vastly overcomplicated widget).

### Widget hierarchy

- container (layout)
  - dropdowns
    - camera_dropdown
    - filter_dropdown
  - camera_view
  - settings
    - qr_container (qr_layout)
      - qr_button_1
      - qr_button_2
    - shape_container (shape_layout)}
      - shape_buttons
    - thermal_container (thermal_layout)
      - thermal_slider_1
      - thermal_slider_2
     
> [!NOTE]
> Although all settings dashboards are set up and added on startup, their containers are hidden until the relevant filter is selected.

### Internal variables

- `filters`: Helper CV filters struct:
  - `none`: No filter active flag.
  - `is_active`: Specific filter active flag.
  - `thermalAdaptiveInterpolation`: Thermal 8x8 sensor processing.
  - `thermalOverlay`: Places processed thermal image over video feed frame.
  - `placeText`: Places text message over video feed frame.
  - `detectQR`: Detects and decodes QR code inside video feed frame.
  - `detectShape`: Detects white shape over black background inside video feed frame.
  - `detectHazmat`: Detects hazmat image inside video feed frame.
  - `colors`: Helper _std::vector&lt;cv::Scalar&gt;_ of colors for image visualization.
  - `labels`: Helper _std::vector&lt;std::string&gt;_ of hazmat image labels.
  - `hazmat_model`: DNN hazmat image detection model.
- `filter_settings`: Helper CV filter settings struct:
  - `_container`: High-level container for settings dashboard.
  - `_layout`: Internal dashboard layout.
  - `_button`: Toggleable setting button.
  - `_slider`: Variable setting slider.
  - `_setting`: Internal setting buffer.
  - `settings_mutex`: Concurrency control for settings used to handle get/set operations in different threads.
- `_dropdown`: Video feed and CV filter options.
- `id`: Subsection widget id.
- `cam_id`: Requested video feed id.
- `camera_view`: Qt's implementation of a frame (as a `QPixmap` inside a `QLabel`). 
- `layout`: General subsection layout.
- `settings`: Combined settings layout.
- `dropdowns`: Dropdown lists layout.
- `container`: High-level subsection container.
- `fullscreen`: Subsection is fullscreen flag.
- `is_cv_running`: CV filters startup/shutdown flag.
- `_mutex`: Concurrency control for internal buffers used to handle get/set operations in different threads.
- `_frame`: Internal frame buffers.
- `cv_thread`: CV filters processing thread.
- `qt_frame`: Another internal frame buffer (more in-depth explanation ahead).
 
### Constructor

1. Declare and initialize container and layouts.
2. Declare and initialize dropdown lists and elements.
3. Deactivate filter flags.
4. Declare and initialize filter settings dashboard.
5. Connect filter settings elements to internal buffers.
6. Initialize and start filter thread.
7. Connect dropdowns to internal settings and buffers.
8. Set placeholder image.
9. Initializer hazmat model.

### Destructor

1. Deactivate all `is_active` and `is_running` flags.
2. Join filter thread.

### setAvailableDevices()

Sets the available video feeds to the dropdown list on startup. The number of video feeds and a _std::vector&lt;std::string&gt;_ names list are passed as arguments.

### updateFrame()

Updates the viewer with the latest frames. The video frame, thermal frame, and compressed video frame (unused) are passed as parameters.

> [!NOTE]
> Although the video and thermal frames are received through different channels, the update function is used for both for convenience, as the thermal frame needs to be updated almost in real time.

1. Update internal frame buffers with _std::lock_guards_
2. Convert _cv::Mat_ to _QImage_.
3. Emit `frameReady` _signal_ with _QImage_ to update in the UI thread.

### mousePressEvent()

Overload of Qt function. Emits `subsectionClicked` signal and proceeds with default behavior.

### placeText()

Filter helper function. A _std::string_ text and a frame are passed as arguments.

1. Determine expected text dimensions.
2. Place text centered in the frame.
3. Return updated frame.

### detectQR()

QR filter function. A video frame is passed as an argument.

1. Declares detected points buffer.
2. Initializes selected QR decoder (Opencv or ZBar according to filter settings).
3. Detects and decodes QR code inside video frame.
4. Draws contour lines on the video frame and writes the decoded text.
5. Returns updated frame.

### detectHazmat()

Hazmat filter function. A video frame is passed as an argument.

1. Declares DNN setup buffers.
2. Detects hazmat image inside the video frame using local `hazmat_model`.
3. Draws contour lines on the video frame and writes the hazmat image's name.
4. Returns updated frame.

### thermalAdaptiveInterpolation()

Thermal sensor processing function. A raw (8x8) thermal frame is passed as an argument.

The function performs several gradient and interpolation techniques to build a more detailed picture and upscale it into standard definition.

### thermalOverlay()

Thermal image filter function (a true filter this time). A video frame, thermal frame, distance and opacity values are passed as arguments.

> [!NOTE]
> Since the thermal sensor and video sources have different fields of view and are placed at different positions, it is not possible to get a true overlay of the thermal image over the video image. To achieve a pseudo-overlay between the frames, the `distance` parameter is given in cm to perform trigonometric calculations and build the overlap frame at the specified distance. This and the opacity (`alpha`) settings can be changed in the filter settings dashboard.

1. Calculates the video and thermal sources fields of view and expected physical capture widths in cm.
2. Calculates the overlap section between the video and thermal frames.
3. Crops the video and thermal frames at the specified dimensions.
4. Overlays the thermal frame's colors over the video frame at the specified opacity.
5. Resizes the final frame to standard definition.
6. Returns frame

###  detectShape()

Shape detection filter function. A video frame is passed as an argument.

> The original implementation expects a victim crate arrangement as the one seen on the Robocup rulebook (and does not work otherwise). A more general implementation is currently under development.

1. WIP
2. Draws contours on the video frame.
3. Returns updated frame.

## MainWindow

What the name implies: the main window and where all of the widgets reside.

This class inherits `QMainWindow`.

### Widget hierarchy

- main_layout
  - left_layout
     - subsections
  - right_layout
    - model
    - dashboard_layout
      - gas_label
      - speech_label
      - magnetometer_label
    - button_layout
        - microphone_button
        - clear_button

### Internal variables

- `_mutex`:  Concurrency control for internal buffers used to handle get/set operations in different threads.
- `_layout`: Internal section layouts.
- `_container`: High-level section containers.
- `_label`: Sensor dashboard labels.
- `_button`: Toggleable dashboard buttons.
- `fullscreen_widget`: Temporary pointer to `SubsectionWidget` instance when set up as fullscreen.
- `subsections`: Pointer list of `SubsectionWidget` instances for video viewers.
- `is_fullscreen`: Fullscreen status flag.
- `model`: Pointer to `ModelWidget` instance for digital twin 3D viewer.
- `cam_map`: Video viewer and video feed identifier.

### Constructor

1. Declare and initialize section layouts and containers.
2. Initialize `SubsectionWidget` instances.
3. Set up fullscreen behavior logic.
4. Connect subsection video feed dropdowns to update `cam_map`.
5. Declare and initialize sensor dashboard labels, layouts and containers.
6. Declare and initialize general settings buttons.
7. Connect general settings buttons to audio channel and sensor dashboard.
8. Initialize `ModelWidget` instance.
9. Assemble `main_layout`.

### closeEvent() [Destructor]

Overload of Qt function.

1. Emits `windowClosing` signal.
2. Destroys subsection widgets.
3. Destroys digital twin viewer widget.
4. Continues with default behavior.

### setCamPorts()

Helper function to set video feed options on startup. The number of video feeds and a list of feed names are passed as arguments, and it forwards them to each `SubsectionWidget`.

### updateFrame()

Helper function that provides the latest video frames. The video feed ID and the compressed video frame are passed as arguments.

1. Decompresses the video frame using OpenCV.
2. Determines which subsections to forward the frame to using the `cam_map`.
3. Gets the latest unprocessed thermal frame from internal buffer.
4. Forwards the video and thermal frames to the requesting subsections.

### updateDashboard()

Helper function that provides the latest sensor data to the dashboard. The label index and the data are passed as arguments.

> [!NOTE]
> Seeing how different data types are present in the dashboard, the function is set up as a template and can automatically identify different data types.

For the selected label, simply updates it with the received data.

### updateThermal()

Helper function that updates the internal buffer with the latest thermal image (unprocessed). A thermal image is passed as an argument and a _std::lock_guard_ is used to update the buffer.

### updateState()

General function to update all relevant widgets after a `BasePacket is received`. An encoded _std::vector&lt;float&gt;_ is passed as an argument.

1. Creates `BasePacket` instance with the encoded data.
2. Updates sensor dashboard labels.
3. Updates digital twin 3D viewer state.

## AppHandler

The main executor, serves as parent class to all others. For convenience, the destructor sets the window up in the background, but no communication nor usage can actually occur until the `init()` method is called and a connection to the relay is established.

### Internal variables

- `_channel`: `SocketStruct` instances for base, audio and companion (Windows only) channels.
- `is_audio_active`: Audio recording flag.
- `port`: Starting ROTAS network port.
- `pa_error`: Required for PortAudio initialization (not actually used).
- `opus_decoder`: Opus decoder instance for audio processing.
- `stream`: PortAudio stream pointer.

### Constructor

1. Initializes `MainWindow` instance.
2. Connects `windowClosing` signal to general destructor.
3. Initializes base channel threads and callback:
  1. Parses encoded data into `BasePacket` instance, video feed names and thermal sensor data.
  2. Updates internal buffers with _std::lock_guard_.
  3. Updates window state with gathered data.
5. Activates base channel flags.
6. Initializes companion channel threads and callback (Windows only):
  1. Updates window state with speech-to-text data.
7. Initializes audio channel threads and callback:
  1. Decodes payload using Opus.
  2. [Windows] Forwards audio sample to companion channel.
  3. [Linux] Forwards audio sample to speech-to-text model.
  4. Plays audio sample using PortAudio
9. Initializes PortAudio and Opus.
10. Activates audio channel flags.
11. Connects window's `cam_map` to video channel flags.

### Destructor

1. Sends shutdown flag to relay through base channel.
2. Deactivates base channel flags.
3. Joins base channel threads.
4. Deactivates companion channel flags (Windows only).
5. Joins base channel threads (Windows only).
6. Deactivates audio channel flags.
7. Joins audio channel threads.
8. Shuts down PortAudio stream.
9. Deactivates video channel flags.
10. Joins video channel threads.

### init()

Called to initialize the application. Awaits a connection to the relay program.

1. Awaits a packet through the base channel.
2. Sends a startup flag through the base channel.
3. Gets number of video sources and forwards information to main window.
4. Initializes video channel send threads:
  1. Gets status from `cam_map`.
  2. Sends `is_active` flag.
5. Initializes video channel recv threads:
  1. Awaits a packet.
6. Initializees video channel callbacks:
  1. Forwards video feed ID and encoded frame to window.
7. Starts PortAudio stream.
8. Initializes audio channel send thread:
  1. Gets status from internal flag.
  2. Sends `is_active` flag.
9. Initializes audio channel recv thread:
  1. Awaits a packet.
10. Initializes companion channel recv thread (Windows only):
  1. Awaits a packet.
12. Shows the window.

## main()

The actual main function called on program startup.

1. Anounces the call with an [i] message (Hi [Windows/Linux]).
2. Declares and initializes `AppHandler` instance.
3. Calls `app_handler.init()`.

> [!TIP]
> It is normal to not see the window after starting the program if the relay program is not active. The number of video feeds is necessary to create the corresponding video channels, but if you wish to skip this and see the window directly, simply comment the lines about the initial handshake on `AppHandler::init()`.

