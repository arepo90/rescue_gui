/*
    ROBOTEC GUI - 2025

    DISCLAIMER:
    APP WILL ONLY RUN WITH THE FOLLOWING DEPENDENCIES:
    - QT6 + WIDGETS + 3D + RANDOM STUFF I DONT REMEMBER
    - OPENCV (vcpkg)
    - PORTAUDIO (vcpkg)
    - OPUS (vcpkg)
    - XINPUT (windows sdk)

    ----> THE MAIN WINDOW WILL NOT APPEAR UNLESS CONNECTION WITH RELAY PROGRAM IS ESTABLISHED <----

    CHECK CONSOLE OUTPUTS FOR MORE INFO
    IVE ONLY EVER TESTED THIS IN MY DEVICE, SO IT MAY OR MAY NOT WORK WITH DIFFERENT CONFIGS
*/

#include "mainwindow.h"
#include <QApplication>

/*
    TODO:
    Thermal overlay HERE (partial, refactor needed)
    Filter settings
    Fix filters (new qr, shape)
*/

// WORK IN PROGRESS OK ?😭😭😭😭😭


/*
    // XINPUT TESTS
    controller_socket = new RTPServer(8000, PayloadType::AUDIO_PCM);
    controller = new Controller(1200);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::thread controller_thread = std::thread(callback);
    controller_thread.join();
    //RTPServer* base_socket = new RTPServer(8000, PayloadType::ROS2_ARRAY);
    RTPServer* audio_socket = new RTPServer(8001, PayloadType::AUDIO_PCM);
    RTPServer* video_socket = new RTPServer(8002, PayloadType::VIDEO_MJPEG);
    Pa_Initialize();
    PaStream* stream;
    Pa_OpenDefaultStream(&stream, 0, 1, paInt16, SAMPLE_RATE, 2880, RTPServer::audioCallback, audio_socket);
    Pa_StartStream(stream);
    qDebug() << "done";
    //base_socket->setFloatCallback(std::bind(&MainWindow::updateState, window, std::placeholders::_1));
    //audio_socket->setUCharCallback(std::bind(&AudioPlayer::decodePlay, player, std::placeholders::_1));
    video_socket->setUCharCallback(std::bind(&MainWindow::updateFrame, window, std::placeholders::_1));
    QObject::connect(window, &MainWindow::windowClosing, [&base_socket, &stream, &audio_socket, &video_socket](){
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        //base_socket->destroy();
        audio_socket->destroy();
        video_socket->destroy();
        WSACleanup();
    });
*/


//ConsoleWindow *console = nullptr;

int main(int argc, char* argv[]){
    QApplication app(argc, argv);

    // console window for debugging while on release exec.
    /*console = new ConsoleWindow();
    console->resize(500, 300);
    console->show();
    qInstallMessageHandler([](QtMsgType type, const QMessageLogContext &context, const QString &msg){
        if(!console) return;
        QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss.zzz");
        QString formatted_msg;
        if(type == QtMsgType::QtCriticalMsg || type == QtMsgType::QtFatalMsg)
            formatted_msg = QString("<span style='color:red'>[%1] [ERROR] %2</span>").arg(timestamp, msg);
        else if (type == QtMsgType::QtWarningMsg)
            formatted_msg = QString("<span style='color:orange'>[%1] [WARNING] %2</span>").arg(timestamp, msg);
        else
            formatted_msg = QString("<span style='color:white'>[%1] [INFO] %2</span>").arg(timestamp, msg);
        console->appendMessage(formatted_msg);
    });*/

    qInfo() << "Hi";
    WSAData wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
    AppHandler* app_handler = new AppHandler(8000);
    app_handler->init();

    return app.exec();
}
