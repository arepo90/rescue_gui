#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QApplication>
#include <Qt3DWindow>
#include <QForwardRenderer>
#include <QOrbitCameraController>
#include <QCuboidMesh>
#include <QCylinderMesh>
#include <QPhongMaterial>
#include <QDiffuseMapMaterial>
#include <QEntity>
#include <QSlider>
#include <QVBoxLayout>
#include <QWidget>
#include <Qt3DRender/QCamera>
#include <QLabel>
#include <QImage>
#include <QObject>
#include <QMouseEvent>
#include <QPushButton>
#include <QMesh>
#include <QForwardRenderer>
#include <QComboBox>
#include <QStandardItemModel>
#include <QTimer>
#include <QDirectionalLight>
#include <QtLogging>
#include <QMainWindow>
#include <QTextEdit>
#include <QDateTime>

#include <portaudio.h>
#include <opencv2/opencv.hpp>
#include <opus/opus.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <Xinput.h>

#pragma comment(lib, "ws2_32.lib")

#define AUDIO_BUFFER_SIZE 2880
#define SAMPLE_RATE 48000
#define MAX_PACKET_SIZE 65536
#define CLIENT_IP "127.0.0.1"

enum PACKET_TYPE{ SETUP = 0, AUDIO = 1, VIDEO = 2 };

struct RTPHeader {
    uint16_t cc:4;
    uint16_t x:1;
    uint16_t p:1;
    uint16_t version:2;
    uint16_t pt:7;
    uint16_t m:1;
    uint16_t seq_num;
    uint32_t timestamp;
    uint32_t ssrc;
};

enum class PayloadType : uint8_t{
    VIDEO_MJPEG = 97,
    AUDIO_PCM = 98,
    ROS2_ARRAY = 99
};

int nMap(int n, int minIn, int maxIn, int minOut, int maxOut);

class ConsoleWindow : public QMainWindow{
    Q_OBJECT
public:
    explicit ConsoleWindow(QWidget *parent = nullptr);
    void appendMessage(const QString &message);
private:
    QTextEdit* text_edit;
};

class Controller : public QObject{
    Q_OBJECT
public:
    Controller(int dead_zone = 1000);
    ~Controller();
    std::vector<int> readState();
private:
    int dead_zone;
};

class SubsectionWidget : public QWidget{
    Q_OBJECT
public:
    explicit SubsectionWidget(QWidget *parent = nullptr);
    ~SubsectionWidget();
    void destroy(){ delete this; }
    void setAvailableDevices(int num_cams);
    void setFullScreenMode(bool fullScreen);
    void updateAvailableOptions(const QSet<QString> &usedOptions);
    std::pair<int, QString> getCurrentSelection();
    void updateFrame(cv::Mat frame);
    cv::Mat detectShapeContours(const cv::Mat& input_frame);
    cv::Mat detectShapeHough(cv::Mat input_frame);
    cv::Mat detectShapeHybrid(cv::Mat input_frame);
signals:
    void subsectionClicked(SubsectionWidget *widget);
    void selectionChanged();
    void frameReady(QImage image);
protected:
    void mousePressEvent(QMouseEvent *event) override;
private:
    struct Filters{
        std::atomic<bool> none;
        std::atomic<bool> is_qr_active;
        std::atomic<bool> is_hazmat_active;
        std::atomic<bool> is_shape1_active;
        std::atomic<bool> is_shape2_active;
        std::atomic<bool> is_shape3_active;
        std::atomic<bool> is_circles_active;
        cv::QRCodeDetector qr_decoder;
    };
    Filters filters;
    QComboBox *camera_dropdown;
    QComboBox *filter_dropdown;
    int cam_id;
    QLabel* cameraView;
    QVBoxLayout* layout;
    QHBoxLayout* dropdowns;
    QWidget* container;
    std::vector<int> availableDevices;
    bool fullScreen = false;
    std::atomic<bool> is_active;
    std::mutex frame_mutex;
    std::mutex filter_mutex;
    cv::Mat latest_frame;
    cv::Mat filter_frame;
    std::vector<cv::Point> filter_points;
    std::thread cv_thread;
    std::atomic<bool> is_cv_running;
    QImage qt_frame;
};

class ModelWidget : public QWidget{
    Q_OBJECT
public:
    explicit ModelWidget(QWidget *parent = nullptr);
    void updatePivot(int index, int axis, float angle);
    void updateModel(float angleX, float angleY, float angleZ);
    void updateColor(int index, QColor color);
private:
    void loadModels();
    Qt3DCore::QEntity* root;
    QWidget* container;
    Qt3DExtras::Qt3DWindow* viewport;
    std::vector<Qt3DCore::QEntity*> parts;
    std::vector<Qt3DCore::QTransform*> pivots;
    std::vector<Qt3DExtras::QPhongMaterial*> band_colors;
};

class RTPStreamHandler: public QObject{
    Q_OBJECT
    template<typename T>
    using DataCallback = std::function<void(const std::vector<T>&)>;
public:
    RTPStreamHandler(int port, std::string address, PayloadType type, QObject *parent = nullptr);
    ~RTPStreamHandler();
    void setFloatCallback(DataCallback<float> callback){ floatCallback = callback; }
    void setUCharCallback(DataCallback<uchar> callback){ ucharCallback = callback; }
    void destroy(){ delete this; }
    static int audioCallback(const void* input, void* output, unsigned long frameCount, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData);
    int audioProcess(const void* input, void* output, unsigned long frameCount, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags);
    void sendPacket(std::vector<int> data);
    void recvPacket();
private:
    struct Stream{
        uint32_t ssrc;
        uint16_t seq_num;
        uint32_t timestamp;
        PayloadType payload_type;
        int port;
    };
    Stream* stream;
    SOCKET send_socket;
    SOCKET recv_socket;
    sockaddr_in send_socket_address;
    sockaddr_in recv_socket_address;
    int socket_address_size = sizeof(send_socket_address);
    DataCallback<float> floatCallback;
    DataCallback<uchar> ucharCallback;
    OpusDecoder* opus_decoder;
};

struct SocketStruct{
    RTPStreamHandler* target_socket;
    std::thread send_thread;
    std::thread recv_thread;
    std::atomic<bool> is_active;
    std::atomic<bool> is_send_running;
    std::atomic<bool> is_recv_running;
    std::vector<float> float_data;
    std::vector<int> int_data;
    std::mutex data_mutex;
    // --- Thingamajig to transfer std::thread ownership ---
    SocketStruct() : target_socket(nullptr) {}
    SocketStruct(SocketStruct&& other) noexcept
        : recv_thread(std::move(other.recv_thread)),
        send_thread(std::move(other.send_thread)),
        target_socket(std::move(other.target_socket)){}
    SocketStruct& operator=(SocketStruct&& other) noexcept {
        if(this != &other){
            recv_thread = std::move(other.recv_thread);
            send_thread = std::move(other.send_thread);
            target_socket = std::move(other.target_socket);
        }
        return *this;
    }
    SocketStruct(const SocketStruct&) = delete;
    SocketStruct& operator=(const SocketStruct&) = delete;
};

class MainWindow : public QWidget{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void updateState(std::vector<float> data);
    void updateFrame(int id, std::vector<unsigned char> data);
    void setCamPorts(int num_cams);
    template<typename T> void updateDashbord(int index, T data);
signals:
    void windowClosing();
    void selectionChanged(std::map<int, int> cam_map);
    void buttonChanged(bool is_active);
protected:
    void closeEvent(QCloseEvent *event) override;
private:
    QHBoxLayout* main_layout;
    QGridLayout* left_layout;
    QGridLayout* dashboard_layout;
    QVBoxLayout* right_layout;
    QWidget* left_container;
    QLabel* sensor_label;
    QLabel* gas_label;
    QLabel* qr_label;
    QLabel* speech_label;
    QLabel* magnetometer_label;
    QPushButton* microphone_button;
    QPushButton* clear_button;
    std::vector<SubsectionWidget*> subsections;
    bool is_fullscreen;
    ModelWidget* model;
    std::vector<int> scanVideoCaptureDevices();
    std::map<int, int> cam_map;
};

// --- DEPRECATED CLASS - USE RTPSTREAMHANDLER INSTEAD ---
/*
class RTPServer : public QObject{
    Q_OBJECT
template <typename T>
using DataCallback = std::function<void(const std::vector<T>&)>;
public:
    RTPServer(int port, PayloadType type, QObject *parent = nullptr);
    ~RTPServer();
    void setFloatCallback(std::function<void(std::vector<float>)> cb) { callback = cb; }
    void setUCharCallback(DataCallback<uchar> callback){ ucharCallback = callback; }
    void destroy(){ delete this; }
    static int audioCallback(const void* input, void* output, unsigned long frameCount, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData);
    int audioProcess(const void* input, void* output, unsigned long frameCount, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags);
    void sendPacket(std::vector<int> data);
    void startListening();
private slots:
    std::vector<int> handshake();
private:
    struct Stream {
        uint32_t ssrc;
        uint16_t seq_num;
        PayloadType type;
        std::mutex frame_mutex;
        bool is_initialized;
    };
    DataCallback<float> floatCallback;
    DataCallback<uchar> ucharCallback;
    SOCKET client_socket;
    sockaddr_in socket_address;
    int socket_address_size = sizeof(socket_address);
    PACKET_TYPE packet_type;
    std::thread listening_thread;
    std::atomic<bool> is_running;
    Stream* stream;
    OpusDecoder* opus_decoder;
    RTPHeader* header;
    std::function<void(std::vector<float>)> callback;
};
*/

class AppHandler : public QObject{
    Q_OBJECT
public:
    AppHandler(int port, QObject *parent = nullptr);
    ~AppHandler();
    void init();
    void destroy(){ delete this; }
private:
    PaError PaErrorCallback(const char *errorText, PaHostApiTypeId hostApiType, PaHostErrorInfo* hostErrorInfo){ return 0; }
    SocketStruct* base_socket;
    std::vector<SocketStruct*> video_sockets;
    std::atomic<bool> is_audio_active;
    int port;
    int pa_error;
    MainWindow* window;
    SocketStruct* audio_socket;
    OpusDecoder* opus_decoder;
    PaStream* stream;
};

#endif
