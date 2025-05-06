#ifndef MAINWINDOW_H
#define MAINWINDOW_H

// --- qt ---
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

// --- c++ ---
#include <string>
#include <portaudio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <opus/opus.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <Xinput.h>
#include <windows.h>
#include <iostream>
#include <random>
#include <fstream>
#pragma comment(lib, "ws2_32.lib")

// --- Comms settings ---
#define AUDIO_BUFFER_SIZE 960       // 960 bytes
#define SAMPLE_RATE 16000           // 16 kHz
#define MAX_PACKET_SIZE 65536       // 65539 bytes
#define CLIENT_IP "127.0.0.1"       //"192.168.0.238"
#define MAX_UDP_PACKET_SIZE 65507   // 65507 bytes
#define FRAGMENTATION_FLAG 0x8000   // RTP Header flag

// --- Hazmat settings ---
#define CFG_PATH "../../net/yolo.cfg"
#define WEIGHTS_PATH "../../net/yolo.weights"
#define LABELS_PATH "../../net/labels.names"
#define INPUT_SIZE cv::Size(416, 416)
#define CONF_THRESH 0.8f
#define NMS_THRESH 0.4f

// --- Rotas prereqs ---
enum PACKET_TYPE{ SETUP = 0, AUDIO = 1, VIDEO = 2 };

struct RTPHeader {
    uint16_t cc:4;
    uint16_t x:1;
    uint16_t p:1;
    uint16_t version:2;
    uint16_t pt:1;
    uint16_t m;
    uint16_t seq;
    uint16_t timestamp;
    uint16_t ssrc;
};

struct BasePacket{
    float body_x = 0;
    float body_y = 0;
    float body_z = 0;
    float arm_l = 0;
    float arm_r = 0;
    float art_1 = 0;
    float art_2 = 0;
    float art_3 = 0;
    float art_4 = 0;
    float track_l = 0;
    float track_r = 0;
    float magnetometer_x = 0;
    float magnetometer_y = 0;
    float magnetometer_z = 0;
    float gas_ppm = 0;
};

enum class PayloadType : uint8_t{
    VIDEO_MJPEG = 97,
    AUDIO_PCM = 98,
    ROS2_ARRAY = 99
};

// --- Helper func ---
int nMap(float n, float minIn, float maxIn, float minOut, float maxOut);

// --- Class declarations ---

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

class ModelWidget : public QWidget{
    Q_OBJECT
public:
    explicit ModelWidget(QWidget *parent = nullptr);
    ~ModelWidget();
    void updatePivot(int index, int axis, float angle);
    void updateModel(float angleX, float angleY, float angleZ);
    void updateColor(int index, QColor color);
    void destroy(){ delete this; }
public slots:
    void updateState(BasePacket model_state);
private:
    void loadModels();
    Qt3DCore::QEntity* root = nullptr;
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
    template <typename T> void sendPacket(std::vector<T> data, int marker = 0, int delay = 0);
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

class SubsectionWidget : public QWidget{
    Q_OBJECT
public:
    explicit SubsectionWidget(int id, QWidget *parent = nullptr);
    ~SubsectionWidget();
    void destroy(){ delete this; }
    void setAvailableDevices(int num_cams);
    void setFullScreenMode(bool fullScreen){
        this->fullScreen = fullScreen;
        QPixmap current = camera_view->pixmap();
        camera_view->setPixmap(current.scaled((fullScreen ? QSize(960, 720) : QSize(480, 360)), Qt::KeepAspectRatio));
    }
    void updateAvailableOptions(const QSet<QString> &usedOptions);
    std::pair<int, QString> getCurrentSelection(){ return std::make_pair(cam_id, camera_dropdown->currentText()); }
    void updateFrame(cv::Mat frame, std::vector<uchar> compressed = {});
signals:
    void subsectionClicked(SubsectionWidget *widget);
    void selectionChanged();
    void frameReady(QImage image);
    void destructorCalled(int id);
protected:
    void mousePressEvent(QMouseEvent *event) override;
private:
    struct Filters{
        std::atomic<bool> none;
        std::atomic<bool> is_qr_active;
        std::atomic<bool> is_hazmat_active;
        std::atomic<bool> is_shape_active;
        std::atomic<bool> is_circles1_active;
        std::atomic<bool> is_circles2_active;
        cv::Mat placeText(std::string text, cv::Mat frame);
        cv::Mat detectQR(cv::Mat frame);
        cv::Mat detectShape(cv::Mat frame);
        cv::Mat detectCircles1(cv::Mat frame);
        cv::Mat detectCircles2(cv::Mat frame);
        cv::Mat detectHazmat(cv::Mat frame);
        std::vector<cv::Scalar> colors;
        std::vector<std::string> labels;
        cv::dnn::DetectionModel hazmat_model;
    };
    Filters filters;
    QComboBox *camera_dropdown;
    QComboBox *filter_dropdown;
    int cam_id;
    int id;
    int num_cams;
    QLabel* camera_view;
    QVBoxLayout* layout;
    QHBoxLayout* dropdowns;
    QWidget* container;
    std::vector<int> availableDevices;
    bool fullScreen = false;
    std::atomic<bool> is_cv_running;
    std::mutex frame_mutex;
    std::mutex filter_mutex;
    std::mutex compressed_mutex;
    std::mutex thermal_mutex;
    cv::Mat latest_frame;
    cv::Mat filter_frame;
    std::vector<uchar> latest_compressed;
    std::thread cv_thread;
    QImage qt_frame;
};

class MainWindow : public QWidget{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    void updateState(std::vector<float> data);
    void updateFrame(int id, std::vector<unsigned char> data);
    void setCamPorts(int num_cams);
    template<typename T> void updateDashbord(int index, T data);
signals:
    void windowClosing();
    void selectionChanged(std::map<int, int> cam_map);
    void buttonChanged(bool is_active);
    void destructorCalled(int id);
    void modelUpdated(BasePacket model_state);
protected:
    void closeEvent(QCloseEvent *event) override;
private:
    QHBoxLayout* main_layout;
    QHBoxLayout* button_layout;
    QGridLayout* left_layout;
    QGridLayout* dashboard_layout;
    QVBoxLayout* right_layout;
    QWidget* left_container;
    QLabel* sensor_label;
    QLabel* gas_label;
    QLabel* speech_label;
    QLabel* magnetometer_label;
    QPushButton* microphone_button;
    QPushButton* clear_button;
    SubsectionWidget* fullscreen_widget = nullptr;
    std::vector<SubsectionWidget*> subsections;
    bool is_fullscreen;
    ModelWidget* model = nullptr;
    std::vector<int> scanVideoCaptureDevices();
    std::map<int, int> cam_map;
};

class AppHandler : public QObject{
    Q_OBJECT
public:
    AppHandler(int port, QObject *parent = nullptr);
    ~AppHandler();
    void init();
    void destroy(){ delete this; }
private:
    PaError PaErrorCallback(const char *errorText, PaHostApiTypeId hostApiType, PaHostErrorInfo* hostErrorInfo){ return 0; }
    SocketStruct* base_channel;
    SocketStruct* audio_channel;
    //CompanionStruct* vosk_channel;
    SocketStruct* vosk_channel;
    std::vector<SocketStruct*> video_channels;
    std::atomic<bool> is_audio_active;
    int port;
    int pa_error;
    MainWindow* window;
    OpusDecoder* opus_decoder;
    PaStream* stream;
};

#endif
