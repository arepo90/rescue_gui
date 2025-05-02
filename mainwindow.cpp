#include "mainwindow.h"

// --- Helper funcs ---
int nMap(float n, float minIn, float maxIn, float minOut, float maxOut){
    return (n - minIn) / (maxIn - minIn) * (maxOut - minOut) + minOut;
}

// --- Controller (xbox) WIP ---
Controller::Controller(int dead_zone){
    this->dead_zone = dead_zone;
}

Controller::~Controller(){
    return; // bruh
}

std::vector<int> Controller::readState(){
    XINPUT_STATE state;
    XInputGetState(0, &state);
    std::vector<int> states;
    states.push_back(state.Gamepad.sThumbLX);
    states.push_back(state.Gamepad.sThumbLY);
    states.push_back(state.Gamepad.sThumbRX);
    states.push_back(state.Gamepad.sThumbRY);
    for(int i = 0; i < states.size(); i++){
        if(std::abs(states[i]) < dead_zone)
            states[i] = 0;
        else
            states[i] = nMap(states[i], -32768, 32768, -255, 255);
    }
    states.push_back(state.Gamepad.bLeftTrigger);
    states.push_back(state.Gamepad.bRightTrigger);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_UP) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_DOWN) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_LEFT) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_RIGHT) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_START) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_BACK) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_THUMB) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_THUMB) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_A) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_B) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_X) ? 1 : 0);
    states.push_back((state.Gamepad.wButtons & XINPUT_GAMEPAD_Y) ? 1 : 0);
    return states;
}

// --- 3D viewer ---
ModelWidget::ModelWidget(QWidget *parent) : QWidget(parent){
    root = new Qt3DCore::QEntity();
    viewport = new Qt3DExtras::Qt3DWindow();
    viewport->setRootEntity(root);
    viewport->defaultFrameGraph()->setClearColor(QColor("#202020"));
    container = QWidget::createWindowContainer(viewport, this);
    //container->setMinimumSize(QSize(1280, 720));
    container->setMinimumSize(QSize(320, 360));
    //container->setMaximumSize(QSize(320, 360));
    this->loadModels();
    Qt3DRender::QCamera *camera = viewport->camera();
    camera->lens()->setPerspectiveProjection(45.0f, 16.0f/9.0f, 0.1f, 1000.0f);
    camera->setPosition(QVector3D(2.0f, 2.0f, 2.0f));
    camera->setViewCenter(QVector3D(0, 0.0f, 0));
    camera->setUpVector(QVector3D(0.0f, 1.0f, 0.0f));
    Qt3DExtras::QOrbitCameraController *cam_controller = new Qt3DExtras::QOrbitCameraController(root);
    cam_controller->setLinearSpeed(10.0f);
    cam_controller->setLookSpeed(180.0f);
    cam_controller->setCamera(camera);
    container->show();
    QTimer* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [this](){ container->update(); });
    timer->start(1000);
}

ModelWidget::~ModelWidget(){
    if(root){
        delete root;
        root = nullptr;
    }
}

void ModelWidget::loadModels(){
    Qt3DCore::QEntity *light_entity = new Qt3DCore::QEntity(root);
    Qt3DRender::QDirectionalLight *directional_light = new Qt3DRender::QDirectionalLight(light_entity);
    directional_light->setColor("white");
    directional_light->setIntensity(0.75);
    directional_light->setWorldDirection(QVector3D(-1.0, -1.0, -1.0));
    light_entity->addComponent(directional_light);
    std::vector<QString> mesh_addresses = {
        "../../assets/body_nobands.obj",
        "../../assets/left_arm.obj",
        "../../assets/right_arm.obj",
        "../../assets/band.obj",
        "../../assets/band.obj",
        "../../assets/parts/seg1.obj",
        "../../assets/parts/seg2.obj",
        "../../assets/parts/seg3.obj",
        "../../assets/parts/seg4.obj",
        "../../assets/parts/seg5.obj"
    };
    std::vector<QQuaternion> mesh_rotations = {
        QQuaternion::fromEulerAngles(0, 0, 0),
        QQuaternion::fromEulerAngles(0, 0, 0),
        QQuaternion::fromEulerAngles(0, 0, 0),
        QQuaternion::fromEulerAngles(0, 0, 0),
        QQuaternion::fromEulerAngles(0, 0, 0),
        QQuaternion::fromEulerAngles(0, 0, 0),
        QQuaternion::fromEulerAngles(0, 0, 0),
        QQuaternion::fromEulerAngles(0, 0, 0),
        QQuaternion::fromEulerAngles(0, 0, 0),
        QQuaternion::fromEulerAngles(0, 0, 0),
    };
    std::vector<QVector3D> mesh_translations = {
        QVector3D(0, 0, 0),
        QVector3D(-0.27, -0.08, 0.0),
        QVector3D(-0.27, -0.08, 0.0),
        QVector3D(-0.02, -0.08, 0.0),
        QVector3D(-0.02, -0.08, 0.0),
        QVector3D(-0.29, -0.14, -0.34),
        QVector3D(-0.29, -0.25, -0.38),
        QVector3D(-0.29, -0.54, -0.38),
        QVector3D(-0.29, -0.61, -0.37),
        QVector3D(-0.29, -0.79, -0.37)
    };
    std::vector<QQuaternion> pivot_rotations = {
        QQuaternion::fromEulerAngles(0, 0, 0),
        QQuaternion::fromEulerAngles(0, 0, 0),
        QQuaternion::fromEulerAngles(180, 0, 0),
        QQuaternion::fromEulerAngles(180, 0, 0),
        QQuaternion::fromEulerAngles(180, 0, 0),
        QQuaternion::fromEulerAngles(0, -45, 0),
        QQuaternion::fromEulerAngles(0, 0, 45),
        QQuaternion::fromEulerAngles(0, 0, 45),
        QQuaternion::fromEulerAngles(0, 0, 0),
        QQuaternion::fromEulerAngles(0, 90, 0)
    };
    std::vector<QVector3D> pivot_translations = {
        QVector3D(0, 0, 0),
        QVector3D(0.085, 0.087, 0.555),
        QVector3D(0.085, 0.087, 0.145),
        QVector3D(0, 0.087, 0.145),
        QVector3D(0, 0.087, 0.617),
        QVector3D(0.29, 0.14, 0.34),
        QVector3D(0, 0.1, 0),
        QVector3D(0, 0.29, 0),
        QVector3D(0, 0.07, -0.01),
        QVector3D(0, 0.18, 0)
    };
    Qt3DExtras::QPhongMaterial *mesh_material = new Qt3DExtras::QPhongMaterial();
    mesh_material->setDiffuse(QColor("#a6a6a6"));
    // part init
    for(int i = 0; i < mesh_addresses.size(); i++){
        Qt3DCore::QEntity *pivot_entity = new Qt3DCore::QEntity((i == 0 ? root : (i <= 5 ? parts[0] : parts.back())));
        Qt3DCore::QTransform *pivot_transform = new Qt3DCore::QTransform(pivot_entity);
        pivot_transform->setTranslation(pivot_translations[i]);
        pivot_transform->setRotation(pivot_rotations[i]);
        pivot_entity->addComponent(pivot_transform);
        Qt3DCore::QEntity *mesh_entity = new Qt3DCore::QEntity(pivot_entity);
        Qt3DCore::QTransform *mesh_transform = new Qt3DCore::QTransform(mesh_entity);
        Qt3DRender::QMesh *mesh = new Qt3DRender::QMesh();
        Qt3DExtras::QPhongMaterial* band_material = new Qt3DExtras::QPhongMaterial();
        band_material->setDiffuse(Qt::black);
        mesh->setSource(QUrl::fromLocalFile(mesh_addresses[i]));
        mesh_transform->setTranslation(mesh_translations[i]);
        mesh_transform->setRotation(mesh_rotations[i]);
        mesh_entity->addComponent(mesh);
        mesh_entity->addComponent((i == 3 || i == 4  ? band_material : mesh_material));
        mesh_entity->addComponent(mesh_transform);
        parts.push_back(pivot_entity);
        pivots.push_back(pivot_transform);
        if(i == 3 || i == 4) band_colors.push_back(band_material);
    }
    // axis init
    for(int i = 0; i < 3; i++){
        Qt3DExtras::QCylinderMesh *segment = new Qt3DExtras::QCylinderMesh();
        Qt3DCore::QEntity *axis_entity = new Qt3DCore::QEntity(root);
        Qt3DExtras::QPhongMaterial *axis_material = new Qt3DExtras::QPhongMaterial();
        Qt3DCore::QTransform *transform = new Qt3DCore::QTransform();
        QVector3D initial_translation((i == 0 ? 0.5f : 0.0f), (i == 1 ? 0.5f : 0.0f), (i == 2 ? 0.5f : 0.0f));
        QQuaternion initial_rotation = QQuaternion::fromEulerAngles(0.0f, (i == 2 ? 90.0f : 0.0f), (i != 1 ? 90.0f : 0.0f));
        segment->setRadius(0.001f);
        segment->setLength(1.0f);
        axis_material->setAmbient(i == 0 ? Qt::red : (i == 1 ? Qt::green : Qt::blue));
        transform->setTranslation(initial_translation);
        transform->setRotation(initial_rotation);
        axis_entity->addComponent(segment);
        axis_entity->addComponent(transform);
        axis_entity->addComponent(axis_material);
    }
}

// ONLY FOR BASE (ROOT) ENTITY
void ModelWidget::updateModel(float angleX, float angleY, float angleZ){
    pivots[0]->setRotation(QQuaternion::fromEulerAngles(angleX, angleY, angleZ));
}

// ONLY FOR ARTICULATION (PIVOT) ENTITIES
void ModelWidget::updatePivot(int index, int axis, float angle){
    if(index >= pivots.size()){
        qCritical() << "MODEL UPDATE PIVOT | Invalid pivot index: out of bounds";
        return;
    }
    if(axis == 0)
        pivots[index]->setRotation(QQuaternion::fromEulerAngles(angle, pivots[index]->rotationY(), pivots[index]->rotationZ()));
    else if(axis == 1)
        pivots[index]->setRotation(QQuaternion::fromEulerAngles(pivots[index]->rotationX(), angle, pivots[index]->rotationZ()));
    else if(axis == 2)
        pivots[index]->setRotation(QQuaternion::fromEulerAngles(pivots[index]->rotationX(), pivots[index]->rotationY(), angle));
    else
        qCritical() << "MODEL UPDATE PIVOT | Invalid pivot axis: out of bounds";
}

void ModelWidget::updateColor(int index, QColor color){
    if(index >= band_colors.size()){
        qCritical() << "MODEL UPDATE COLOR | Invalid part index: out of bounds";
        return;
    }
    band_colors[index]->setDiffuse(color);
}

cv::Mat SubsectionWidget::Filters::placeText(std::string text, cv::Mat frame){
    int temp = 0;
    cv::Size text_size = cv::getTextSize(text,  cv::FONT_HERSHEY_SIMPLEX, 3.5, 3, &temp);
    cv::putText(frame, text, cv::Point((frame.cols - text_size.width)/2, (frame.rows + text_size.height)/2),  cv::FONT_HERSHEY_SIMPLEX, 3.5, cv::Scalar(255, 0, 0), 3);
    return frame;
}

// --- cam subsections ---
SubsectionWidget::SubsectionWidget(int id, QWidget *parent) : QWidget(parent){
    is_local.store(false);
    this->id = id;
    container = new QWidget(this);
    layout = new QVBoxLayout();
    dropdowns = new QHBoxLayout();
    camera_view = new QLabel();
    camera_dropdown = new QComboBox();
    camera_dropdown->addItem("No Camera");
    filter_dropdown = new QComboBox();
    filter_dropdown->addItems({ "No filter", "QR Code", "Hazmat", "Shape - Hough", "Circles - Gap", "Circles - Rays" });
    filter_dropdown->setEnabled(false);
    cam_id = -1;
    dropdowns->addWidget(camera_dropdown);
    dropdowns->addWidget(filter_dropdown);
    layout->addLayout(dropdowns);
    layout->addWidget(camera_view);
    layout->setSpacing(0);
    layout->setContentsMargins(0, 0, 0, 0);
    container->setLayout(layout);
    container->setStyleSheet("border: 0.5px solid gray;");
    is_cv_running.store(true);
    filters.none.store(true);
    filters.is_qr_active.store(false);
    filters.is_shape_active.store(false);
    filters.is_circles1_active.store(false);
    filters.is_circles2_active.store(false);
    filters.is_hazmat_active.store(false);
    filter_channel = new SocketStruct;
    filter_channel->target_socket = new RTPStreamHandler(9000 + id*2, "127.0.0.1", PayloadType::VIDEO_MJPEG);
    filter_channel->is_recv_running.store(true);
    filter_channel->is_send_running.store(true);
    long long start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    cv_thread = std::thread([this](){
        while(is_cv_running.load()){
            if(!is_local.load()){
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }
            if(filters.none.load()){
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                frame = latest_frame;
            }
            if(frame.empty()){
                qDebug() << "empty frame";
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            if(filters.is_qr_active.load())
                frame = filters.detectQR(frame);
            else if(filters.is_hazmat_active.load())
                frame = filters.placeText("HAZMAT NOT AVAILABLE LOCALLY", frame);
            else if(filters.is_shape_active.load())
                frame = filters.detectShape(frame);
            else if(filters.is_circles1_active.load())
                frame = filters.detectCircles1(frame);
            else if(filters.is_circles2_active.load())
                frame = filters.detectCircles2(frame);
            else{
                qWarning() << "SUBSECTION " << this->id << " CV LOOP | Invalid marker: no active filter";
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(filter_mutex);
                filter_frame = frame;
            }
        }
    });
    filter_channel->target_socket->setUCharCallback([this, id, start](std::vector<uchar> data){
        if(data.empty()){
            qCritical() << "SUBSECTION " << this->id << " PY CALLBACK | Invalid payload: data buffer empty";
            return;
        }
        auto curr = std::chrono::high_resolution_clock::now();
        cv::Mat frame = cv::imdecode(data, cv::IMREAD_COLOR);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        std::lock_guard<std::mutex> lock(filter_mutex);
        filter_frame = frame;
    });
    filter_channel->send_thread = std::thread([this](){
        while(filter_channel->is_send_running.load()){
            if(is_local.load()){
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }
            if(filters.none.load()){
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            std::vector<uchar> compressed_data;
            {
                std::lock_guard<std::mutex> lock(compressed_mutex);
                compressed_data = latest_compressed;
            }
            if(compressed_data.empty()){
                qDebug() << "empty compressed";
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            int marker = 0;
            if(filters.is_qr_active.load())
                marker = 1;
            else if(filters.is_hazmat_active.load())
                marker = 2;
            else if(filters.is_shape_active.load())
                marker = 3;
            else if(filters.is_circles1_active.load())
                marker = 4;
            else if(filters.is_circles2_active.load())
                marker = 5;
            else{
                qWarning() << "SUBSECTION " << this->id << " PY LOOP | Invalid marker: no active filter";
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            filter_channel->target_socket->sendPacket(compressed_data, marker);
            //qDebug() << "sending " << compressed_data.size() << " bytes, marker: " << marker;
        }
    });
    filter_channel->recv_thread = std::thread([this](){
        while(filter_channel->is_recv_running.load()){
            if(is_local.load()){
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }
            filter_channel->target_socket->recvPacket();
        }
    });
    connect(camera_dropdown,  &QComboBox::currentIndexChanged, this, [this](int index){
        cam_id = index - 1;
        emit selectionChanged();
        if(index == 0){
            filter_dropdown->setCurrentIndex(0);
            filter_dropdown->setEnabled(false);
            this->updateFrame(cv::imread("../../assets/404.png"));
        }
        else
            filter_dropdown->setEnabled(true);
    });
    connect(filter_dropdown, &QComboBox::currentIndexChanged, this, [this](int index){
        filters.none.store(index == 0);
        filters.is_qr_active.store(index == 1);
        filters.is_hazmat_active.store(index == 2);
        filters.is_shape_active.store(index == 3);
        filters.is_circles1_active.store(index == 4);
        filters.is_circles2_active.store(index == 5);
    });
    camera_view->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    camera_view->setPixmap(QPixmap("../../assets/404.png").scaled((this->fullScreen ? QSize(960, 720) : QSize(480, 360)), Qt::KeepAspectRatio));
    /*
    cv_thread = std::thread([this](){
        while(is_cv_running.load()){
            if(filters.none.load()){
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            if(filters.is_qr_active.load()){
                std::string decodedText;
                std::vector<cv::Point> points;
                cv::Mat frame;
                {
                    std::unique_lock<std::mutex> lock(frame_mutex);
                    if(latest_frame.empty()){
                        qDebug() << "qr empty latest";
                        lock.unlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                        continue;
                    }
                    frame = latest_frame;
                }
                decodedText = filters.qr_decoder.detectAndDecode(frame, points);
                if(!decodedText.empty())
                    qDebug() << "QR Code: " << decodedText;
                else
                    qDebug() << "qr empty";
                std::lock_guard<std::mutex> lock(filter_mutex);
                //filter_frame = frame;
                filter_points.clear();
                filter_points = points;
            }
            else if(filters.is_shape1_active.load() || filters.is_shape2_active.load() || filters.is_shape3_active.load()){
                cv::Mat frame, result;
                {
                    std::unique_lock<std::mutex> lock(frame_mutex);
                    if(latest_frame.empty()){
                        qDebug() << "empty frame on filter";
                        lock.unlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                        continue;
                    }
                    frame = latest_frame;
                }
                if(filters.is_shape1_active.load())
                    result = filters.detectShapeHough(frame);
                else if(filters.is_shape2_active.load())
                    result = filters.detectShapeContours(frame);
                else if(filters.is_shape3_active.load())
                    result = filters.detectShapeHybrid(frame);
                std::lock_guard<std::mutex> lock(filter_mutex);
                filter_frame = result;
            }
            else if(filters.is_circles_active.load()){
                // implementation missing
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
            else if(filters.is_hazmat_active.load()){
                cv::Mat frame;
                {
                    std::unique_lock<std::mutex> lock(frame_mutex);
                    if(latest_frame.empty()){
                        qDebug() << "empty frame on filter";
                        lock.unlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                        continue;
                    }
                    frame = latest_frame;
                }

                std::lock_guard<std::mutex> lock(filter_mutex);
                filter_frame = frame;
            }
        }
    });
    */
    connect(this, &SubsectionWidget::frameReady, this, [this](QImage image){
        if(!image.isNull())
            camera_view->setPixmap(QPixmap::fromImage(qt_frame).scaled((this->fullScreen ? QSize(960, 720) : QSize(480, 360)), Qt::KeepAspectRatio));
        else
            qCritical() << "SUBSECTION " << this->id << " FRAME READY | Invalid image: frame is null";
    });
}

cv::Mat SubsectionWidget::Filters::detectQR(cv::Mat frame){
    std::vector<cv::Point> points;
    cv::QRCodeDetector qr_decoder;
    std::string text = qr_decoder.detectAndDecode(frame, points);
    if(!text.empty()){
        std::vector<std::vector<cv::Point>> contour = { points };
        cv::polylines(frame, contour, true, cv::Scalar(0, 0, 255), 5);
        cv::putText(frame, text, points[0]+cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
    }
    else
        frame = this->placeText("NO QR CODE DETECTED", frame);
    return frame;
}

cv::Mat SubsectionWidget::Filters::detectCircles1(cv::Mat frame){
    double scale = 4;
    int rad_checks = 20;

    cv::Mat gray_frame, temp;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    cv::resize(gray_frame, temp, cv::Size(), 1.0/scale, 1.0/scale, cv::INTER_AREA);

    std::vector<cv::Vec3f> ext_circles;
    HoughCircles(temp, ext_circles, cv::HOUGH_GRADIENT, 1, temp.rows/8, 100, 50, temp.rows/8, temp.rows/4);

    double min_dis = DBL_MAX;
    cv::Vec3f ext_sector;

    if (!ext_circles.empty()) {
        for (size_t i = 0; i < ext_circles.size(); i++) {
            ext_circles[i][0] *= scale;
            ext_circles[i][1] *= scale;
            ext_circles[i][2] *= scale;

            cv::Point center(cvRound(ext_circles[i][0]), cvRound(ext_circles[i][1]));
            int radius = cvRound(ext_circles[i][2]);
            float dis = pow(frame.cols - center.x, 2) + pow(frame.rows - center.y, 2);

            if (dis < min_dis) {
                min_dis = dis;
                ext_sector = ext_circles[i];
            }
        }
    }

    if (min_dis == DBL_MAX) {
        return this->placeText("MISSING TASK SECTOR", frame);
    }

    cv::Mat ext_mask = cv::Mat::zeros(frame.size(), CV_8UC1), frame_roi;
    circle(ext_mask, cv::Point(cvRound(ext_sector[0]), cvRound(ext_sector[1])), cvRound(ext_sector[2]), cv::Scalar(255), -1);
    bitwise_and(gray_frame, gray_frame, frame_roi, ext_mask);
    cv::Rect roi1 = cv::boundingRect(ext_mask);
    frame_roi = frame_roi(roi1);

    if (frame_roi.empty() || frame_roi.rows < 8 || frame_roi.cols < 8) {
        return this->placeText("MISSING TASK SECTOR", frame);
    }

    std::vector<cv::Vec3f> inn_circles;
    cv::HoughCircles(frame_roi, inn_circles, cv::HOUGH_GRADIENT, 1, frame_roi.rows/8, 100, 50, frame_roi.rows/8, frame_roi.rows/3);

    if (inn_circles.empty()) {
        return this->placeText("MISSING TASK SECTOR", frame);
    }

    cv::Mat mask_roi = cv::Mat::zeros(frame_roi.size(), CV_8UC1), final_roi, thresh, kernel = cv::Mat::ones(3, 3, CV_8UC1);
    cv::circle(mask_roi, cv::Point(cvRound(inn_circles[0][0]), cvRound(inn_circles[0][1])), cvRound(inn_circles[0][2]) - 10, cv::Scalar(255), -1);
    cv::bitwise_and(frame_roi, frame_roi, final_roi, mask_roi);
    cv::Rect roi2 = cv::boundingRect(mask_roi);
    final_roi = final_roi(roi2);
    cv::resize(final_roi, final_roi, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::threshold(final_roi, thresh, 120, 255, cv::THRESH_BINARY);
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> filtered_contours;
    for (const auto& cnt : contours) {
        if (contourArea(cnt) > 100) {
            filtered_contours.push_back(cnt);
        }
    }

    std::sort(filtered_contours.begin(), filtered_contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b){
        return cv::contourArea(a) > cv::contourArea(b);
    });

    //cv::Mat mini;
    //cv::cvtColor(thresh, mini, cv::COLOR_GRAY2BGR);
    //cv::drawContours(mini, filtered_contours, -1, cv::Scalar(255, 0, 255), 2, cv::LINE_8);

    std::vector<cv::Scalar> colors = { cv::Scalar(0,255,0), cv::Scalar(255,0,0), cv::Scalar(0,0,255), cv::Scalar(255,0,255), cv::Scalar(255,255,0) };

    for (int i = 0; i < min(filtered_contours.size(), 3); i++) {
        cv::Rect bound = cv::boundingRect(filtered_contours[i]);
        float aspect_ratio = static_cast<float>(bound.width) / bound.height;

        if (aspect_ratio > 0.95f && aspect_ratio < 1.05f) {
            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(filtered_contours[i], center, radius);

            std::vector<float> angles;
            int max_empty = 0;
            float best_angle = 0;

            for (int ang = 0; ang < 360; ++ang) {
                float angle = ang * CV_PI / 180.0f;
                int empty_count = 0;

                for (int j = 0; j < rad_checks; ++j) {
                    float r = (radius * 1.1f) * j / rad_checks;
                    int x = static_cast<int>(center.x + r * cos(angle));
                    int y = static_cast<int>(center.y + r * sin(angle));

                    if (x >= 0 && x < final_roi.cols && y >= 0 && y < final_roi.rows) {
                        if (cv::pointPolygonTest(filtered_contours[i], cv::Point2f(x, y), false) == -1) {
                            empty_count++;
                        }
                    }
                }

                if (empty_count >= max_empty) {
                    max_empty = empty_count;
                    best_angle = angle;
                }

                if (empty_count == rad_checks) {
                    angles.push_back(angle);
                }
            }

            if (!angles.empty()) {
                float sum = std::accumulate(angles.begin(), angles.end(), 0.0f);
                best_angle = sum / angles.size();
            }

            int gap_x = static_cast<int>(center.x + radius * cos(best_angle));
            int gap_y = static_cast<int>(center.y + radius * sin(best_angle));

            cv::line(frame, cv::Point(static_cast<int>(center.x/scale) + roi1.x + roi2.x, static_cast<int>(center.y/scale) + roi1.y + roi2.y), cv::Point(static_cast<int>(gap_x/scale) + roi1.x + roi2.x, static_cast<int>(gap_y/scale) + roi1.y + roi2.y), colors[i], 8);
            //cv::line(mini, center, cv::Point(gap_x, gap_y), colors[i], 2);
        }
    }

    //cv::imshow("yikes", mini);
    return frame;
}

cv::Mat SubsectionWidget::Filters::detectCircles2(cv::Mat frame){
    double scale = 4;
    int rad_checks = 72;

    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

    cv::Mat temp;
    cv::resize(gray_frame, temp, cv::Size(), 1.0/scale, 1.0/scale, cv::INTER_AREA);

    std::vector<cv::Vec3f> ext_circles;
    cv::HoughCircles(temp, ext_circles, cv::HOUGH_GRADIENT, 1, temp.rows/8, 100, 50, temp.rows/8, temp.rows/4);

    double min_dis = std::numeric_limits<double>::infinity();
    cv::Vec3f ext_sector;

    if (!ext_circles.empty()) {
        for (std::size_t i = 0; i < ext_circles.size(); ++i) {
            ext_circles[i][0] *= scale;
            ext_circles[i][1] *= scale;
            ext_circles[i][2] *= scale;
            cv::Point center(std::round(ext_circles[i][0]), std::round(ext_circles[i][1]));
            int radius = std::round(ext_circles[i][2]);
            double dis = std::pow(frame.cols - center.x, 2) + std::pow(frame.rows - center.y, 2);

            if(dis < min_dis){
                min_dis = dis;
                ext_sector = ext_circles[i];
            }
        }
    }

    if (min_dis == std::numeric_limits<double>::infinity())
        return this->placeText("MISSING TASK SECTOR", frame);

    cv::Mat ext_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::circle(ext_mask, cv::Point(std::round(ext_sector[0]), std::round(ext_sector[1])), std::round(ext_sector[2]), 255, -1);

    cv::Mat frame_roi = cv::Mat::zeros(gray_frame.size(), gray_frame.type());
    cv::bitwise_and(gray_frame, gray_frame, frame_roi, ext_mask);

    cv::Rect roi_rect = cv::boundingRect(ext_mask);
    frame_roi = frame_roi(roi_rect);

    if (frame_roi.empty() || frame_roi.rows < 8 || frame_roi.cols < 8) {
        return this->placeText("MISSING TASK SECTOR", frame);
    }

    std::vector<cv::Vec3f> inn_circles;
    cv::HoughCircles(frame_roi, inn_circles, cv::HOUGH_GRADIENT, 1, frame_roi.rows / 8, 100, 50, frame_roi.rows / 8, frame_roi.rows / 3);

    cv::Mat roi_mask = cv::Mat::ones(frame_roi.size(), CV_8UC1) * 255;

    if (!inn_circles.empty()) {
        for (std::size_t i = 0; i < inn_circles.size(); ++i) {
            cv::Point center(std::round(inn_circles[i][0]), std::round(inn_circles[i][1]));
            int radius = std::round(inn_circles[i][2]) + 5;
            cv::circle(roi_mask, center, radius, 0, 8);
        }
    }
    else
        return this->placeText("MISSING INNER CIRCLE", frame);


    cv::Mat mask_roi = cv::Mat::zeros(frame_roi.size(), CV_8UC1);
    cv::circle(mask_roi, cv::Point(std::round(inn_circles[0][0]), std::round(inn_circles[0][1])), std::round(inn_circles[0][2]) - 10, 255, -1);

    cv::Mat final;
    cv::bitwise_and(frame_roi, frame_roi, final, mask_roi);

    cv::Rect final_rect = cv::boundingRect(mask_roi);
    final = final(final_rect);
    cv::resize(final, final, cv::Size(), scale, scale, cv::INTER_LINEAR);

    cv::Mat thresh;
    cv::threshold(final, thresh, 120, 255, cv::THRESH_BINARY);

    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> filtered_contours;
    for (const auto& cnt : contours) {
        if (cv::contourArea(cnt) > 100)
            filtered_contours.push_back(cnt);
    }

    std::sort(filtered_contours.begin(), filtered_contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
        return cv::contourArea(a) > cv::contourArea(b);
    });
    std::vector<cv::Scalar> colors = { cv::Scalar(0,255,0), cv::Scalar(255,0,0), cv::Scalar(0,0,255), cv::Scalar(255,255,255), cv::Scalar(255,255,255) };

    for (std::size_t i = 0; i < filtered_contours.size() && i < 3; ++i) {
        cv::Rect bbox = cv::boundingRect(filtered_contours[i]);
        float aspect_ratio = static_cast<float>(bbox.width) / bbox.height;

        if (0.95f < aspect_ratio && aspect_ratio < 1.05f) {
            cv::Point2f center;
            float r;
            cv::minEnclosingCircle(filtered_contours[i], center, r);

            int min_touch = INT_MAX;
            double best_angle = 0.0;
            std::vector<double> angles;

            for (int j = 0; j < rad_checks; ++j) {
                double angle = 2.0 * CV_PI * j / rad_checks;

                cv::Mat mask = cv::Mat::zeros(final.size(), CV_8UC1);
                cv::drawContours(mask, std::vector<std::vector<cv::Point>>{filtered_contours[i]}, -1, 255, -1);

                cv::Mat line_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
                int x3 = static_cast<int>(center.x + r * std::cos(angle));
                int y3 = static_cast<int>(center.y + r * std::sin(angle));
                cv::line(line_mask, cv::Point(static_cast<int>(center.x), static_cast<int>(center.y)), cv::Point(x3, y3), 255, 1);

                cv::Mat overlap;
                cv::bitwise_and(mask, line_mask, overlap);
                int temp = cv::countNonZero(overlap);

                if (temp < min_touch) {
                    min_touch = temp;
                    best_angle = angle;
                }
                if (temp == 0) {
                    angles.push_back(angle);
                }
            }

            if (!angles.empty()) {
                double sum = std::accumulate(angles.begin(), angles.end(), 0.0);
                best_angle = sum / angles.size();
            }

            int gap_x = static_cast<int>(center.x + r * std::cos(best_angle));
            int gap_y = static_cast<int>(center.y + r * std::sin(best_angle));

            cv::line(frame, cv::Point(static_cast<int>((center.x/scale)+roi_rect.x+final_rect.x), static_cast<int>(center.y/scale + roi_rect.y + final_rect.y)), cv::Point(static_cast<int>((gap_x/scale)+roi_rect.x+final_rect.x), static_cast<int>(gap_y/scale + roi_rect.y + final_rect.y)), colors[i], 8);
        }
    }

    return frame;
}

cv::Mat SubsectionWidget::Filters::detectShape(cv::Mat frame){
    float scale = 4.0;

    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    cv::Mat temp_resized;
    cv::resize(gray_frame, temp_resized, cv::Size(), 1.0/scale, 1.0/scale, cv::INTER_AREA);
    std::vector<cv::Vec3f> ext_circles_vec;
    cv::HoughCircles(temp_resized, ext_circles_vec, cv::HOUGH_GRADIENT, 1, temp_resized.rows/8.0, 100, 50, temp_resized.rows/8, temp_resized.rows/4);

    double min_dis = DBL_MAX;
    cv::Vec3f ext_sector;

    if (!ext_circles_vec.empty()) {
        for (const auto& circle : ext_circles_vec) {
            float x = circle[0] * scale;
            float y = circle[1] * scale;
            float r = circle[2] * scale;
            cv::Point center = cv::Point(cvRound(x), cvRound(y));
            double dis = (center.x*center.x) + (frame.rows-center.y)*(frame.rows-center.y);
            if(dis < min_dis){
                min_dis = dis;
                ext_sector = cv::Vec3f(x, y, r);
            }
        }
    }

    if(min_dis == DBL_MAX)
        return this->placeText("MISSING TASK SECTOR", frame);;

    cv::Mat ext_mask = cv::Mat::zeros(frame.size(), CV_8UC1), masked_gray;
    cv::circle(ext_mask, cv::Point(cvRound(ext_sector[0]), cvRound(ext_sector[1])), cvRound(ext_sector[2]), cv::Scalar(255), -1);
    cv::bitwise_and(gray_frame, gray_frame, masked_gray, ext_mask);
    cv::Rect ext_box = cv::boundingRect(ext_mask);
    cv::Mat frame_roi = masked_gray(ext_box);
    if(frame_roi.empty() || frame_roi.rows < 8 || frame_roi.cols < 8)
        return this->placeText("MISSING TASK SECTOR", frame);

    std::vector<cv::Vec3f> inner_circles_vec;
    cv::HoughCircles(frame_roi, inner_circles_vec, cv::HOUGH_GRADIENT, 1, frame_roi.rows/8.0, 100, 50, frame_roi.rows/8, frame_roi.rows/3);
    if(inner_circles_vec.empty())
        return this->placeText("MISSING INNER ROI", frame);
    cv::Vec3f inner_circle = inner_circles_vec[0];

    cv::Mat mask_roi = cv::Mat::zeros(frame_roi.size(), CV_8UC1), final_roi, final_thresh;
    cv::circle(mask_roi, cv::Point(cvRound(inner_circle[0]), cvRound(inner_circle[1])), cvRound(inner_circle[2]) - 10, cv::Scalar(255), -1);
    cv::bitwise_and(frame_roi, frame_roi, final_roi, mask_roi);
    cv::threshold(final_roi, final_thresh, 200, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(final_thresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> filtered_contours;
    for(const auto& contour : contours){
        double area = cv::contourArea(contour);
        if(area <= 10.0) continue;

        cv::Rect bound_rect = cv::boundingRect(contour);
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        if(bound_rect.height == 0 || hull.empty()) continue;

        double aspect_ratio = static_cast<double>(bound_rect.width) / bound_rect.height;
        double solidity = area / cv::contourArea(hull);

        if(aspect_ratio > 0.5 && aspect_ratio < 1.5 && solidity > 0.5)
            filtered_contours.push_back(contour);
    }

    if(!filtered_contours.empty()){
        cv::Point roi_center(final_roi.cols/2, final_roi.rows/2);
        double min_distance = DBL_MAX;
        std::vector<cv::Point> best_contour;

        for(int i = 0; i < filtered_contours.size(); i++){
            std::vector<cv::Point> contour = filtered_contours[i];
            cv::Moments M = cv::moments(contour);

            if(M.m00 != 0){
                cv::Point center_of_mass(static_cast<int>(M.m10 / M.m00), static_cast<int>(M.m01 / M.m00));
                double distance = cv::norm(center_of_mass - roi_center);

                if(distance < min_distance){
                    min_distance = distance;
                    best_contour = contour;
                }
            }
        }

        if(!best_contour.empty()){
            cv::Rect best_contour_box = cv::boundingRect(best_contour), final_box;

            final_box.x = best_contour_box.x + ext_box.x - 10;
            final_box.y = best_contour_box.y + ext_box.y - 10;
            final_box.width = best_contour_box.width + 20;
            final_box.height = best_contour_box.height + 20;

            cv::rectangle(frame, final_box, cv::Scalar(0, 255, 0), 5);
        }
    }
    else
        return this->placeText("MISSING SHAPE CONTOUR", frame);

    return frame;
}

SubsectionWidget::~SubsectionWidget(){
    camera_dropdown->setCurrentIndex(0);
    filters.none.store(true);
    is_cv_running.store(false);
    filter_channel->is_active.store(false);
    filter_channel->is_recv_running.store(false);
    filter_channel->is_send_running.store(false);
    emit destructorCalled(id);
    cv_thread.join();
    if(this->id == 0){
        for(int i = 0; i < 10; i++){
            filter_channel->target_socket->sendPacket(std::vector<uchar>{0x00}, 0);
        }
    }
    filter_channel->target_socket->destroy();
    filter_channel->recv_thread.join();
    filter_channel->send_thread.join();
}

void SubsectionWidget::updateAvailableOptions(const QSet<QString> &usedOptions) {
    return;
    auto * model = qobject_cast<QStandardItemModel*>(camera_dropdown->model());
    if(!model) return;
    for(int i = 0; i < camera_dropdown->count(); i++){
        QString option = camera_dropdown->itemText(i);
        auto * item = model->item(i);
        if(!item) continue;
        item->setEnabled(!usedOptions.contains(option));
    }
}

void SubsectionWidget::setAvailableDevices(int num_cams) {
    for(int i = 1; i <= num_cams; i++){
        camera_dropdown->addItem(QString("Camera %1").arg(i), i);
    }
}

/*
void SubsectionWidget::onCameraSelected(int index) {
    cam_id = index - 1;
    if(index <= 0)
        camera_view->setPixmap(QPixmap("../../assets/404.png").scaled((fullScreen ? QSize(960, 720) : QSize(480, 360)), Qt::KeepAspectRatio));
}
*/

void SubsectionWidget::updateFrame(cv::Mat frame, std::vector<uchar> compressed){
    {
        std::lock_guard<std::mutex> lock(compressed_mutex);
        latest_compressed = compressed;
    }
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        latest_frame = frame;
    }
    if(!filters.none.load()){
        std::lock_guard<std::mutex> lock(filter_mutex);
        if(!filter_frame.empty())
            frame = filter_frame;
    }
    if(!frame.empty() && frame.data != nullptr && frame.cols > 0 && frame.rows > 0) {
        if(frame.type() != CV_8UC3)
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        QImage image(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
        qt_frame = image.copy();  // ????????????????
        emit frameReady(qt_frame);
    }
    else
        qWarning() << "SUBSECTION " << this->id << " UPDATE | Invalid frame: empty or corrupt";
}

void SubsectionWidget::mousePressEvent(QMouseEvent *event) {
    emit subsectionClicked(this);
    QWidget::mousePressEvent(event);
}

// --- main window ---
MainWindow::MainWindow(QWidget *parent) : QWidget(parent){
    main_layout = new QHBoxLayout(this);
    main_layout->setSpacing(0);
    main_layout->setContentsMargins(0, 0, 0, 0);
    left_container = new QWidget(this);
    left_layout = new QGridLayout(left_container);
    right_layout = new QVBoxLayout;
    right_layout->setSpacing(0);
    right_layout->setContentsMargins(0, 0, 0, 0);
    left_layout->setSpacing(0);
    left_layout->setContentsMargins(0, 0, 0, 0);
    this->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    this->setFixedSize(1280, 720);
    is_fullscreen = false;
    int id = 0;
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++){
            cam_map.insert({subsections.size(), -1});
            SubsectionWidget* widget = new SubsectionWidget(id, this);
            id++;
            connect(widget, &SubsectionWidget::subsectionClicked, this, [this](SubsectionWidget* clicked_widget){
                if(fullscreen_widget){
                    clicked_widget->hide();
                    for(int k = 0; k < subsections.size(); k++){
                        subsections[k]->setMinimumSize(QSize(480, 360));
                        subsections[k]->setMaximumSize(QSize(480, 360));
                        left_layout->addWidget(subsections[k], k/2, k%2);
                        subsections[k]->setFullScreenMode(false);
                        subsections[k]->show();
                    }
                    fullscreen_widget = nullptr;
                }
                else{
                    fullscreen_widget = clicked_widget;
                    for(SubsectionWidget* sw : subsections){
                        sw->hide();
                    }
                    clicked_widget->setMaximumSize(QSize(960, 720));
                    clicked_widget->setMinimumSize(QSize(960, 720));
                    left_layout->addWidget(clicked_widget, 0, 0, 2, 2);
                    clicked_widget->setFullScreenMode(true);
                    clicked_widget->show();
                }
                /*
                if(!is_fullscreen){
                    for(int k = 0; k < subsections.size(); k++){
                        if(subsections[k] == clicked_widget){
                            clicked_widget->setFullScreenMode(true);
                            left_layout->addWidget(clicked_widget, 0, 0, 2, 2);
                        }
                        else
                            subsections[k]->hide();
                    }
                    is_fullscreen = true;
                }
                else{
                    for(int k = 0; k < subsections.size(); k++){
                        subsections[k]->setFullScreenMode(false);
                        left_layout->addWidget(subsections[k], k/2, k%2);
                        subsections[k]->show();
                    }
                    is_fullscreen = false;
                }
                */
            });
            connect(widget, &SubsectionWidget::selectionChanged, this, [this](){
                QSet<QString> used_options;
                for(int k = 0; k < subsections.size(); k++){
                    std::pair<int, QString> selection = subsections[k]->getCurrentSelection();
                    QString selection_text = selection.second;
                    if(selection.first >= 0){
                        used_options.insert(selection.second);
                        cam_map[k] = selection.first;
                    }
                    else
                        cam_map[k] = -1;
                }
                for(int k = 0; k < subsections.size(); k++){
                    subsections[k]->updateAvailableOptions(used_options);
                }
                emit selectionChanged(cam_map);
            });
            connect(widget, &SubsectionWidget::destructorCalled, this, [this](int id){ emit destructorCalled(id); });
            subsections.push_back(widget);
            left_layout->addWidget(widget, i, j);
        }
    }
    gas_label = new QLabel("No data");
    gas_label->setObjectName("sensor");
    speech_label = new QLabel("No data");
    speech_label->setObjectName("sensor");
    magnetometer_label = new QLabel("No data");
    magnetometer_label->setObjectName("sensor");
    magnetometer_label->setAlignment({Qt::AlignHCenter, Qt::AlignVCenter});
    microphone_button = new QPushButton("Toggle audio");
    microphone_button->setCheckable(true);
    microphone_button->setObjectName("mic");
    clear_button = new QPushButton("Clear data");
    clear_button->setObjectName("clear");
    local_button = new QPushButton("Local filters");
    local_button->setCheckable(true);
    local_button->setObjectName("mic");
    dashboard_layout = new QGridLayout();
    button_layout = new QHBoxLayout();
    std::vector<QString> labels = {"Gas sensor: ", "Speech: ", "Magnetometer: "};
    for(int i = 0; i < labels.size(); i++){
        sensor_label = new QLabel(labels[i]);
        sensor_label->setObjectName("sensor");
        dashboard_layout->addWidget(sensor_label, i, 0);
    }
    dashboard_layout->addWidget(gas_label, 0, 1);
    dashboard_layout->addWidget(speech_label, 1, 1);
    dashboard_layout->addWidget(magnetometer_label, 2, 1);

    button_layout->addWidget(microphone_button);
    button_layout->addWidget(clear_button);
    button_layout->addWidget(local_button);

    //dashboard_layout->addWidget(microphone_button, 4, 0);
    //dashboard_layout->addWidget(clear_button, 4, 1);
    setStyleSheet(R"(
        QLabel#sensor {
            color: white;
            font-size: 14px;
            padding: 15px;
            border: 1.5px solid gray;
            font-family: Consolas;
        }
        QPushButton {
            font-size: 14px;
            color: white;
            border: none;
            padding: 5px;
            border-radius: 5px;
            margin: 5px;
        }
        QPushButton#mic {
            background-color: red;
        }
        QPushButton#clear {
            background-color: black;
        }
        QPushButton#mic:checked {
            background-color: green;
            color: white;
        }
    )");
    connect(microphone_button, &QPushButton::clicked, this, [this](){ emit buttonChanged(microphone_button->isChecked()); });
    connect(clear_button, &QPushButton::clicked, this, [this](){
        gas_label->setText("No data");
        speech_label->setText("No data");
        magnetometer_label->setText("No data");
    });
    connect(local_button, &QPushButton::clicked, this, [this](){
        for(int i = 0; i < subsections.size(); i++){
            subsections[i]->setLocal(local_button->isChecked());
        }
    });

    // 3D MODEL VIEWER
    model = new ModelWidget(this);
    right_layout->addWidget(model);
    right_layout->addLayout(dashboard_layout);

    right_layout->addLayout(button_layout);

    main_layout->addWidget(left_container, 3); // 3/4 width
    main_layout->addLayout(right_layout, 1);  // 1/4 width
}

void MainWindow::setCamPorts(int num_cams){
    for(int i = 0; i < subsections.size(); i++){
        subsections[i]->setAvailableDevices(num_cams);
    }
}

void MainWindow::updateFrame(int id, std::vector<unsigned char> data){
    cv::Mat frame = cv::imdecode(data, cv::IMREAD_COLOR);
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    int sub_id = -1;
    std::vector<int> sub_ids = {};
    for(auto it = cam_map.begin(); it != cam_map.end(); it++){
        if(it->second == id){
            sub_id = it->first;
            sub_ids.push_back(it->first);
        }
    }
    if(sub_id != -1){
        //subsections[sub_id]->updateFrame(frame, data);
        for(int i = 0; i < sub_ids.size(); i++){
            subsections[sub_ids[i]]->updateFrame(frame.clone(), data);
        }
    }
}

template<typename T> void MainWindow::updateDashbord(int index, T data){
    if(index == 0){
        if constexpr(std::is_same_v<T, int>)
            gas_label->setText(QString("%1 ppm").arg(data));
    }
    else if(index == 1){
        if constexpr(std::is_same_v<T, QString>)
            speech_label->setText(QString("%1 %2").arg(speech_label->text()).arg(data));
    }
    else if(index == 2){
        if constexpr(std::is_same_v<T, QVector3D>)
            magnetometer_label->setText(QString("X: %1 Y: %2\nZ: %3").arg(data.x(), 0, 'f', 2).arg(data.y(), 0, 'f', 2).arg(data.z(), 0, 'f', 2));
    }
    else
        qWarning() << "WINDOW UPDATE | Invalid dashboard index: out of bounds";
}

// SIGNAL INTERCEPT
void MainWindow::closeEvent(QCloseEvent* event) {
    emit windowClosing();
    qInfo() << "Closing main window...";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for(int i = 0; i < subsections.size(); i++){
        subsections[i]->destroy();
    }
    qInfo() << "Filter channels closed";
    model->destroy();
    event->accept();
    WSACleanup();
    qInfo() << "Bye";
}

void MainWindow::updateState(std::vector<float> data){
    BasePacket model_state;
    std::memcpy(&model_state, data.data(), sizeof(BasePacket));
    this->updateDashbord(0, (int)model_state.gas_ppm);
    this->updateDashbord(2, QVector3D(model_state.magnetometer_x, model_state.magnetometer_y, model_state.magnetometer_z));
    if(model == nullptr){
        qWarning() << "MAINWINDOW MODEL UPDATE | Invalid model: uninitialized pointer";
        return;
    }

    model->updateModel(model_state.body_x, model_state.body_y, model_state.body_z);
    model->updatePivot(1, 2, model_state.arm_l);
    model->updatePivot(2, 2, 180.0-model_state.arm_r);
    model->updatePivot(5, 1, model_state.art_1);
    model->updatePivot(6, 2, model_state.art_2);
    model->updatePivot(7, 2, model_state.art_3);
    // not actually hand
    //model->updatePivot(9, 2, model_state.art_4);

    QColor color;
    if(model_state.track_l < 0.0)
        color = QColor(nMap(model_state.track_l, -1.0, 0.0, 20, 250), 0, 0);
    else if(model_state.track_l > 0.0)
        color = QColor(0, nMap(model_state.track_l, 0.0, 1.0, 20, 250), 0);
    else
        color = Qt::black;
    model->updateColor(0, color);

    if(model_state.track_r < 0)
        color = QColor(nMap(model_state.track_r, -1.0, 0.0, 20, 250), 0, 0);
    else if(model_state.track_r > 0)
        color = QColor(0, nMap(model_state.track_r, 0.0, 1.0, 20, 250), 0);
    else
        color = Qt::black;
    model->updateColor(1, color);
}

// --- ROTAS stream handler ---
RTPStreamHandler::RTPStreamHandler(int port, std::string address, PayloadType type, QObject *parent) : QObject(parent){
    stream = new Stream;
    stream->ssrc = 0;
    stream->seq_num = 0 & 0xFFFF;
    stream->timestamp = 0;
    stream->payload_type = type;
    stream->port = port;

    // --- UDP Socket init ---
    // -- send --
    send_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    send_socket_address.sin_family = AF_INET;
    send_socket_address.sin_port = htons(port + 1);
    inet_pton(AF_INET, address.c_str(), &send_socket_address.sin_addr);
    // -- recv --
    recv_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    int recv_buff_size = 1024 * 1024;   // 1MB
    setsockopt(recv_socket, SOL_SOCKET, SO_RCVBUF, (char*)&recv_buff_size, sizeof(recv_buff_size));
    recv_socket_address.sin_family = AF_INET;
    recv_socket_address.sin_port = htons(port);
    recv_socket_address.sin_addr.s_addr = INADDR_ANY;
    bind(recv_socket, (struct sockaddr*)&recv_socket_address, socket_address_size);
    qInfo() << "Channel created bound to ports (" << port << ", " << port + 1 << ")";
}

RTPStreamHandler::~RTPStreamHandler(){
    shutdown(recv_socket, SD_BOTH);
    closesocket(send_socket);
    closesocket(recv_socket);
    qInfo() << "Closing channel (" << stream->port << ", " << stream->port + 1 << ")";
}

template <typename T> void RTPStreamHandler::sendPacket(std::vector<T> data, int marker){
    // --- Initial settings ---
    int max_size = MAX_UDP_PACKET_SIZE - sizeof(RTPHeader);
    int num_fragments = ((data.size()*sizeof(T)) + max_size - 1) / max_size;
    // -- (Pseudo)random ssrc --
    thread_local uint16_t ssrc = 1;
    ssrc ^= ssrc << 7;
    ssrc ^= ssrc >> 9;
    ssrc ^= ssrc << 8;

    // --- Fragment setup ---
    for(int i = 0; i < num_fragments; i++){
        // -- RTP header info --
        RTPHeader header;
        header.version = 2;
        header.p = 0;
        header.x = 0;
        header.cc = 0;
        header.m = (uint16_t)num_fragments;
        header.pt = 0;
        header.timestamp = (uint16_t)marker;
        header.ssrc = ssrc;
        header.seq = (uint16_t)i;
        if(num_fragments > 1)
            header.seq |= FRAGMENTATION_FLAG;
        // -- Merge header + packet --
        int current_size = (max_size < ((data.size()*sizeof(T)) - (i*max_size)) ? max_size : (data.size()*sizeof(T)) - (i*max_size));
        std::vector<char> packet(current_size + sizeof(RTPHeader));
        std::memcpy(packet.data(), &header, sizeof(RTPHeader));
        std::memcpy(packet.data() + sizeof(RTPHeader), data.data() + (i*max_size), current_size);

        if(sendto(send_socket, (const char*)packet.data(), packet.size(), 0, (struct sockaddr*)&send_socket_address, socket_address_size) == SOCKET_ERROR){
            qWarning() << "ROTAS SEND | Winsock error: " << WSAGetLastError();
        }
    }
}

void RTPStreamHandler::recvPacket(){
    std::vector<std::vector<char>> fragments;
    std::vector<char> packet, buffer(MAX_PACKET_SIZE);
    int i = 0, num_fragments = -1, ssrc = -1;
    do{
        int bytes_received = recvfrom(recv_socket, buffer.data(), MAX_PACKET_SIZE, 0, (struct sockaddr*)&recv_socket_address, &socket_address_size);
        if(bytes_received == SOCKET_ERROR){
            int error = WSAGetLastError();
            if(error != 10004)
                qCritical() << "ROTAS RECV | Winsock error: " << error;
            return;
        }
        else if (bytes_received < sizeof(RTPHeader)) {
            qCritical() << "ROTAS RECV | Invalid RTP header: incomplete packet received";
            return;
        }

        RTPHeader* header = new RTPHeader;
        std::memcpy(header, buffer.data(), sizeof(RTPHeader));
        packet.resize(bytes_received - sizeof(RTPHeader));
        std::memcpy(packet.data(), buffer.data() + sizeof(RTPHeader), bytes_received - sizeof(RTPHeader));

        if((header->seq & FRAGMENTATION_FLAG) == 0) break;
        else if(i == 0){
            num_fragments = header->m;
            ssrc = header->ssrc;
            fragments.resize(num_fragments);
        }
        else if(ssrc != header->ssrc){
            qWarning() << "ROTAS RECV | Fragmentation error: previous packet dropped";
            i = 0;
            fragments.clear();
            num_fragments = header->m;
            ssrc = header->ssrc;
            fragments.resize(num_fragments);
        }

        fragments[header->seq & ~FRAGMENTATION_FLAG] = packet;
        if(i == num_fragments - 1){
            packet.clear();
            for(int i = 0; i < num_fragments; i++){
                packet.insert(packet.end(), fragments[i].begin(), fragments[i].end());
            }
        }

        i++;
    } while(i < num_fragments);

    if(stream->payload_type == PayloadType::ROS2_ARRAY && floatCallback){
        std::vector<float> data(packet.size() / sizeof(float));
        std::memcpy(data.data(), packet.data(), packet.size());
        floatCallback(data);
    }
    else if((stream->payload_type == PayloadType::VIDEO_MJPEG || stream->payload_type == PayloadType::AUDIO_PCM) && ucharCallback){
        std::vector<uchar> data(packet.size());
        std::memcpy(data.data(), packet.data(), packet.size());
        ucharCallback(data);
    }
    else
        qCritical() << "ROTAS RECV | Invalid packet data: mismatched payload/callback";
}

AppHandler::AppHandler(int port, QObject* parent) : QObject(parent){
    qInfo() << "Starting GUI...";
    window = new MainWindow;
    window->setWindowTitle("GUI - beta");
    window->resize(1280, 720);
    QObject::connect(window, &MainWindow::windowClosing, [this](){ this->destroy(); });
    this->port = port;

    qInfo() << "Starting base channel...";
    base_channel = new SocketStruct;
    base_channel->target_socket = new RTPStreamHandler(port, CLIENT_IP, PayloadType::ROS2_ARRAY);
    base_channel->target_socket->setFloatCallback([this](std::vector<float> data){
        if(data.size() < sizeof(BasePacket)/sizeof(float)){
            qCritical() << "APPHANDLER BASE CB | Invalid payload: incomplete data";
            return;
        }
        {
            std::lock_guard<std::mutex> lock(base_channel->data_mutex);
            base_channel->float_data = data;
        }
        window->updateState(std::vector<float>(data.begin()+1, data.end()));
    });
    base_channel->is_recv_running.store(true);
    base_channel->is_send_running.store(true);

    qInfo() << "Starting audio channel...";

    vosk_channel = new SocketStruct;
    vosk_channel->target_socket = new RTPStreamHandler(9008, "127.0.0.1", PayloadType::AUDIO_PCM);
    vosk_channel->target_socket->setUCharCallback([this](std::vector<uchar> data){
        std::string str(data.begin(), data.end());
        window->updateDashbord(1, str);
        qDebug() << "vosk recv: " << str;
    });
    vosk_channel->is_recv_running.store(true);
    vosk_channel->is_send_running.store(true);
    audio_channel = new SocketStruct;
    audio_channel->target_socket = new RTPStreamHandler(port + 2, CLIENT_IP, PayloadType::AUDIO_PCM);
    audio_channel->target_socket->setUCharCallback([this](std::vector<uchar> data){
        if(vosk_channel->is_send_running.load())
            vosk_channel->target_socket->sendPacket(data, 1);
        std::vector<opus_int16> output(AUDIO_BUFFER_SIZE);
        int frames = opus_decode(opus_decoder, data.data(), data.size(), output.data(), output.size(), 0);
        Pa_WriteStream(stream, output.data(), frames);
    });
    audio_channel->is_recv_running.store(true);
    audio_channel->is_send_running.store(true);

    opus_decoder = opus_decoder_create(SAMPLE_RATE, 1, &pa_error);
    Pa_Initialize();
    Pa_OpenDefaultStream(&stream, 0, 1, paInt16, SAMPLE_RATE, AUDIO_BUFFER_SIZE, nullptr, nullptr);
    is_audio_active.store(false);
    connect(window, &MainWindow::buttonChanged, this, [this](bool is_pressed){ is_audio_active.store(is_pressed); });
    qRegisterMetaType<std::map<int, int>>("std::map<int,int>");
    connect(window, &MainWindow::selectionChanged, this, [this](std::map<int,int> cam_map){
        for(int i = 0; i < video_channels.size(); i++){
            video_channels[i]->is_active.store(false);
        }
        for(auto it = cam_map.begin(); it != cam_map.end(); it++){
            if(it->second >= 0)
                video_channels[it->second]->is_active.store(true);
        }
    });
    qInfo() << "Setup complete";
}

AppHandler::~AppHandler(){
    qInfo() << "Closing program...";
    base_channel->target_socket->sendPacket(std::vector<int>{0, -1});
    base_channel->is_recv_running.store(false);
    base_channel->is_send_running.store(false);
    base_channel->target_socket->destroy();
    if(base_channel->send_thread.joinable())
        base_channel->send_thread.join();
    if(base_channel->recv_thread.joinable())
        base_channel->recv_thread.join();
    qInfo() << "Base channel closed";
    vosk_channel->is_recv_running.store(false);
    vosk_channel->is_send_running.store(false);
    vosk_channel->target_socket->destroy();
    if(vosk_channel->recv_thread.joinable())
        vosk_channel->recv_thread.join();
    audio_channel->is_recv_running.store(false);
    audio_channel->is_send_running.store(false);
    audio_channel->target_socket->destroy();
    if(audio_channel->recv_thread.joinable())
        audio_channel->recv_thread.join();
    if(audio_channel->send_thread.joinable())
        audio_channel->send_thread.join();
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    opus_decoder_destroy(opus_decoder);
    qInfo() << "Audio channels closed";

    for(int i = 0; i < video_channels.size(); i++){
        video_channels[i]->is_recv_running.store(false);
        video_channels[i]->is_send_running.store(false);
        video_channels[i]->target_socket->destroy();
        if(video_channels[i]->recv_thread.joinable())
            video_channels[i]->recv_thread.join();
        if(video_channels[i]->send_thread.joinable())
            video_channels[i]->send_thread.join();
    }
    qInfo() << "Video channels closed";
}

void AppHandler::init(){
    qInfo() << "Starting ROTAS stream...";

    int num_cams = 0;
    base_channel->target_socket->sendPacket(std::vector<int>{0, 0});
    qInfo() << "Awaiting response...";
    base_channel->target_socket->recvPacket();
    {
        std::lock_guard<std::mutex> lock(base_channel->data_mutex);
        if(base_channel->float_data.empty() || base_channel->float_data[0] < 0){
            qCritical() << "APPHANDLER INIT | Base handshake failed";
            return;
        }
        num_cams = (int)base_channel->float_data[0];
    }
    qInfo() << "Connection established. Received " << num_cams << " video sources";
    window->setCamPorts(num_cams);

    base_channel->recv_thread = std::thread([this](){
        while(base_channel->is_recv_running.load()){
            base_channel->target_socket->recvPacket();
        }
    });

    for(int i = 0; i < num_cams; i++){
        SocketStruct* video_socket = new SocketStruct;
        video_socket->target_socket = new RTPStreamHandler(port + (2 * i) + 4, CLIENT_IP, PayloadType::VIDEO_MJPEG);
        video_channels.push_back(std::move(video_socket));
    }
    for(int i = 0; i < video_channels.size(); i++){
        video_channels[i]->is_active.store(false);
        video_channels[i]->is_send_running.store(true);
        video_channels[i]->is_recv_running.store(true);
        video_channels[i]->target_socket->setUCharCallback([this, i](std::vector<uchar> data) { window->updateFrame(i, data); });
        video_channels[i]->recv_thread = std::thread([i, this](){
            while(video_channels[i]->is_recv_running.load()){
                video_channels[i]->target_socket->recvPacket();
            }
        });
        video_channels[i]->send_thread = std::thread([i, this](){
            while(video_channels[i]->is_send_running.load()){
                video_channels[i]->target_socket->sendPacket(std::vector<int>{0, (int)video_channels[i]->is_active.load()});
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
    }
    Pa_StartStream(stream);
    audio_channel->recv_thread = std::thread([this](){
        while(audio_channel->is_recv_running.load()){
            audio_channel->target_socket->recvPacket();
        }
    });
    audio_channel->send_thread = std::thread([this](){
        while(audio_channel->is_send_running.load()){
            audio_channel->target_socket->sendPacket(std::vector<int>{0, (int)is_audio_active.load()});
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    vosk_channel->recv_thread = std::thread([this](){
        while(vosk_channel->is_recv_running.load()){
            vosk_channel->target_socket->recvPacket();
        }
    });
    qDebug() << "vosk recv done";

    window->show();
    qInfo() << "Program init complete";
}

ConsoleWindow::ConsoleWindow(QWidget *parent) : QMainWindow(parent){
    text_edit = new QTextEdit(this);
    setWindowTitle("Debug console");
    setCentralWidget(text_edit);
    text_edit->setReadOnly(true);
    text_edit->setStyleSheet("background-color: black; font-size: 16; font-family: Consolas;");
}

void ConsoleWindow::appendMessage(const QString &message) {
    text_edit->append(message);
}
