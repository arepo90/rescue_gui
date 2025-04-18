#include "mainwindow.h"

// --- Helper funcs ---
int nMap(int n, int minIn, int maxIn, int minOut, int maxOut){
    return float((n - minIn)) / float((maxIn - minIn)) * (maxOut - minOut) + minOut;
}
int nMap(float n, float minIn, float maxIn, float minOut, float maxOut){
    return float((n - minIn)) / float((maxIn - minIn)) * (maxOut - minOut) + minOut;
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
        QQuaternion::fromEulerAngles(0, 0, 45),
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
        qCritical() << "Failed to update pivot. Out of bounds";
        return;
    }
    if(axis == 0)
        pivots[index]->setRotation(QQuaternion::fromEulerAngles(angle, pivots[index]->rotationY(), pivots[index]->rotationZ()));
    else if(axis == 1)
        pivots[index]->setRotation(QQuaternion::fromEulerAngles(pivots[index]->rotationX(), angle, pivots[index]->rotationZ()));
    else if(axis == 2)
        pivots[index]->setRotation(QQuaternion::fromEulerAngles(pivots[index]->rotationX(), pivots[index]->rotationY(), angle));
    else
        qCritical() << "Failed to update pivot. Unvalid axis value";
}

void ModelWidget::updateColor(int index, QColor color){
    if(index >= band_colors.size()){
        qCritical() << "Failed to update color. Out of bounds";
        return;
    }
    band_colors[index]->setDiffuse(color);
}

// --- cam subsections ---
SubsectionWidget::SubsectionWidget(int id, QWidget *parent) : QWidget(parent){
    this->id = id;
    container = new QWidget(this);
    layout = new QVBoxLayout();
    dropdowns = new QHBoxLayout();
    cameraView = new QLabel();
    camera_dropdown = new QComboBox();
    camera_dropdown->addItem("No Camera");
    filter_dropdown = new QComboBox();
    filter_dropdown->addItems({ "No filter", "QR Code", "Hazmat", "Shape - Hough", "Shape - Contours", "Shape - Hybrid", "Circles" });
    cam_id = -1;
    dropdowns->addWidget(camera_dropdown);
    dropdowns->addWidget(filter_dropdown);
    layout->addLayout(dropdowns);
    layout->addWidget(cameraView);
    layout->setSpacing(0);
    layout->setContentsMargins(0, 0, 0, 0);
    container->setLayout(layout);
    container->setStyleSheet("border: 0.5px solid gray;");
    filters.none.store(true);
    connect(camera_dropdown,  &QComboBox::currentIndexChanged, this, [this](int index){
        cam_id = index - 1;
        emit selectionChanged();
        if(index == 0){
            this->updateFrame(cv::imread("../../assets/404.png"));
        }
    });
    connect(filter_dropdown, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index){
        if(camera_dropdown->currentIndex() == 0){
            filter_dropdown->blockSignals(true);
            filter_dropdown->setCurrentIndex(0);
            filter_dropdown->blockSignals(false);
        }
        filters.none.store(index == 0);
        filters.is_qr_active.store(index == 1);
        filters.is_hazmat_active.store(index == 2);
        filters.is_shape1_active.store(index == 3);
        filters.is_shape2_active.store(index == 4);
        filters.is_shape3_active.store(index == 5);
        filters.is_circles_active.store(index == 6);
    });
    cameraView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    cameraView->setPixmap(QPixmap("../../assets/404.png").scaled((fullScreen ? QSize(960, 720) : QSize(480, 360)), Qt::KeepAspectRatio));
    is_active.store(false);
    is_cv_running.store(true);
    cv_thread = std::thread([this](){
        while(is_cv_running.load()){
            if(filters.none.load()){
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            auto start = std::chrono::high_resolution_clock::now();
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
                /*
                if(!points.empty()){
                    for(int i = 0; i < points.size(); i++) {
                        cv::line(frame, points[i], points[(i+1) % points.size()], cv::Scalar(0, 255, 0), 2);
                    }
                }
                */
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
                    result = this->detectShapeHough(frame);
                else if(filters.is_shape2_active.load())
                    result = this->detectShapeContours(frame);
                else if(filters.is_shape3_active.load())
                    result = this->detectShapeHybrid(frame);
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
            auto end = std::chrono::high_resolution_clock::now();
            qDebug() << "filter: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms";
        }
    });
    connect(this, &SubsectionWidget::frameReady, this, [this](QImage image){
        if(!image.isNull())
            cameraView->setPixmap(QPixmap::fromImage(qt_frame).scaled((fullScreen ? QSize(960, 720) : QSize(480, 360)), Qt::KeepAspectRatio));
        else
            qDebug() << "Image is null after conversion!";
    });
}

SubsectionWidget::~SubsectionWidget(){
    emit destructorCalled(id);
    filters.none.store(true);
    is_cv_running.store(false);
    if(cv_thread.joinable()) cv_thread.join();
}

void SubsectionWidget::updateAvailableOptions(const QSet<QString> &usedOptions) {
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
        cameraView->setPixmap(QPixmap("../../assets/404.png").scaled((fullScreen ? QSize(960, 720) : QSize(480, 360)), Qt::KeepAspectRatio));
}
*/

void SubsectionWidget::updateFrame(cv::Mat frame){
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        latest_frame = frame;
    }
    if(!filters.none.load()){
        std::vector<cv::Point> points;
        {
            std::lock_guard<std::mutex> lock(filter_mutex);
            //points = filter_points;

            if(!filter_frame.empty())
                frame = filter_frame;
        }

        /*
        if(!points.empty()){
            for(int i = 0; i < points.size(); i++) {
                cv::line(frame, points[i], points[(i+1) % points.size()], cv::Scalar(0, 255, 0), 2);
            }
        }
        */
    }
    if(!frame.empty() && frame.data != nullptr && frame.cols > 0 && frame.rows > 0) {
        if(frame.type() != CV_8UC3)
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        QImage image(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
        qt_frame = image.copy();  // ????????????????
        emit frameReady(qt_frame);

        /*
        if(!image.isNull())
            cameraView->setPixmap(QPixmap::fromImage(qt_frame).scaled((fullScreen ? QSize(960, 720) : QSize(480, 360)), Qt::KeepAspectRatio));
        else
            qDebug() << "Image is null after conversion!";
        */

    }
    else
        qDebug() << "Invalid frame: empty or corrupt";
    //QImage image(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
    //cameraView->setPixmap(QPixmap::fromImage(image).scaled((fullScreen ? QSize(960, 720) : QSize(480, 360)), Qt::KeepAspectRatio));
}

void SubsectionWidget::mousePressEvent(QMouseEvent *event) {
    emit subsectionClicked(this);
    QWidget::mousePressEvent(event);
}

cv::Mat SubsectionWidget::detectShapeHybrid(cv::Mat input_frame){
    cv::Mat frame = input_frame, gray_frame, thresh_frame;
    std::vector<std::vector<cv::Point>> sector_contours, contours_copy;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

    cv::threshold(gray_frame, thresh_frame, 100, 255, cv::THRESH_BINARY_INV);
    //std::vector<cv::Vec4i> hierarchy;
    auto start = std::chrono::high_resolution_clock::now();
    cv::findContours(thresh_frame, sector_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    auto end = std::chrono::high_resolution_clock::now();
    qDebug() << "first contours: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    //cv::findContours(thresh_frame, contours_copy, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    /*
    if(contours_copy.empty()){
        qDebug() << "no contours";
        return cv::Mat();
    }
    qDebug() << "contours: " << contours_copy.size();
    for (int i = 0; i < contours_copy.size(); i++) {
        if (hierarchy[i][2] == -1) {
            sector_contours.push_back(contours_copy[i]);
        }
    }*/
    //return thresh_frame;

    cv::Rect task_sector;
    double min_dis = DBL_MAX;
    long long area = 0;
    for(int i = 0; i < sector_contours.size(); i++){
        double area = cv::contourArea(sector_contours[i]);
        if(area < 10000) continue;
        cv::Rect temp = cv::boundingRect(sector_contours[i]);
        double dis = temp.x*temp.x + (frame.rows - temp.y)*(frame.rows - temp.y);
        area = max(area, temp.area());
        if(dis < min_dis){
            min_dis = dis;
            task_sector = temp;
        }
        cv::rectangle(frame, temp, cv::Scalar(255, 0, 255), 2);
    }
    if(min_dis == DBL_MAX){
        qDebug() << "no task sector";
        return cv::Mat();
    }
    cv::rectangle(frame, task_sector, cv::Scalar(0, 255, 255), 2);
    cv::Mat final = gray_frame(task_sector);


    //cv::HoughCircles(task_mat, circles, cv::HOUGH_GRADIENT, 1, task_mat.rows/8, 100, 50, task_mat.rows/8, task_mat.rows/3);
    //cv::Mat mask = cv::Mat::ones(task_mat.size(), CV_8UC1) * 255;

    std::vector<cv::Vec3f> circles;

    start = std::chrono::high_resolution_clock::now();
    cv::HoughCircles(final, circles, cv::HOUGH_GRADIENT, 1, final.rows/8, 100, 50, final.rows/8, final.rows/3);
    end = std::chrono::high_resolution_clock::now();
    qDebug() << "circles: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    cv::Mat mask = cv::Mat::ones(final.size(), CV_8UC1) * 255;

    //qDebug() << "circles: " << circles.size();
    for(int i = 0; i < sector_contours.size(); i++){
        double area = cv::contourArea(sector_contours[i]);
        if(area < 10000) continue;
        cv::Rect temp = cv::boundingRect(sector_contours[i]);
        double dis = temp.x*temp.x + (frame.rows - temp.y)*(frame.rows - temp.y);
        if(dis < min_dis){
            min_dis = dis;
            task_sector = temp;
        }
        cv::rectangle(frame, temp, cv::Scalar(255, 0, 255), 2);
    }

    min_dis = DBL_MAX;
    cv::Vec3f circle_sector;

    for(int i = 0; i < circles.size(); i++){
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]) + 5;
        cv::circle(mask, center, radius, cv::Scalar(0), 8);
        double dis = circles[i][0]*circles[i][0] + circles[i][1]*circles[i][1];
        if(dis < min_dis){
            min_dis = dis;
            circle_sector = { circles[i][0], circles[i][1], circles[i][2] };
        }
    }
    if(min_dis == DBL_MAX){
        qDebug() << "no inner circles sector";
        return cv::Mat();
    }
    cv::circle(frame, cv::Point(cvRound(circle_sector[0]+task_sector.x), cvRound(circle_sector[1]+task_sector.y)), cvRound(circle_sector[2])-10, cv::Scalar(0, 0, 255), 2);

    std::vector<std::vector<cv::Point>> contours;

    /*
    cv::Mat masked;
    cv::bitwise_and(morphed, mask, masked);
    mask = cv::Mat::zeros(task_thresh.size(), CV_8UC1);
    if(circles.size() == 0){
        cv::imshow("circles", task_mat);
        qDebug() << "no circles found";
        return cv::Mat();
    }
    cv::circle(mask, cv::Point(cvRound(circles[0][0]), cvRound(circles[0][1])), cvRound(circles[0][2])-15, cv::Scalar(255), -1);
    cv::Mat result;
    task_thresh.copyTo(result, mask);
    cv::imshow("please", result);
    cv::findContours(masked, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    */

    cv::Mat mask_roi = cv::Mat::zeros(final.size(), CV_8UC1), roi, ahorasi;
    // FIX THIS SHIT BRUH;
    //circle_sector[0] += task_sector.x;
    //circle_sector[1] += task_sector.y;
    cv::circle(mask_roi, cv::Point(cvRound(circle_sector[0]), cvRound(circle_sector[1])), cvRound(circle_sector[2])-10, cv::Scalar(255), -1);
    final.copyTo(roi, mask_roi);
    cv::Rect bounding_box2 = cv::boundingRect(mask_roi);
    cv::Mat finalfinal = roi(bounding_box2);
    cv::threshold(roi, ahorasi, 200, 255, cv::THRESH_BINARY);
    start = std::chrono::high_resolution_clock::now();
    cv::findContours(ahorasi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    end = std::chrono::high_resolution_clock::now();
    qDebug() << "second contuors: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    std::vector<std::vector<cv::Point>> filtered_contours;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area <= 10.0) continue;

        cv::Rect bound_rect = cv::boundingRect(contour);
        double aspect_ratio = static_cast<double>(bound_rect.width) / bound_rect.height;

        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double hull_area = cv::contourArea(hull);
        double solidity = area / hull_area;

        if (aspect_ratio > 0.5 && aspect_ratio < 1.5 && solidity > 0.5) {
            filtered_contours.push_back(contour);
        }
    }

    if (!filtered_contours.empty()) {
        //cv::Point center(task_mat.cols / 2, task_mat.rows / 2);
        cv::Point center(ahorasi.cols / 2, ahorasi.rows / 2);
        double min_distance = DBL_MAX;
        int best_contour_idx = -1;

        for (int i = 0; i < filtered_contours.size(); ++i) {
            cv::Moments m = cv::moments(filtered_contours[i]);
            if (m.m00 != 0) {
                cv::Point center_of_mass(m.m10 / m.m00, m.m01 / m.m00);
                center.x += task_sector.x;
                center.y += task_sector.y;
                double distance = cv::norm(center_of_mass - center);
                if (distance < min_distance){
                    min_distance = distance;
                    best_contour_idx = i;
                }
                /*
                cv::Rect bounding_box3 = cv::boundingRect(filtered_contours[i]);
                bounding_box3.x += bounding_box.x - 10;
                bounding_box3.y += bounding_box.y - 10;
                bounding_box3.width += 20;
                bounding_box3.height += 20;
                cv::rectangle(frame, bounding_box3, cv::Scalar(255, 255, 0), 2);
                */

            }
        }

        if (best_contour_idx >= 0) {
            cv::Rect bounding_box3 = cv::boundingRect(filtered_contours[best_contour_idx]);
            //bounding_box.x += task_sector.x - 10;
            //bounding_box.y += task_sector.y - 10;
            bounding_box3.x += task_sector.x - 10;
            bounding_box3.y += task_sector.y - 10;
            bounding_box3.width += 20;
            bounding_box3.height += 20;
            cv::rectangle(frame, bounding_box3, cv::Scalar(0, 255, 0), 2);
        }

    }
    return frame;
}

cv::Mat SubsectionWidget::detectShapeContours(const cv::Mat& input_frame){

    cv::Mat gray, thresh, frame = input_frame.clone();

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, thresh, 240, 255, cv::THRESH_BINARY);

    int morph_size = 3;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));

    cv::erode(thresh, thresh, element);
    //cv::erode(thresh, thresh, element);
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, element);
    return thresh;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double min_dis = DBL_MAX;
    cv::Rect actual_box;
    for(int i = 0; i < contours.size(); i++){
        cv::Rect box = cv::boundingRect(contours[i]);
        double aspectRatio = (double)box.width / box.height;
        std::vector<cv::Point> hull;
        cv::convexHull(contours[i], hull);
        double hullArea = cv::contourArea(hull);
        double solidity = cv::contourArea(contours[i]) / hullArea;
        double dis = box.x*box.x + (frame.rows-box.y)*(frame.rows-box.y);
        if(aspectRatio > 0.8 && aspectRatio < 1.2 && solidity > 0.8 && dis < min_dis){
            min_dis = dis;
            box.width += 10;
            box.height += 10;
            box.x -= 5;
            box.y -= 5;
            actual_box = box;
        }
        cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);
        /*
    if(aspectRatio > 0.8 && aspectRatio < 1.2 && solidity > 0.8)
        cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2);
    else
        cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);
    */
    }
    cv::rectangle(frame, actual_box, cv::Scalar(0, 255, 0), 2);

    return frame;

    /*
    cv::Mat frame = input_frame.clone();
    cv::Mat gray_quadrant;
    cv::cvtColor(input_frame, gray_quadrant, cv::COLOR_BGR2GRAY);
    cv::Mat thresh_quadrant;
    cv::threshold(gray_quadrant, thresh_quadrant, 50, 255, cv::THRESH_BINARY_INV);
    std::vector<std::vector<cv::Point>> quadrant_contours;

    return thresh_quadrant;
    cv::findContours(thresh_quadrant, quadrant_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect roi;
    double best_dist = DBL_MAX;
    for(const auto& contour : quadrant_contours) {
        double area = cv::contourArea(contour);
        if(area < 1000) continue;
        cv::Rect temp = cv::boundingRect(contour);
        double dist = temp.x*temp.x + (input_frame.rows - temp.y)*(input_frame.rows - temp.y);
        if(dist < best_dist){
            best_dist = dist;
            roi = temp;
        }
    }
    int padding = 5;
    roi.x = max(0, roi.x - padding);
    roi.y = max(0, roi.y - padding);
    roi.width = min(input_frame.cols - roi.x, roi.width + 2 * padding);
    roi.height = min(input_frame.rows - roi.y, roi.height + 2 * padding);
    cv::rectangle(frame, roi, cv::Scalar(0, 255, 255), 2);

    cv::Mat roi_mat = frame(roi);
    cv::Mat gray;
    cv::cvtColor(roi_mat, gray, cv::COLOR_BGR2GRAY);
    cv::Mat thresh;
    cv::threshold(gray, thresh, 200, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat morphed;
    cv::morphologyEx(thresh, morphed, cv::MORPH_OPEN, kernel);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, gray.rows / 8, 100, 30, gray.rows / 4, gray.rows / 2);

    cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1) * 255;
    for (const auto& circle : circles) {
        cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
        int radius = cvRound(circle[2]);
        cv::circle(mask, center, radius, cv::Scalar(0), 2);
    }

    cv::Mat masked;
    cv::bitwise_and(morphed, mask, masked);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(masked, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> filtered_contours;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area <= 10.0) continue;

        cv::Rect bound_rect = cv::boundingRect(contour);
        double aspect_ratio = static_cast<double>(bound_rect.width) / bound_rect.height;

        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double hull_area = cv::contourArea(hull);
        double solidity = area / hull_area;

        if (aspect_ratio > 0.2 && aspect_ratio < 5.0 && solidity > 0.5) {
            filtered_contours.push_back(contour);
        }
    }

    if (!filtered_contours.empty()) {
        cv::Point center(gray.cols / 2, gray.rows / 2);
        double min_distance = DBL_MAX;
        int best_contour_idx = -1;

        for (int i = 0; i < filtered_contours.size(); ++i) {
            cv::Moments m = cv::moments(filtered_contours[i]);
            if (m.m00 != 0) {
                cv::Point center_of_mass(m.m10 / m.m00, m.m01 / m.m00);
                double distance = cv::norm(center_of_mass - center);
                if (distance < min_distance) {
                    min_distance = distance;
                    best_contour_idx = i;
                }
            }
        }

        if (best_contour_idx >= 0) {
            cv::Rect bounding_box = cv::boundingRect(filtered_contours[best_contour_idx]);
            bounding_box.x += roi.x;
            bounding_box.y += roi.y;
            cv::rectangle(frame, bounding_box, cv::Scalar(0, 255, 0), 2);
        }
    }
    return frame;
    */
}

cv::Mat SubsectionWidget::detectShapeHough(cv::Mat frame){
    cv::Mat gray_frame, thresh_frame;
    std::vector<std::vector<cv::Point>> shape_contours, filtered_contours;
    std::vector<cv::Vec3f> ext_circles, inner_circles;

    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    cv::Mat temp;
    cv::resize(gray_frame, temp, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
    cv::HoughCircles(temp, ext_circles, cv::HOUGH_GRADIENT, 1, temp.rows/8, 100, 50, temp.rows/8, temp.rows/4);

    double min_dis = DBL_MAX;
    cv::Vec3f ext_sector;
    for(int i = 0; i < ext_circles.size(); i++){
        ext_circles[i][0] *= 4.0;
        ext_circles[i][1] *= 4.0;
        ext_circles[i][2] *= 4.0;
        cv::Point center(cvRound(ext_circles[i][0]), cvRound(ext_circles[i][1]));
        int radius = cvRound(ext_circles[i][2]);
        double dis = center.x*center.x + (frame.rows - center.y)*(frame.rows - center.y);
        if(dis < min_dis){
            min_dis = dis;
            ext_sector = ext_circles[i];
        }
    }
    if(min_dis == DBL_MAX){
        qDebug() << "no outer circle";
        return cv::Mat();
    }

    cv::Mat ext_mask = cv::Mat::zeros(frame.size(), CV_8UC1), frame_roi;
    cv::circle(ext_mask, cv::Point(cvRound(ext_sector[0]), cvRound(ext_sector[1])), cvRound(ext_sector[2]), cv::Scalar(255), -1);
    gray_frame.copyTo(temp, ext_mask);
    cv::Rect ext_box = cv::boundingRect(ext_mask);
    frame_roi = temp(ext_box);
    if(frame_roi.empty() || frame_roi.rows < 8 || frame_roi.cols < 8){
        qDebug() << "empty final";
        return cv::Mat();
    }

    cv::HoughCircles(frame_roi, ext_circles, cv::HOUGH_GRADIENT, 1, frame_roi.rows/8, 100, 50, frame_roi.rows/8, frame_roi.rows/3);
    cv::Mat roi_mask = cv::Mat::ones(frame_roi.size(), CV_8UC1) * 255;
    for(int i = 0; i < ext_circles.size(); i++){
        cv::Point center(cvRound(ext_circles[i][0]), cvRound(ext_circles[i][1]));
        int radius = cvRound(ext_circles[i][2]) + 5;
        cv::circle(roi_mask, center, radius, cv::Scalar(0), 8);
        center.x += ext_box.x;
        center.y += ext_box.y;
        //cv::circle(frame, center, radius, cv::Scalar(255, 0, 0), 2);
    }
    if(ext_circles.empty()){
        qDebug() << "no inner circles";
        return cv::Mat();
    }
    else if(ext_circles.size() > 1)
        qWarning() << "Inner circle conflict";

    cv::Mat mask_roi = cv::Mat::zeros(frame_roi.size(), CV_8UC1), final, final_thresh;
    // FIX THIS
    cv::circle(mask_roi, cv::Point(cvRound(ext_circles[0][0]), cvRound(ext_circles[0][1])), cvRound(ext_circles[0][2])-10, cv::Scalar(255), -1);
    frame_roi.copyTo(final, mask_roi);
    cv::threshold(final, final_thresh, 200, 255, cv::THRESH_BINARY);

    cv::findContours(final_thresh, shape_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < shape_contours.size(); i++) {
        double area = cv::contourArea(shape_contours[i]);
        if (area <= 10.0) continue;
        cv::Rect bound_rect = cv::boundingRect(shape_contours[i]);
        std::vector<cv::Point> hull;
        double aspect_ratio = static_cast<double>(bound_rect.width) / bound_rect.height;
        cv::convexHull(shape_contours[i], hull);
        double solidity = area / cv::contourArea(hull);
        if (aspect_ratio > 0.5 && aspect_ratio < 1.5 && solidity > 0.5) {
            filtered_contours.push_back(shape_contours[i]);
        }
    }

    if (!filtered_contours.empty()) {
        cv::Point center(final.cols / 2, final.rows / 2);
        double min_distance = DBL_MAX;
        int best_contour_idx = -1;
        for(int i = 0; i < filtered_contours.size(); ++i) {
            cv::Moments m = cv::moments(filtered_contours[i]);
            if(m.m00 != 0) {
                cv::Point center_of_mass(m.m10 / m.m00, m.m01 / m.m00);
                center.x += ext_box.x;
                center.y += ext_box.y;
                double distance = cv::norm(center_of_mass - center);
                if (distance < min_distance){
                    min_distance = distance;
                    best_contour_idx = i;
                }
            }
        }
        if (best_contour_idx >= 0) {
            cv::Rect box = cv::boundingRect(filtered_contours[best_contour_idx]);
            box.x += ext_box.x - 10;
            box.y += ext_box.y - 10;
            box.width += 20;
            box.height += 20;
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        }
    }
    return frame;
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
            SubsectionWidget *widget = new SubsectionWidget(id++, this);
            connect(widget, &SubsectionWidget::subsectionClicked, this, [this](SubsectionWidget* clicked_widget){
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
                        subsections[k]->show();
                        left_layout->addWidget(subsections[k], k/2, k%2);
                    }
                    is_fullscreen = false;
                }
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
    qr_label = new QLabel("No data");
    qr_label->setObjectName("sensor");
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
    dashboard_layout = new QGridLayout();
    //dashboard_layout->setRowStretch(0, 1);
    //dashboard_layout->setColumnStretch(0, 1);
    std::vector<std::string> labels = {"Gas sensor: ", "QR Code: ", "Speech: ", "Magnetometer: "};
    for(int i = 0; i < labels.size(); i++){
        sensor_label = new QLabel("Gas sensor: ");
        sensor_label->setObjectName("sensor");
        dashboard_layout->addWidget(sensor_label, i, 0);
    }
    dashboard_layout->addWidget(gas_label, 0, 1);
    dashboard_layout->addWidget(qr_label, 1, 1);
    dashboard_layout->addWidget(speech_label, 2, 1);
    dashboard_layout->addWidget(magnetometer_label, 3, 1);
    dashboard_layout->addWidget(microphone_button, 4, 0);
    dashboard_layout->addWidget(clear_button, 4, 1);
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
        qr_label->setText("No data");
        speech_label->setText("No data");
        magnetometer_label->setText("No data");
    });

    // 3D MODEL VIEWER
    //model = new ModelWidget(this);
    //right_layout->addWidget(model);
    right_layout->addLayout(dashboard_layout);
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
    for(auto it = cam_map.begin(); it != cam_map.end(); it++){
        if(it->second == id)
            sub_id = it->first;
    }
    if(sub_id != -1)
        subsections[sub_id]->updateFrame(frame);
}

template<typename T> void MainWindow::updateDashbord(int index, T data){
    if(index == 0){
        if constexpr(std::is_same_v<T, int>)
            gas_label->setText(QString("%1 ppm").arg(data));
    }
    else if(index == 1){
        if constexpr(std::is_same_v<T, QString>)
            qr_label->setText(QString("%1").arg(data));
    }
    else if(index == 2){
        if constexpr(std::is_same_v<T, QString>)
            speech_label->setText(QString("%1").arg(data));
    }
    else if(index == 3){
        if constexpr(std::is_same_v<T, QVector3D>)
            magnetometer_label->setText(QString("X: %1 Y: %2\nZ: %3").arg(data.x(), 0, 'f', 2).arg(data.y(), 0, 'f', 2).arg(data.z(), 0, 'f', 2));
    }
    else
        qWarning() << "Failed to update dashboard. Out of bounds";
}

// SIGNAL INTERCEPT
void MainWindow::closeEvent(QCloseEvent* event) {
    emit windowClosing();
    qInfo() << "Closing main window...";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for(int i = 0; i < subsections.size(); i++){
        subsections[i]->destroy();
    }
    event->accept();
    qInfo() << "Bye";
}

void MainWindow::updateState(std::vector<float> data){
    // WIP
    /*
    for(int i = 0; i < data.size(); i++){
        qDebug() << "ros2 [" << i << "]: " << data[i];
    }    
    model->updateModel(data[0], data[1], data[2]);
    model->updatePivot(0, 0, data[3]);
    model->updatePivot(1, 0, data[4]);
    model->updatePivot(2, 0, data[5]);
    model->updatePivot(3, 0, data[6]);
    model->updatePivot(4, 0, data[7]);
    model->updatePivot(5, 0, data[8]);
    for(int i = 0; i < 4; i++){
        QColor color;
        if(data[i+9] < 0)
            color = QColor(nMap((int)data[i+9], -100, 0, 20, 250), 0, 0);
        else if(data[i+9] > 0)
            color = QColor(0, nMap((int)data[i+9], 0, 100, 20, 250), 0);
        else
            color = Qt::black;
        model->updateColor(i, color);
    }
    */
}

// --- DEPRECATED CLASS - USE RTPSTREAMHANDLER INSTEAD ---
/*
RTPServer::RTPServer(int port, PayloadType type, QObject *parent) : QObject(parent){
    client_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if(client_socket == INVALID_SOCKET){
        qDebug("[e] Could not create socket");
        WSACleanup();
    }
    int recv_buff_size = 1024 * 1024;  // 1MB
    setsockopt(client_socket, SOL_SOCKET, SO_RCVBUF, (char*)&recv_buff_size, sizeof(recv_buff_size));
    socket_address.sin_family = AF_INET;
    socket_address.sin_port = htons(port);
    socket_address.sin_addr.s_addr = INADDR_ANY;
    if(::bind(client_socket, (struct sockaddr*)&socket_address, socket_address_size) == SOCKET_ERROR){
        qDebug() << "bind failed, error:" << WSAGetLastError();
    }
    is_running = true;
    stream = new Stream;
    stream->type = type;
    stream->is_initialized = false;
    if(type == PayloadType::AUDIO_PCM){
        int error;
        opus_decoder = opus_decoder_create(SAMPLE_RATE, 1, &error);
    }
    else listening_thread = std::thread(&RTPServer::startListening, this);
}

RTPServer::~RTPServer(){
    is_running = false;
    closesocket(client_socket);
    if(listening_thread.joinable()) listening_thread.join();
    if(stream->type == PayloadType::AUDIO_PCM) opus_decoder_destroy(opus_decoder);
}

void RTPServer::startListening(){
    std::vector<char> packet(MAX_PACKET_SIZE);
    int socket_address_size = sizeof(socket_address);
    sockaddr_in senderAddr;
    int senderAddrLen = sizeof(senderAddr);
    qDebug() << "server awaiting data";
    while(is_running){
        //packet.resize(MAX_PACKET_SIZE);
        //int bytes_received = recvfrom(client_socket, reinterpret_cast<char*>(packet.data()), MAX_PACKET_SIZE, 0, (struct sockaddr*)&socket_address, &socket_address_size);
        int bytes_received = recvfrom(client_socket, packet.data(), packet.size(), 0, (struct sockaddr*)&socket_address, &socket_address_size);
        if(bytes_received == SOCKET_ERROR){
            qDebug() << "socket error: " << WSAGetLastError();
            continue;
        }
        else if(bytes_received < sizeof(RTPHeader)) continue;
        qDebug() << "bytes received: " << bytes_received;


        header = new RTPHeader;
        std::memcpy(header, packet.data(), sizeof(RTPHeader));
        if(!stream->is_initialized) stream->is_initialized = true;
        else if(header->seq_num != stream->seq_num + 1 && stream->seq_num != 65535){
            qDebug() << "Packet loss detected! Expected: " << stream->seq_num + 1 << " Got: " << header->seq_num;
        }
        stream->seq_num = header->seq_num;


        if(stream->type == PayloadType::ROS2_ARRAY && floatCallback){
            //std::vector<float> data((bytes_received - sizeof(RTPHeader)) / sizeof(float));
            //std::memcpy(data.data(), packet.data() + sizeof(RTPHeader), bytes_received - sizeof(RTPHeader));
            std::vector<float> data((bytes_received - sizeof(RTPHeader)) / sizeof(float));
            std::memcpy(data.data(), packet.data() + sizeof(RTPHeader), bytes_received - sizeof(RTPHeader));
            floatCallback(data);
        }
        else if(stream->type == PayloadType::VIDEO_MJPEG && ucharCallback){
            //std::vector<unsigned char> data(bytes_received - sizeof(RTPHeader));
            //std::copy(packet.begin() + sizeof(RTPHeader), packet.begin() + bytes_received, data.begin());
            //maybeno std::memcpy(data.data(), packet.data() + sizeof(RTPHeader), bytes_received - sizeof(RTPHeader));
            std::vector<unsigned char> data(bytes_received - sizeof(RTPHeader));
            std::copy(packet.begin() + sizeof(RTPHeader), packet.begin() + bytes_received, data.begin());
            ucharCallback(data);
        }


        const int16_t* int16Ptr = reinterpret_cast<const int16_t*>(packet.data());
        size_t int16Size = bytes_received / 2;
        std::vector<int16_t> data(int16Ptr, int16Ptr + int16Size);


        RTPHeader* header = reinterpret_cast<RTPHeader*>(packet.data());
        char* payload = packet.data() + sizeof(RTPHeader);
        size_t payload_size = bytes_received - sizeof(RTPHeader);
        header->seq = ntohs(header->seq);
        header->timestamp = ntohl(header->timestamp);
        header->ssrc = ntohl(header->ssrc);


        //RTPHeader* header = new RTPHeader;
        //std::memcpy(header, packet.data(), sizeof(RTPHeader));

        if(!stream->is_initialized) stream->is_initialized = true;
        else if(header->seq != stream->last_seq + 1 && stream->last_seq != 65535){
            qDebug() << "Packet loss detected! Expected: " << stream->last_seq + 1 << " Got: " << header->seq;
        }
        stream->last_seq = header->seq;

        if(stream->type == PayloadType::ROS2_ARRAY && floatCallback){
            std::vector<float> data((bytes_received - sizeof(RTPHeader)) / sizeof(float));
            std::memcpy(data.data(), packet.data() + sizeof(RTPHeader), bytes_received - sizeof(RTPHeader));
            floatCallback(data);
        }
        else if((stream->type == PayloadType::AUDIO_PCM || stream->type == PayloadType::VIDEO_MJPEG) && ucharCallback){
            //std::vector<unsigned char> data(bytes_received - sizeof(RTPHeader));
            //std::copy(packet.begin() + sizeof(RTPHeader), packet.begin() + bytes_received, data.begin());
            //maybeno std::memcpy(data.data(), packet.data() + sizeof(RTPHeader), bytes_received - sizeof(RTPHeader));
            std::vector<unsigned char> data(bytes_received);
            std::copy(packet.begin(), packet.begin()+bytes_received, data.begin());
            qDebug() << "about to callback, data size: " << data.size();
            ucharCallback(data);
        }

        //if(data_callback) data_callback(payload);


        if(static_cast<PayloadType>(header->pt) == PayloadType::AUDIO_PCM || static_cast<PayloadType>(header->pt) == PayloadType::ROS2_ARRAY){
            if(data_callback) data_callback(header->ssrc, payload, payload_size);
        }
        else{
            if(stream->current_frame.data.empty()) stream->current_frame.timestamp = header->timestamp;
            stream->current_frame.data.insert(stream->current_frame.data.end(), payload, payload + payload_size);
            if(header->m){
                stream->current_frame.complete = true;
                if(data_callback) data_callback(header->ssrc, stream->current_frame.data.data(), stream->current_frame.data.size());
                stream->complete_frames.push(std::move(stream->current_frame));
                stream->current_frame = Frame();
                if(stream->complete_frames.size() > 30) stream->complete_frames.pop();
            }
        }

        //packet.clear();
    }
}

int RTPServer::audioCallback(const void* input, void* output, unsigned long frameCount, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData){
    RTPServer* self = static_cast<RTPServer*>(userData);
    return self->audioProcess(input, output, frameCount, timeInfo, statusFlags);
}

int RTPServer::audioProcess(const void* input, void* output, unsigned long frameCount, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags){
    if(!is_running) return paContinue;
    unsigned char packet[4000];
    int socket_address_size = sizeof(socket_address);
    int bytes_received = recvfrom(client_socket, (char*)packet, sizeof(packet), 0, (struct sockaddr*)&socket_address, &socket_address_size);
    if(bytes_received > sizeof(uint32_t)) {
        opus_decode(opus_decoder, packet, bytes_received, (opus_int16*)output, AUDIO_BUFFER_SIZE, 0);
    }
    return paContinue;
}

void RTPServer::sendPacket(std::vector<int> data){
    std::vector<char> packet(data.size() * sizeof(int));
    std::memcpy(packet.data(), data.data(), data.size()*sizeof(int));
    if(sendto(client_socket, packet.data(), packet.size(), 0, (struct sockaddr*)&socket_address, sizeof(socket_address)) == SOCKET_ERROR){
        qWarning() << "Packet send failed" << WSAGetLastError();
    }
}
*/

// --- ROTAS stream handler ---
RTPStreamHandler::RTPStreamHandler(int port, std::string address, PayloadType type, QObject *parent) : QObject(parent){
    stream = new Stream;
    stream->ssrc = 0;
    stream->seq_num = 0 & 0xFFFF;
    stream->timestamp = 0;
    stream->payload_type = type;
    stream->port = port;
    // --- UDP Socket init ---
    // --- send ---
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
    qInfo() << "Channel created, bound to ports (" << port << ", " << port + 1 << ")";
}

RTPStreamHandler::~RTPStreamHandler(){
    qInfo() << "Closing channel (" << stream->port << ", " << stream->port + 1 << ")";
    shutdown(recv_socket, SD_BOTH);
    closesocket(send_socket);
    closesocket(recv_socket);
}

void RTPStreamHandler::recvPacket(){
    std::vector<std::vector<char>> fragments;
    std::vector<char> packet, buffer(MAX_PACKET_SIZE);
    int i = 0, num_fragments = -1, ssrc = -1;
    do{
        int bytes_received = recvfrom(recv_socket, buffer.data(), MAX_PACKET_SIZE, 0, (struct sockaddr*)&recv_socket_address, &socket_address_size);
        qDebug() << "bytes received: " << bytes_received;
        if(bytes_received == SOCKET_ERROR){
            int error = WSAGetLastError();
            if(error != 10004)
                qCritical() << "Packet recv failed. Winsock error: " << error;
            return;
        }
        else if (bytes_received < sizeof(RTPHeader)) {
            qCritical() << "Received incomplete RTP header.";
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
            qWarning() << "Packet fragmentation error. Previous packet dropped";
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
        qCritical() << "Payload / Callback error on packet recv";

    /*
    std::vector<char> packet(MAX_PACKET_SIZE);
    int bytes_received = recvfrom(recv_socket, packet.data(), packet.size(), 0, (struct sockaddr*)&recv_socket_address, &socket_address_size);
    if(bytes_received == SOCKET_ERROR){
        int error = WSAGetLastError();
        if(error != 10004)
            qCritical() << "Packet recv failed. Winsock error: " << error;
        return;
    }
    RTPHeader* header = new RTPHeader;
    std::memcpy(header, packet.data(), sizeof(RTPHeader));
    if(bytes_received <= sizeof(RTPHeader)){
        qWarning() << "Empty packet received. Size: " << bytes_received;
        return;
    }
    if(stream->payload_type == PayloadType::ROS2_ARRAY && floatCallback){
        std::vector<float> data((bytes_received - sizeof(RTPHeader)) / sizeof(float));
        std::memcpy(data.data(), packet.data() + sizeof(RTPHeader), bytes_received - sizeof(RTPHeader));
        floatCallback(data);
    }
    else if((stream->payload_type == PayloadType::VIDEO_MJPEG || stream->payload_type == PayloadType::AUDIO_PCM) && ucharCallback){
        std::vector<unsigned char> data(bytes_received - sizeof(RTPHeader));
        std::memcpy(data.data(), packet.data() + sizeof(RTPHeader), bytes_received - sizeof(RTPHeader));
        ucharCallback(data);
    }
    else
        qCritical() << "Payload / Callback error on packet recv";
    */
}

/*
template <typename T> void RTPStreamHandler::sendPacket(std::vector<T> data){

    // --- RTP header info ---
    RTPHeader header;
    header.version = 2;
    header.p = 0;
    header.x = 0;
    header.cc = 0;
    header.m = 1;
    header.pt = static_cast<uint8_t>(stream->payload_type);
    header.seq = stream->seq_num++;
    header.timestamp = stream->timestamp;
    header.ssrc = stream->ssrc;
    stream->timestamp += 100; // fix this please
    // --- Prepare packet for send ---
    std::vector<char> packet((data.size() * sizeof(int)) + sizeof(RTPHeader));
    std::memcpy(packet.data(), &header, sizeof(RTPHeader));
    std::memcpy(packet.data() + sizeof(RTPHeader), data.data(), data.size() * sizeof(int));
    if(sendto(send_socket, packet.data(), packet.size(), 0, (struct sockaddr*)&send_socket_address, socket_address_size) == SOCKET_ERROR)
        qWarning() << "Packet send failed. Winsock error: " << WSAGetLastError();

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
            header.timestamp = 0;
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
                qWarning() << "Packet send failed on fragment " << i << ". Winsock error: " << WSAGetLastError();
            }
        }
}
*/

AppHandler::AppHandler(int port, QObject* parent) : QObject(parent){
    qInfo() << "Starting GUI...";
    window = new MainWindow;
    window->setWindowTitle("GUI - alpha");
    window->resize(1280, 720);
    QObject::connect(window, &MainWindow::windowClosing, [this](){ this->destroy(); });
    this->port = port;
    qInfo() << "Starting base stream handler...";
    base_socket = new SocketStruct;
    base_socket->target_socket = new RTPStreamHandler(port, CLIENT_IP, PayloadType::ROS2_ARRAY);
    base_socket->target_socket->setFloatCallback([this](std::vector<float> data){
        window->updateState(data);
        base_socket->float_data = data;
    });
    base_socket->is_recv_running.store(true);
    base_socket->is_send_running.store(true);
    audio_socket = new SocketStruct;
    audio_socket->target_socket = new RTPStreamHandler(port + 2, CLIENT_IP, PayloadType::AUDIO_PCM);
    audio_socket->target_socket->setUCharCallback([this](std::vector<uchar> data){
        std::vector<opus_int16> output(AUDIO_BUFFER_SIZE);
        int frames = opus_decode(opus_decoder, data.data(), data.size(), output.data(), output.size(), 0);
        Pa_WriteStream(stream, output.data(), frames);
    });
    audio_socket->is_recv_running.store(true);
    audio_socket->is_send_running.store(true);
    opus_decoder = opus_decoder_create(SAMPLE_RATE, 1, &pa_error);
    Pa_Initialize();
    Pa_OpenDefaultStream(&stream, 0, 1, paInt16, SAMPLE_RATE, AUDIO_BUFFER_SIZE, nullptr, nullptr);
    is_audio_active.store(false);
    connect(window, &MainWindow::buttonChanged, this, [this](bool is_pressed){ is_audio_active.store(is_pressed); });
    qRegisterMetaType<std::map<int, int>>("std::map<int,int>");
    connect(window, &MainWindow::selectionChanged, this, [this](std::map<int,int> cam_map){
        for(int i = 0; i < video_sockets.size(); i++){
            video_sockets[i]->is_active.store(false);
        }
        for(auto it = cam_map.begin(); it != cam_map.end(); it++){
            if(it->second >= 0)
                video_sockets[it->second]->is_active.store(true);
        }
    });
    qInfo() << "Setup complete";
}

AppHandler::~AppHandler(){
    qInfo() << "Closing program...";
    base_socket->target_socket->sendPacket(std::vector<int>{0, -1});
    base_socket->is_recv_running.store(false);
    base_socket->is_send_running.store(false);
    base_socket->target_socket->destroy();
    if(base_socket->send_thread.joinable())
        base_socket->send_thread.join();
    if(base_socket->recv_thread.joinable())
        base_socket->recv_thread.join();
    qInfo() << "Base channel closed";
    audio_socket->is_recv_running.store(false);
    audio_socket->is_send_running.store(false);
    audio_socket->target_socket->destroy();
    if(audio_socket->recv_thread.joinable())
        audio_socket->recv_thread.join();
    if(audio_socket->send_thread.joinable())
        audio_socket->send_thread.join();
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    opus_decoder_destroy(opus_decoder);
    qInfo() << "Audio channel closed";
    for(int i = 0; i < video_sockets.size(); i++){
        video_sockets[i]->is_recv_running.store(false);
        video_sockets[i]->is_send_running.store(false);
        video_sockets[i]->target_socket->destroy();
        if(video_sockets[i]->recv_thread.joinable())
            video_sockets[i]->recv_thread.join();
        if(video_sockets[i]->send_thread.joinable())
            video_sockets[i]->send_thread.join();
    }
    qInfo() << "Video channels closed";
    WSACleanup();
}

void AppHandler::init(){
    qInfo() << "Starting ROTAS stream...";
    int num_cams = 0;

    // handshake
    base_socket->target_socket->sendPacket(std::vector<int>{0, 0});
    base_socket->target_socket->recvPacket();
    {
        std::lock_guard<std::mutex> lock(base_socket->data_mutex);
        if(base_socket->float_data.empty() || base_socket->float_data[0] < 0){
            qCritical() << "Base handshake failed";
            return;
        }
        num_cams = (int)base_socket->float_data[0];
    }
    qInfo() << "Connection established. Received " << num_cams << " video sources";

    window->setCamPorts(num_cams);
    for(int i = 0; i < num_cams; i++){
        SocketStruct* video_socket = new SocketStruct;
        video_socket->target_socket = new RTPStreamHandler(port + (2 * i) + 4, CLIENT_IP, PayloadType::VIDEO_MJPEG);
        video_sockets.push_back(std::move(video_socket));
    }
    for(int i = 0; i < video_sockets.size(); i++){
        video_sockets[i]->is_active.store(false);
        video_sockets[i]->is_send_running.store(true);
        video_sockets[i]->is_recv_running.store(true);
        video_sockets[i]->target_socket->setUCharCallback([this, i](std::vector<uchar> data) { window->updateFrame(i, data); });
        video_sockets[i]->recv_thread = std::thread([i, this](){
            while(video_sockets[i]->is_recv_running.load()){
                video_sockets[i]->target_socket->recvPacket();
            }
        });
        video_sockets[i]->send_thread = std::thread([i, this](){
            while(video_sockets[i]->is_send_running.load()){
                video_sockets[i]->target_socket->sendPacket(std::vector<int>{0, (int)video_sockets[i]->is_active.load()});
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
    }
    Pa_StartStream(stream);
    audio_socket->recv_thread = std::thread([this](){
        while(audio_socket->is_recv_running.load()){
            audio_socket->target_socket->recvPacket();
        }
    });
    audio_socket->send_thread = std::thread([this](){
        while(audio_socket->is_send_running.load()){
            audio_socket->target_socket->sendPacket(std::vector<int>{0, (int)is_audio_active.load()});
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });
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
