#include "mainwindow.h"

// --- Helper funcs ---
float nMap(float n, float minIn, float maxIn, float minOut, float maxOut){
    return (n - minIn) / (maxIn - minIn) * (maxOut - minOut) + minOut;
}

cv::Mat fisheyeTransform(cv::Mat frame){
    int w = frame.cols;
    int h = frame.rows;

    // intrinsics
    cv::Matx33f K(w / 2.0f, 0, w / 2.0f,
                  0, h / 2.0f, h / 2.0f,
                  0, 0, 1);
    // k1, k2, k3, k4
    cv::Vec4f D(0, 0, 0.0, 0.0);

    cv::Mat map_1, map_2, output;
    cv::Matx33f new_k;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, frame.size(), cv::Matx33f::eye(), new_k, 0.0f);
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Matx33f::eye(), new_k, frame.size(), CV_16SC2, map_1, map_2);
    cv::remap(frame, output, map_1, map_2, cv::INTER_LINEAR);

    return output;
}

std::string parseJSON(std::string json, std::string keyword){
    if(json.find(keyword) == std::string::npos) return "";
    size_t start = json.find(keyword);
    start = json.find(":", start);
    start = json.find("\"", start);
    start++;
    size_t end = json.find("\"", start);
    std::string text = json.substr(start, end - start);
    return text;
}

// --- Controller (xbox) ---
Controller::Controller(int dead_zone){
    this->dead_zone = dead_zone;
    if(SDL_Init(SDL_INIT_GAMECONTROLLER) < 0){
        qCritical() << "SDL could not initialize: " << SDL_GetError();
        return;
    }
    if (SDL_NumJoysticks() < 1) {
        qCritical() << "No controllers connected!";
        return;
    }
    controller = SDL_GameControllerOpen(0);
    qInfo() << "Controller connected: " << SDL_GameControllerName(controller);
}

Controller::~Controller(){
    qInfo() << "Closing controller...";
    if(controller){
        SDL_GameControllerClose(controller);
    }
    SDL_Quit();
    controller = nullptr;
}

std::vector<int> Controller::readState(){
    std::vector<int> states(20, 0);
    Sint16 axis;
    if(!controller) return states;

    SDL_Event event;
    while(SDL_PollEvent(&event)){
        if(event.type == SDL_QUIT) return states;
    }

    // Left X
    axis = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_LEFTX);
    states[0] = (std::abs(axis) < dead_zone) ? 0 : static_cast<int>(nMap(axis, -32768, 32767, -255, 255));
    // Left Y
    axis = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_LEFTY);
    states[1] = (std::abs(axis) < dead_zone) ? 0 : -static_cast<int>(nMap(axis, -32768, 32767, -255, 255));

    // Right X
    axis = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_RIGHTX);
    states[2] = (std::abs(axis) < dead_zone) ? 0 : static_cast<int>(nMap(axis, -32768, 32767, -255, 255));
    // Right Y
    axis = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_RIGHTY);
    states[3] = (std::abs(axis) < dead_zone) ? 0 : -static_cast<int>(nMap(axis, -32768, 32767, -255, 255));

    // Triggers
    states[4] = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_TRIGGERLEFT) / 128;
    states[5] = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_TRIGGERRIGHT) / 128;

    // Buttons
    states[6] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_DPAD_UP) ? 1 : 0;
    states[7] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_DPAD_DOWN) ? 1 : 0;
    states[8] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_DPAD_LEFT) ? 1 : 0;
    states[9] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_DPAD_RIGHT) ? 1 : 0;
    states[10] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_START) ? 1 : 0;
    states[11] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_BACK) ? 1 : 0;
    states[12] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_LEFTSTICK) ? 1 : 0;
    states[13] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_RIGHTSTICK) ? 1 : 0;
    states[14] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_LEFTSHOULDER) ? 1 : 0;
    states[15] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER) ? 1 : 0;
    states[16] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_A) ? 1 : 0;
    states[17] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_B) ? 1 : 0;
    states[18] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_X) ? 1 : 0;
    states[19] = SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_Y) ? 1 : 0;

    return states;
}


// --- 3D viewer ---
ModelWidget::ModelWidget(QWidget *parent) : QWidget(parent){
    root = new Qt3DCore::QEntity();
    viewport = new Qt3DExtras::Qt3DWindow();
    viewport->defaultFrameGraph()->setClearColor(QColor("#202020"));
    container = QWidget::createWindowContainer(viewport, this);
    //container->setMinimumSize(QSize(480, 360));
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    container->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    //container->resize(QSize(600, 400));
    //this->resize(QSize(600, 400));
    loadModels();
    Qt3DRender::QCamera *camera = viewport->camera();
    camera->lens()->setPerspectiveProjection(45.0f, 16.0f/9.0f, 0.1f, 1000.0f);
    camera->setPosition(QVector3D(1.5f, 1.5f, 1.5f));
    camera->setViewCenter(QVector3D(0.0f, 0.0f, 0.0f));
    camera->setUpVector(QVector3D(0.0f, 1.0f, 0.0f));
    Qt3DExtras::QOrbitCameraController* cam_controller = new Qt3DExtras::QOrbitCameraController(root);
    cam_controller->setLinearSpeed(10.0f);
    cam_controller->setLookSpeed(180.0f);
    cam_controller->setCamera(camera);
    viewport->setRootEntity(root);
    //container->show();
}

ModelWidget::~ModelWidget(){
    if(root){
        delete root;
        root = nullptr;
    }
}

void ModelWidget::updateState(BasePacket model_state){
    if(root == nullptr){
        qWarning() << "ModelWidget::updateState | Invalid model: uninitialized pointer";
        return;
    }

    updateModel(-model_state.body_y, model_state.body_z, -model_state.body_x);
    updatePivot(1, 2, model_state.arm_l);
    updatePivot(2, 2, 180.0-model_state.arm_r);
    updatePivot(5, 1, model_state.art_1);
    updatePivot(6, 2, model_state.art_2);
    updatePivot(7, 2, model_state.art_3);
    // not actually hand
    //model->updatePivot(9, 2, model_state.art_4);

    QColor color;
    if(model_state.track_l < 0.0)
        color = QColor((int)nMap(model_state.track_l, -1.0, 0.0, 20, 250), 0, 0);
    else if(model_state.track_l > 0.0)
        color = QColor(0, (int)nMap(model_state.track_l, 0.0, 1.0, 20, 250), 0);
    else
        color = Qt::black;
    updateColor(1, color);

    if(model_state.track_r < 0)
        color = QColor((int)nMap(model_state.track_r, -1.0, 0.0, 20, 250), 0, 0);
    else if(model_state.track_r > 0)
        color = QColor(0, (int)nMap(model_state.track_r, 0.0, 1.0, 20, 250), 0);
    else
        color = Qt::black;
    updateColor(0, color);
}

void ModelWidget::loadModels(){
    Qt3DCore::QEntity* light_entity = new Qt3DCore::QEntity(root);
    Qt3DRender::QDirectionalLight* directional_light = new Qt3DRender::QDirectionalLight(light_entity);
    directional_light->setColor("white");
    directional_light->setIntensity(0.75);
    directional_light->setWorldDirection(QVector3D(-1.0, -1.0, -1.0));
    light_entity->addComponent(directional_light);
    std::vector<QString> mesh_addresses = {
        "../../assets/parts/body_nobands.obj",
        "../../assets/parts/left_arm.obj",
        "../../assets/parts/right_arm.obj",
        "../../assets/parts/band.obj",
        "../../assets/parts/band.obj",
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
    Qt3DExtras::QPhongMaterial* mesh_material = new Qt3DExtras::QPhongMaterial();
    mesh_material->setDiffuse(QColor("#a6a6a6"));
    for(int i = 0; i < mesh_addresses.size(); i++){
        Qt3DCore::QEntity *pivot_entity = new Qt3DCore::QEntity((i == 0 ? root : (i <= 5 ? parts[0] : parts.back())));
        Qt3DCore::QTransform *pivot_transform = new Qt3DCore::QTransform(pivot_entity);
        pivot_transform->setTranslation(pivot_translations[i]);
        pivot_transform->setRotation(pivot_rotations[i]);
        pivot_entity->addComponent(pivot_transform);
        Qt3DCore::QEntity* mesh_entity = new Qt3DCore::QEntity(pivot_entity);
        Qt3DCore::QTransform* mesh_transform = new Qt3DCore::QTransform(mesh_entity);
        Qt3DRender::QMesh* mesh = new Qt3DRender::QMesh();
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
    /*
    Qt3DRender::QMesh *mesh = new Qt3DRender::QMesh(root);
    mesh->setSource(QUrl::fromLocalFile("body_nobands.obj"));
    Qt3DExtras::QPhongMaterial *material = new Qt3DExtras::QPhongMaterial(root);
    material->setDiffuse(QColor(QRgb(0x665423)));
    Qt3DCore::QTransform *transform = new Qt3DCore::QTransform;
    transform->setScale(1.0f);
    Qt3DCore::QEntity *modelEntity = new Qt3DCore::QEntity(root);
    modelEntity->addComponent(mesh);
    modelEntity->addComponent(material);
    modelEntity->addComponent(transform);
    */
}

void ModelWidget::updateModel(float angleX, float angleY, float angleZ){
    pivots[0]->setRotation(QQuaternion::fromEulerAngles(angleX, angleY, angleZ));
}

void ModelWidget::updatePivot(int index, int axis, float angle){
    if(index >= pivots.size()){
        qCritical() << "ModelWidget::updatePivot | Invalid pivot index: out of bounds";
        return;
    }
    if(axis == 0)
        pivots[index]->setRotation(QQuaternion::fromEulerAngles(angle, pivots[index]->rotationY(), pivots[index]->rotationZ()));
    else if(axis == 1)
        pivots[index]->setRotation(QQuaternion::fromEulerAngles(pivots[index]->rotationX(), angle, pivots[index]->rotationZ()));
    else if(axis == 2)
        pivots[index]->setRotation(QQuaternion::fromEulerAngles(pivots[index]->rotationX(), pivots[index]->rotationY(), angle));
    else
        qCritical() << "ModelWidget::updatePivot | Invalid pivot axis: out of bounds";
}

void ModelWidget::updateColor(int index, QColor color){
    if(index >= band_colors.size()){
        qCritical() << "ModelWidget::updateColor | Invalid part index: out of bounds";
        return;
    }
    band_colors[index]->setDiffuse(color);
}

// --- cam subsections ---
SubsectionWidget::SubsectionWidget(int id, QWidget *parent) : QWidget(parent){
    this->id = id;
    cam_id = -1;
    container = new QWidget(this);
    container->resize(480, 360);
    current_size = QSize(480, 360);
    layout = new QVBoxLayout();
    settings = new QVBoxLayout();
    dropdowns = new QHBoxLayout();
    camera_view = new QLabel();

    camera_view->setStyleSheet("border: 1px solid gray;");
    camera_dropdown = new QComboBox();
    camera_dropdown->addItem("No Camera");
    filter_dropdown = new QComboBox();
    filter_dropdown->addItems({ "No filter", "QR Code", "Hazmat", "Shape", "Thermal" });
    filter_dropdown->setEnabled(false);
    settings_dropdown = new QComboBox();
    settings_dropdown->addItems({ "1280x720", "1920x1080", "1280x1280 *", "720x720 *" });
    settings_dropdown->setEnabled(false);
    dropdowns->addWidget(camera_dropdown);
    dropdowns->addWidget(filter_dropdown);
    dropdowns->addWidget(settings_dropdown);
    layout->addLayout(dropdowns);
    layout->addWidget(camera_view);
    layout->addLayout(settings);
    layout->setSpacing(0);
    layout->setContentsMargins(0, 0, 0, 0);
    container->setLayout(layout);

    is_cv_running.store(true);
    filters.none.store(true);
    filters.is_qr_active.store(false);
    filters.is_shape_active.store(false);
    filters.is_thermal_active.store(false);
    filters.is_hazmat_active.store(false);

    filter_settings.qr_container = new QWidget();
    filter_settings.shape_container = new QWidget();
    filter_settings.thermal_container = new QWidget();
    filter_settings.qr_layout = new QHBoxLayout();
    filter_settings.shape_layout = new QGridLayout();
    filter_settings.thermal_layout = new QGridLayout();
    filter_settings.shape_button_layout = new QGridLayout();
    filter_settings.qr_button_1 = new QPushButton("OpenCV");
    filter_settings.qr_button_2 = new QPushButton("ZBar");
    filter_settings.thermal_slider_1 = new QSlider(Qt::Horizontal);
    filter_settings.thermal_slider_2 = new QSlider(Qt::Horizontal);
    filter_settings.shape_slider_1 = new QSlider(Qt::Horizontal);
    filter_settings.shape_slider_2 = new QSlider(Qt::Horizontal);
    filter_settings.shape_button_group = new QButtonGroup();
    filter_settings.shape_label_1 = new QLabel(" Corner ");
    filter_settings.shape_label_2 = new QLabel(" Threshold ");
    filter_settings.shape_label_3 = new QLabel(" Tolerance ");
    filter_settings.shape_label_4 = new QLabel("  50 ");
    filter_settings.shape_label_5 = new QLabel(" 0.25 ");
    filter_settings.thermal_label_1 = new QLabel(" Distance (cm) ");
    filter_settings.thermal_label_2 = new QLabel(" Opacity ");
    filter_settings.thermal_label_3 = new QLabel(" 50 ");
    filter_settings.thermal_label_4 = new QLabel(" 0.5 ");
    std::vector<QString> shape_buttons{ "Upper left", "Upper right", "Lower left", "Lower right" };
    for(int i = 0; i < 4; i++){
        QPushButton* button = new QPushButton(shape_buttons[i]);
        filter_settings.shape_buttons.push_back(button);
    }

    filter_settings.qr_button_1->setStyleSheet(R"(
        QPushButton {
            color: white;
            background-color: black;
            margin: 0px;
            border-top-left-radius: 5px;
            border-bottom-left-radius: 5px;
            border: 1px solid white;
        }
        QPushButton:checked {
            background-color: gray;
        }
    )");
    filter_settings.qr_button_2->setStyleSheet(R"(
        QPushButton {
            color: white;
            background-color: black;
            margin: 0px;
            border-top-right-radius: 5px;
            border-bottom-right-radius: 5px;
            border: 1px solid white;
        }
        QPushButton:checked {
            background-color: gray;
        }
    )");
    filter_settings.qr_button_1->setCheckable(true);
    filter_settings.qr_button_1->setChecked(true);
    filter_settings.qr_button_2->setCheckable(true);
    filter_settings.qr_layout->setSpacing(0);
    filter_settings.qr_layout->setContentsMargins(5, 5, 5, 5);
    filter_settings.qr_layout->addWidget(filter_settings.qr_button_1);
    filter_settings.qr_layout->addWidget(filter_settings.qr_button_2);
    filter_settings.qr_container->setLayout(filter_settings.qr_layout);
    filter_settings.qr_container->hide();

    filter_settings.shape_button_group->setExclusive(true);
    for(int i = 0; i < filter_settings.shape_buttons.size(); i++){
        filter_settings.shape_buttons[i]->setCheckable(true);
        filter_settings.shape_buttons[i]->setStyleSheet(QString(R"(
            QPushButton {
                background-color: black;
                color: white;
                margin: 0px;
                border: 1px solid white;
                %1
            }
            QPushButton:checked {
                background-color: gray;
            }
        )").arg((i == 0 ? "border-top-left-radius: 5px;" : (
                 i == 1 ? "border-top-right-radius: 5px;" : (
                 i == 2 ? "border-bottom-left-radius: 5px;" : (
                 i == 3 ? "border-bottom-right-radius: 5px;" : "")
        )))));
        filter_settings.shape_button_group->addButton(filter_settings.shape_buttons[i], i);
        filter_settings.shape_button_layout->addWidget(filter_settings.shape_buttons[i], i/2, 1 + i%2);
    }
    filter_settings.shape_buttons[0]->setChecked(true);
    filter_settings.shape_slider_1->setRange(0, 255);
    filter_settings.shape_slider_1->setValue(50);
    filter_settings.shape_slider_1->setTickInterval(5);
    filter_settings.shape_slider_2->setRange(0, 20);
    filter_settings.shape_slider_2->setValue(5);
    filter_settings.shape_slider_2->setTickInterval(1);
    filter_settings.shape_layout->addLayout(filter_settings.shape_button_layout, 0, 1, 1, 2);
    filter_settings.shape_layout->addWidget(filter_settings.shape_label_1, 0, 0);
    filter_settings.shape_layout->addWidget(filter_settings.shape_label_2, 1, 0);
    filter_settings.shape_layout->addWidget(filter_settings.shape_slider_1, 1, 1);
    filter_settings.shape_layout->addWidget(filter_settings.shape_label_4, 1, 2);
    filter_settings.shape_layout->addWidget(filter_settings.shape_label_3, 2, 0);
    filter_settings.shape_layout->addWidget(filter_settings.shape_slider_2, 2, 1);
    filter_settings.shape_layout->addWidget(filter_settings.shape_label_5, 2, 2);
    filter_settings.shape_button_layout->setSpacing(0);
    filter_settings.shape_button_layout->setContentsMargins(0, 0, 0, 0);
    filter_settings.shape_layout->setSpacing(5);
    filter_settings.shape_layout->setContentsMargins(5, 5, 5, 5);
    filter_settings.shape_container->setLayout(filter_settings.shape_layout);
    filter_settings.shape_container->hide();

    filter_settings.thermal_slider_1->setRange(0, 100);
    filter_settings.thermal_slider_1->setValue(50);
    filter_settings.thermal_slider_1->setTickInterval(5);
    filter_settings.thermal_slider_2->setRange(0, 10);
    filter_settings.thermal_slider_2->setValue(5);
    filter_settings.thermal_slider_2->setTickInterval(1);
    filter_settings.thermal_layout->addWidget(filter_settings.thermal_label_1, 0, 1);
    filter_settings.thermal_layout->addWidget(filter_settings.thermal_slider_1, 0, 2);
    filter_settings.thermal_layout->addWidget(filter_settings.thermal_label_3, 0, 3);
    filter_settings.thermal_layout->addWidget(filter_settings.thermal_label_2, 1, 1);
    filter_settings.thermal_layout->addWidget(filter_settings.thermal_slider_2, 1, 2);
    filter_settings.thermal_layout->addWidget(filter_settings.thermal_label_4, 1, 3);
    filter_settings.thermal_layout->setSpacing(5);
    filter_settings.thermal_layout->setContentsMargins(5, 5, 5, 5);
    filter_settings.thermal_container->setLayout(filter_settings.thermal_layout);
    filter_settings.thermal_container->hide();

    connect(filter_settings.qr_button_1, &QPushButton::clicked, this, [this](){
        if(filter_settings.qr_button_2->isChecked())
            filter_settings.qr_button_2->setChecked(false);
        if(!filter_settings.qr_button_1->isChecked()){
            filter_settings.qr_button_1->setChecked(true);
            return;
        }
        std::lock_guard<std::mutex> lock(filter_settings.settings_mutex);
        filter_settings.qr_setting = 0;
    });
    connect(filter_settings.qr_button_2, &QPushButton::clicked, this, [this](){
        if(filter_settings.qr_button_1->isChecked())
            filter_settings.qr_button_1->setChecked(false);
        if(!filter_settings.qr_button_2->isChecked()){
            filter_settings.qr_button_2->setChecked(true);
            return;
        }
        std::lock_guard<std::mutex> lock(filter_settings.settings_mutex);
        filter_settings.qr_setting = 1;
    });
    connect(filter_settings.shape_button_group, &QButtonGroup::idClicked, this, [this](int id){
        std::lock_guard<std::mutex> lock(filter_settings.settings_mutex);
        filter_settings.shape_setting = id;
    });
    connect(filter_settings.shape_slider_1, &QSlider::valueChanged, this, [this](int value){
        filter_settings.shape_label_4->setText(QString(" %1 ").arg(value));
        std::lock_guard<std::mutex> lock(filter_settings.settings_mutex);
        filter_settings.shape_threshold = value;
    });
    connect(filter_settings.shape_slider_2, &QSlider::valueChanged, this, [this](int value){
        filter_settings.shape_label_5->setText(QString(" %1 ").arg(static_cast<float>(value)/20.0));
        std::lock_guard<std::mutex> lock(filter_settings.settings_mutex);
        filter_settings.shape_tolerance = static_cast<float>(value) / 20.0;
    });
    connect(filter_settings.thermal_slider_1, &QSlider::valueChanged, this, [this](int value){
        filter_settings.thermal_label_3->setText(QString(" %1 ").arg(value));
        std::lock_guard<std::mutex> lock(filter_settings.settings_mutex);
        filter_settings.thermal_distance = value;
    });
    connect(filter_settings.thermal_slider_2, &QSlider::valueChanged, this, [this](int value){
        filter_settings.thermal_label_4->setText(QString(" %1 ").arg(static_cast<float>(value)/10.0));
        std::lock_guard<std::mutex> lock(filter_settings.settings_mutex);
        filter_settings.thermal_alpha = static_cast<float>(value) / 10.0;
    });

    settings->addWidget(filter_settings.qr_container);
    settings->addWidget(filter_settings.shape_container);
    settings->addWidget(filter_settings.thermal_container);
    settings->addStretch();

    cv_thread = std::thread([this, id](){
        while(is_cv_running.load()){
            if(filters.none.load()){
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            cv::Mat frame, thermal;
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                frame = latest_frame.clone();
                thermal = thermal_frame.clone();
            }
            if(frame.empty()){
                qWarning() << "SubsectionWidget" << id << "::cv_thread | Invalid frame: empty cv::Mat passed to filters";
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            if(filters.is_qr_active.load()){
                int mode = 0;
                {
                    std::lock_guard<std::mutex> lock(filter_settings.settings_mutex);
                    mode = filter_settings.qr_setting;
                }
                frame = filters.detectQR(frame, mode);
            }
            else if(filters.is_hazmat_active.load())
                frame = filters.detectHazmat(frame);
            else if(filters.is_shape_active.load()){
                int corner = 0, threshold = 0, tolerance = 0;
                {
                    std::lock_guard<std::mutex> lock(filter_settings.settings_mutex);
                    corner = filter_settings.shape_setting;
                    threshold = filter_settings.shape_threshold;
                    tolerance = filter_settings.shape_tolerance;

                }
                frame = filters.detectShape(frame, corner, false, threshold, tolerance);
            }
            else if(filters.is_thermal_active.load()){
                int distance = 0;
                float opacity = 0.0;
                {
                    std::lock_guard<std::mutex> lock(filter_settings.settings_mutex);
                    distance = filter_settings.thermal_distance;
                    opacity = filter_settings.thermal_alpha;
                }
                thermal = filters.thermalAdaptiveInterpolation(thermal);
                frame = filters.thermalOverlay(frame, thermal, distance, opacity);
                //frame = process_camera_frames(frame, thermal, 40, 0.5);
            }
            else{
                qWarning() << "SubsectionWidget" << id << "::cv_thread | Invalid filter: no filter flag active";
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(filter_mutex);
                filter_frame = frame.clone();
            }
        }
    });
    connect(camera_dropdown,  &QComboBox::currentIndexChanged, this, [this](int index){
        cam_id = index - 1;
        if(index != 0){
        qDebug() << "calling resolution on " << cam_id;
        emit resolutionChanged(cam_id, 1280, 720);
        }
        qDebug() << "before emit";
        emit selectionChanged();
        filter_dropdown->setCurrentIndex(0);
        settings_dropdown->blockSignals(true);
        settings_dropdown->setCurrentIndex(0);
        settings_dropdown->blockSignals(false);
        if(index == 0){
            qDebug() << "before enabled";
            filter_dropdown->setEnabled(false);
            settings_dropdown->setEnabled(false);
            if(!fullScreen)
                current_size = QSize(480, 360);
            else
                current_size = QSize(960, 720);
            qDebug() << "before update";
            updateFrame(cv::imread("../../assets/imgs/404.png"));
        }
        else{
            filter_dropdown->setEnabled(true);
            settings_dropdown->setEnabled(true);
        }
    });
    connect(filter_dropdown, &QComboBox::currentIndexChanged, this, [this](int index){
        filters.none.store(index == 0);
        filters.is_qr_active.store(index == 1);
        filters.is_hazmat_active.store(index == 2);
        filters.is_shape_active.store(index == 3);
        filters.is_thermal_active.store(index == 4);

        if(index == 0){
            filter_settings.qr_container->hide();
            filter_settings.shape_container->hide();
            filter_settings.thermal_container->hide();
        }
        else if(index == 1){
            filter_settings.shape_container->hide();
            filter_settings.thermal_container->hide();
            filter_settings.qr_container->show();
        }
        else if(index == 3){
            filter_settings.thermal_container->hide();
            filter_settings.qr_container->hide();
            filter_settings.shape_container->show();
        }
        else if(index == 4){
            filter_settings.thermal_container->show();
            filter_settings.qr_container->hide();
            filter_settings.shape_container->hide();
        }
        else{
            filter_settings.qr_container->hide();
            filter_settings.shape_container->hide();
            filter_settings.thermal_container->hide();
        }
    });
    connect(settings_dropdown, &QComboBox::currentIndexChanged, this, [this](int index){
        qDebug() << "settings changed";
        int width, height;
        switch(index){
            case 0:
            default:
                width = 1280;
                height = 720;
                break;
            case 1:
                width = 1920;
                height = 1080;
                break;
            case 2:
                width = 1280;
                height = 1280;
                break;
            case 3:
                width = 720;
                height = 720;
                break;
        }
        if(width == height){
            if(!fullScreen)
                current_size = QSize(480, 480);
            else
                current_size = QSize(960, 960);
            container->resize(current_size);
        }
        else{
            if(!fullScreen)
                current_size = QSize(480, 360);
            else
                current_size = QSize(960, 720);
            container->resize(current_size);
        }
        emit resolutionChanged(camera_dropdown->currentIndex()-1, width, height);
    });

    camera_view->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    camera_view->setPixmap(QPixmap("../../assets/imgs/404.png").scaled(QSize(480, 270)));
    connect(this, &SubsectionWidget::frameReady, this, [this, id](QImage image){
        if(!image.isNull()){
            //camera_view->setPixmap(QPixmap::fromImage(qt_frame).scaled(container->size(), Qt::KeepAspectRatio));
            camera_view->setPixmap(QPixmap::fromImage(qt_frame).scaled(current_size, Qt::KeepAspectRatio));
        }
        else
            qCritical() << "SubsectionWidget" << id << "::frameReady | Invalid frame: null QImage";
    });

    std::ifstream file(LABELS_PATH);
    if(!file.is_open()) {
        qCritical() << "SubsectionWidget" << id << "::constructor | Invalid hazmat labels: file can't be opened";
        return;
    }
    std::string line;
    while(std::getline(file, line)){
        if(!line.empty())
            filters.labels.push_back(line);
    }
    file.close();
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for(int i = 0; i < filters.labels.size(); i++) {
        filters.colors.emplace_back(dist(rng), dist(rng), dist(rng));
    }

    filters.hazmat_model = cv::dnn::DetectionModel(CFG_PATH, WEIGHTS_PATH);
    filters.hazmat_model.setInputScale(1.0 / 255.0);
    filters.hazmat_model.setInputSize(INPUT_SIZE);
    filters.hazmat_model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    filters.hazmat_model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

// --- Visual filters ---
cv::Mat SubsectionWidget::Filters::thermalAdaptiveInterpolation(cv::Mat frame){
    cv::resize(frame, frame, cv::Size(64, 64), 0, 0, cv::INTER_CUBIC);
    double min_val, max_val;
    cv::minMaxLoc(frame, &min_val, &max_val);
    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(frame, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(frame, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_mag);
    cv::Mat up_gradient;
    cv::resize(grad_mag, up_gradient, cv::Size(64, 64), 0, 0, cv::INTER_LINEAR);
    cv::normalize(up_gradient, up_gradient, 0, 1, cv::NORM_MINMAX);
    cv::Mat enhanced = frame.clone();
    for(int y = 0; y < enhanced.rows; y++){
        for(int x = 0; x < enhanced.cols; x++){
            enhanced.at<float>(y, x) = frame.at<float>(y, x) * (1.0f + up_gradient.at<float>(y, x) * 0.3f);
        }
    }
    cv::normalize(enhanced, enhanced, min_val, max_val, cv::NORM_MINMAX);
    cv::Mat normalized;
    cv::normalize(enhanced, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    cv::Mat colormap;
    cv::applyColorMap(normalized, colormap, cv::COLORMAP_JET);
    frame = colormap.clone();
    cv::resize(frame, frame, cv::Size(1280, 720), cv::INTER_NEAREST);
    cv::rotate(frame, frame, cv::ROTATE_180);
    return frame;
    /*
    double min_val, max_val;
    cv::Mat grad_x, grad_y, output;
    cv::resize(frame, frame, cv::Size(64, 64), 0, 0, cv::INTER_CUBIC);
    cv::minMaxLoc(frame, &min_val, &max_val);
    cv::Sobel(frame, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(frame, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, output);
    cv::resize(output, output, cv::Size(64, 64), 0, 0, cv::INTER_LINEAR);
    cv::normalize(output, output, 0, 1, cv::NORM_MINMAX);
    for(int i = 0; i < frame.rows; i++){
        for(int j = 0; j < frame.cols; j++){
            output.at<float>(i, j) = frame.at<float>(i, j) * (1.0f + output.at<float>(i, j) * 0.3f);
        }
    }
    cv::normalize(output, output, min_val, max_val, cv::NORM_MINMAX);
    cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
    output.convertTo(output, CV_8U);
    cv::applyColorMap(output, output, cv::COLORMAP_JET);
    //cv::resize(output, output, cv::Size(1280, 720), cv::INTER_CUBIC);
    cv::resize(output, output, cv::Size(1280, 720), cv::INTER_NEAREST);
    return output;
    */
}

cv::Mat SubsectionWidget::Filters::thermalOverlay(cv::Mat frame, cv::Mat thermal, float distance, float alpha){
    if(distance == 0.0) return thermal;

    float cam_width = 2.0 * distance * tan(CAM_FOV.first / 2.0 * DEG_TO_RAD), cam_height = 2.0 * distance * tan(CAM_FOV.second / 2.0 * DEG_TO_RAD),
        thermal_width = 2.0 * distance * tan(THERMAL_FOV.first / 2.0 * DEG_TO_RAD), thermal_height = 2.0 * distance * tan(THERMAL_FOV.second / 2.0 * DEG_TO_RAD);
    float left = std::max(-thermal_width/2.0, THERMAL_X_DIFF-(cam_width/2.0)), right = std::min(thermal_width/2.0, THERMAL_X_DIFF+(cam_width/2.0)),
        up = std::min(thermal_height/2.0, (cam_height/2.0)-THERMAL_Y_DIFF), down = std::max(-thermal_height/2.0, -(cam_height/2.0)-THERMAL_Y_DIFF);
    int cam_overlap_width = (right-left) / cam_width * frame.cols, cam_overlap_height = (up-down) / cam_height * frame.rows,
        thermal_overlap_width = (right-left) / thermal_width * thermal.cols, thermal_overlap_height = (up-down) / thermal_height * thermal.rows;
    //qDebug() << cam_width << " " << thermal_width << " " << left << " " << right << " " << cam_overlap_width << " " << thermal_overlap_width;
    //qDebug() << cam_height << " " << thermal_height << " " << up << " " << down << " " << cam_overlap_height << " " << thermal_overlap_height;
    //qDebug() << thermal.cols-thermal_overlap_width << " " << thermal.cols-thermal_overlap_height << " " << thermal_overlap_width << " " << thermal_overlap_height;
    if(thermal_overlap_width <= 0 || thermal_overlap_height <= 0 || cam_overlap_width <= 0 || cam_overlap_height <= 0)
        return placeText("BOUNDS ERROR", frame);
    cv::Rect thermal_overlap(thermal.cols-thermal_overlap_width, thermal.rows-thermal_overlap_height, thermal_overlap_width, thermal_overlap_height);
    //qDebug() << frame.cols-cam_overlap_width << " " << frame.cols-cam_overlap_height << " " << cam_overlap_width << " " << cam_overlap_height;
    //cv::Rect frame_overlap(0, 0, cam_overlap_width, cam_overlap_height);
    cv::Rect frame_overlap(0, 0, cam_overlap_width, cam_overlap_height);
    frame = frame(frame_overlap);
    //qDebug() << "done frame";
    //qDebug() << "thermal: " << thermal.cols << " " << thermal.rows << " " << thermal_overlap.x << " " << thermal_overlap.y << " " << thermal_overlap.width << " " << thermal_overlap.height;
    thermal = thermal(thermal_overlap);
    //qDebug() << "done thermal";
    cv::resize(frame, frame, cv::Size(1280, 720), cv::INTER_NEAREST);
    cv::resize(thermal, thermal, cv::Size(1280, 720), cv::INTER_NEAREST);
    cv::addWeighted(frame, 1.0 - alpha, thermal, alpha, 0.0, frame);

    return frame;
}

cv::Mat SubsectionWidget::Filters::detectHazmat(cv::Mat frame){
    int H = frame.rows;
    int W = frame.cols;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    hazmat_model.detect(frame, classIds, confidences, boxes, CONF_THRESH, NMS_THRESH);
    if(classIds.empty())
        return placeText("NO HAZMAT DETECTED", frame);

    for(int i = 0; i < classIds.size(); i++) {
        int cid = classIds[i];
        float conf = confidences[i];
        cv::Rect box = boxes[i];
        int cx = box.x + box.width / 2;
        int cy = box.y + box.height / 2;
        float ncx = static_cast<float>(cx) / W;
        float ncy = static_cast<float>(cy) / H;

        cv::Scalar color = colors[cid];
        std::string label = labels[cid] + ": " + cv::format("%.2f", conf);
        cv::rectangle(frame, box, color, 2);
        cv::putText(frame, label, cv::Point(box.x, box.y - 15), cv::FONT_HERSHEY_SIMPLEX, 1.5, color, 3);
    }
    return frame;
}

cv::Mat SubsectionWidget::Filters::placeText(std::string text, cv::Mat frame){
    int temp = 0;
    cv::Size text_size = cv::getTextSize(text,  cv::FONT_HERSHEY_SIMPLEX, 3, 3, &temp);
    cv::putText(frame, text, cv::Point((frame.cols - text_size.width)/2, (frame.rows + text_size.height)/2),  cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255, 0, 0), 3);
    return frame;
}

cv::Mat SubsectionWidget::Filters::detectQR(cv::Mat frame, int mode){
    std::vector<cv::Point> points;
    std::string decoded_text;
    if(mode == 0){
        cv::QRCodeDetector qr_decoder;
        try{
            decoded_text = qr_decoder.detectAndDecode(frame, points);
        }
        catch(const cv::Exception& e){
            return placeText("NO QR CODE DETECTED, EXC", frame);
        }
    }
    else{
        cv::Mat grayscale;
        cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(grayscale, grayscale, cv::Size(3, 3), 0);
        grayscale = grayscale.clone();
        cv::resize(grayscale, grayscale, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
        zbar::ImageScanner scanner;
        scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 0);
        scanner.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);
        zbar::Image zbar_frame(grayscale.cols, grayscale.rows, "Y800", (uchar*)grayscale.data, grayscale.cols*grayscale.rows);
        int codes = scanner.scan(zbar_frame);

        if(codes == 0)
            return placeText("NO QR CODE DETECTED", frame);

        for(zbar::Image::SymbolIterator symbol = zbar_frame.symbol_begin(); symbol != zbar_frame.symbol_end(); ++symbol){
            decoded_text = symbol->get_data();
            for(int i = 0; i < symbol->get_location_size(); ++i) {
                points.emplace_back(cv::Point(symbol->get_location_x(i)/2.0, symbol->get_location_y(i)/2.0));
            }
        }
    }

    if(!decoded_text.empty() && !points.empty()){
        std::vector<std::vector<cv::Point>> contour = { points };
        cv::polylines(frame, contour, true, cv::Scalar(0, 0, 255), 5);
        cv::Point corner = *std::min_element(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b){
            return (a.x + a.y) < (b.x + b.y);
        });
        cv::putText(frame, decoded_text, corner+cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
    }
    else
        frame = placeText("NO QR CODE DETECTED", frame);

    return frame;
}

cv::Mat SubsectionWidget::Filters::detectShape(cv::Mat frame, int corner, bool mode, int threshold, double shape_tolerance){
    double scale = 1.0, min_dis = DBL_MAX;
    cv::Mat gray_frame, gray_resized, inv_thresh, inv_task_sector, task_sector;
    std::vector<cv::Vec3f> circles;
    std::vector<std::vector<cv::Point>> contours, shapes;
    cv::Rect sector;

    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    cv::resize(gray_frame, gray_resized, cv::Size(), 1.0/scale, 1.0/scale, cv::INTER_AREA);
    cv::threshold(gray_resized, inv_thresh, threshold, 255, cv::THRESH_BINARY_INV);

    std::vector<cv::Point> cont;
    if(mode){
        cv::Vec3f circ_sector;
        cv::HoughCircles(gray_resized, circles, cv::HOUGH_GRADIENT, 1, gray_resized.rows/8.0, 100, 50, gray_resized.rows/8, gray_resized.rows/4);
        for(int i = 0; i < circles.size(); i++) {
            double x = circles[i][0] * scale,
                y = circles[i][1] * scale,
                r = circles[i][2] * scale;
            double dis = (x*x) + (frame.rows-y)*(frame.rows-y);
            cv::circle(frame, cv::Point(x, y), r, cv::Scalar(255, 0, 0), 4);
            if(dis < min_dis){
                min_dis = dis;
                circ_sector = cv::Vec3f(x, y, r);
            }
        }
        if(min_dis == DBL_MAX)
            return placeText("NO CIRCLES", frame);
        cv::Mat mask = cv::Mat::zeros(inv_thresh.size(), CV_8UC1);
        cv::circle(mask, cv::Point(circ_sector[0], circ_sector[1]), circ_sector[2], cv::Scalar(255), -1);
        cv::bitwise_and(inv_thresh, inv_thresh, inv_task_sector, mask);
        sector = cv::boundingRect(mask);
        inv_task_sector = inv_task_sector(sector);
    }
    else{
        cv::findContours(inv_thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2){
            return cv::contourArea(c1) > cv::contourArea(c2);
        });
        for(int i = 0; i < contours.size(); i++){
            if(cv::contourArea(contours[i]) < 1000) break;
            cv::Rect contour = cv::boundingRect(contours[i]);
            cv::Point center(contour.x + contour.width/2, contour.y + contour.height/2);
            double dis = cv::norm(center - cv::Point(((corner == 0 || corner == 2) ? 0 : inv_thresh.cols), (corner <= 1 ? 0 : inv_thresh.rows)));
            if(dis < min_dis){
                min_dis = dis;
                sector = contour;
                cont = contours[i];
            }
        }
        if(min_dis == DBL_MAX)
            return placeText("NO CONTOURS", frame);
        inv_task_sector = inv_thresh(sector);
    }
    cv::rectangle(frame, sector, cv::Scalar(0, 0, 255), 3);

    cv::bitwise_not(inv_task_sector, task_sector);
    cv::findContours(task_sector, shapes, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    //frame = frame(sector);
    //cv::drawContours(frame, shapes, -1, cv::Scalar(0, 255, 0), 1);

    qDebug() << "found " << shapes.size();

    min_dis = DBL_MAX;
    std::vector<cv::Point> shape;
    cv::Point sector_center(task_sector.cols/2, task_sector.rows/2);
    for(int i = 0; i < shapes.size(); i++){
        double area = cv::contourArea(shapes[i]);
        if(area <= 100.0) continue;
        cv::Rect box = cv::boundingRect(shapes[i]);
        std::vector<cv::Point> hull;
        cv::convexHull(shapes[i], hull);
        if(box.height == 0 || hull.empty()) continue;
        double aspect_ratio = static_cast<double>(box.width) / box.height;
        double solidity = area / cv::contourArea(hull);
        if(aspect_ratio < 1.0f - shape_tolerance || aspect_ratio > 1.0f + shape_tolerance || solidity < 1.0f - shape_tolerance) continue;
        cv::Rect contour = cv::boundingRect(shapes[i]);
        cv::Point shape_center(contour.x + contour.width/2, contour.y + contour.height/2);
        //double dis = cv::norm(shape_center - sector_center); // prob not working as intended
        double dis = (shape_center.x-sector_center.x)*(shape_center.x-sector_center.x) + (shape_center.y-sector_center.y)*(shape_center.y-sector_center.y);
        if(dis < min_dis){
            min_dis = dis;
            shape = shapes[i];
        }
    }
    if(min_dis == DBL_MAX)
        return placeText("NO SHAPE", frame);

    cv::Rect box = cv::boundingRect(shape), final;
    final.x = box.x + sector.x - 10;
    final.y = box.y + sector.y - 10;
    final.width = box.width + 20;
    final.height = box.height + 20;
    cv::rectangle(frame, final, cv::Scalar(0, 255, 0), 5);
    return frame;
}

SubsectionWidget::~SubsectionWidget(){
    filters.none.store(true);
    is_cv_running.store(false);
    cv_thread.join();
}

void SubsectionWidget::setAvailableDevices(int num_cams, std::vector<std::string> cam_names) {
    for(int i = 1; i <= num_cams; i++){
        camera_dropdown->addItem(QString::fromStdString(cam_names[i-1]), i);
    }
}

void SubsectionWidget::updateFrame(cv::Mat frame, cv::Mat thermal, std::vector<uchar> compressed){
    {
        std::lock_guard<std::mutex> lock(compressed_mutex);
        latest_compressed = compressed;
    }
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        latest_frame = frame;
        thermal_frame = thermal;
    }
    if(!filters.none.load()){
        std::lock_guard<std::mutex> lock(filter_mutex);
        if(!filter_frame.empty())
            frame = filter_frame;
    }
    if(!frame.empty() && frame.data != nullptr && frame.cols > 0 && frame.rows > 0) {
        if(frame.type() != CV_8UC3)
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        if(current_size.width() == current_size.height())
            frame = fisheyeTransform(frame);


        QImage image(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
        qt_frame = image.copy();  // ????????????????
        emit frameReady(qt_frame);
    }
    else
        qWarning() << "SubsectionWidget" << id << "::updateFrame | Invalid frame: empty or corrup cv::Mat";
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
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->resize(1440, 800);
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
                        subsections[k]->resize(this->width() / 4, this->height() / 2);
                        left_layout->addWidget(subsections[k], k/2, k%2);
                        subsections[k]->setFullScreenMode(false, QSize(this->width() / 3, this->height() / 2));
                        subsections[k]->show();
                    }
                    fullscreen_widget = nullptr;
                }
                else{
                    fullscreen_widget = clicked_widget;
                    for(int i = 0; i < subsections.size(); i++){
                        subsections[i]->hide();
                    }
                    clicked_widget->resize(this->width() * 2 / 3, this->height());
                    left_layout->addWidget(clicked_widget, 0, 0, 2, 2);
                    clicked_widget->setFullScreenMode(true, QSize(this->width() * 2 / 3, this->height()));
                    clicked_widget->show();
                }
            });
            connect(widget, &SubsectionWidget::selectionChanged, this, [this](){
                QSet<QString> used_options;
                qDebug() << "before for";
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
                qDebug() << "after for";
                emit selectionChanged(cam_map);
            });
            connect(widget, &SubsectionWidget::resolutionChanged, this, [this, i, j](int id, int width, int height){ emit resolutionChanged(id, width, height); });
            //connect(widget, &SubsectionWidget::destructorCalled, this, [this](int id){ emit destructorCalled(id); });
            subsections.push_back(widget);
            left_layout->addWidget(widget, i, j);
        }
    }
    connect(this, &MainWindow::windowResized, this, [this](QSize new_size, QSize old_size){
        model->resize(new_size.width() / 3, new_size.height() / 2);
        model->resizeLocal(new_size.width() / 3, new_size.height() / 2);
        dashboard_container->resize(new_size.width() / 3, new_size.height() / 2);
        if(fullscreen_widget){
            fullscreen_widget->resize(new_size.width() * 2 / 3, new_size.height());
            fullscreen_widget->resizeLocal(new_size.width() * 2 / 3, new_size.height());
            return;
        }
        for(int i = 0; i < subsections.size(); i++){
            subsections[i]->resize(new_size.width() / 3, new_size.height() / 2);
            subsections[i]->resizeLocal(new_size.width() / 3, new_size.height() / 2);
        }
    });
    gas_label = new QLabel("No data");
    gas_label->setAlignment({Qt::AlignHCenter, Qt::AlignVCenter});
    gas_label->setStyleSheet(R"(
        QLabel {
            color: white;
            font-size: 14px;
            padding: 15px;
            border: 1.5px solid gray;
            font-family: Consolas;
        }
    )");
    speech_label = new QLabel("No data");
    speech_label->setAlignment({Qt::AlignHCenter, Qt::AlignVCenter});
    speech_label->setStyleSheet(R"(
        QLabel {
            color: white;
            font-size: 14px;
            padding: 15px;
            border: 1.5px solid gray;
            font-family: Consolas;
        }
    )");
    speech_label->setWordWrap(true);
    magnetometer_label = new QLabel("No data");
    magnetometer_label->setStyleSheet(R"(
        QLabel {
            color: white;
            font-size: 14px;
            padding: 15px;
            border: 1.5px solid gray;
            font-family: Consolas;
        }
    )");
    magnetometer_label->setAlignment({Qt::AlignHCenter, Qt::AlignVCenter});
    microphone_button = new QPushButton("Toggle audio");
    microphone_button->setCheckable(true);
    microphone_button->setStyleSheet(R"(
        QPushButton {
            font-size: 14px;
            color: black;
            border: none;
            padding: 5px;
            border-radius: 5px;
            margin: 2px;
            background-color: yellow;
        }
        QPushButton:checked {
            color: white;
            background-color: green;
        }
    )");
    clear_button = new QPushButton("Clear data");
    clear_button->setStyleSheet(R"(
        QPushButton {
            font-size: 14px;
            color: white;
            border: none;
            padding: 5px 10px 5px 10px;
            border-radius: 5px;
            margin: 2px;
            background-color: black;
        }
        QPushButton:pressed {
            background-color: gray;
        }
    )");
    estop_button = new QPushButton("E-STOP");
    estop_button->setCheckable(true);
    estop_button->setStyleSheet(R"(
        QPushButton {
            font-size: 14px;
            color: white;
            border: none;
            padding: 5px 10px 5px 10px;
            border-radius: 5px;
            margin: 2px;
            background-color: red;
        }
        QPushButton:pressed {
            background-color: orange;
        }
    )");
    restart_button = new QPushButton("Restart");
    restart_button->setCheckable(true);
    restart_button->setStyleSheet(R"(
        QPushButton {
            font-size: 14px;
            color: white;
            border: none;
            padding: 5px 10px 5px 10px;
            border-radius: 5px;
            margin: 2px;
            background-color: red;
        }
        QPushButton:pressed {
            background-color: orange;
        }
    )");
    controller_button = new QPushButton("Controller");
    controller_button->setCheckable(true);
    controller_button->setStyleSheet(R"(
        QPushButton {
            font-size: 14px;
            color: white;
            border: none;
            padding: 5px 10px 5px 10px;
            border-radius: 5px;
            margin: 2px;
            background-color: black;
        }
        QPushButton:pressed {
            background-color: gray;
        }
    )");

    settings_label_1 = new QLabel(" Max speed ");
    settings_label_2 = new QLabel(" 1600 ");
    settings_label_3 = new QLabel(" Setting ");
    settings_label_4 = new QLabel(" 5 ");
    settings_slider_1 = new QSlider(Qt::Horizontal);
    settings_slider_2 = new QSlider(Qt::Horizontal);
    settings_slider_1->setRange(1400, 2000);
    settings_slider_1->setValue(1600);
    settings_slider_1->setTickInterval(100);
    settings_slider_2->setRange(0, 10);
    settings_slider_2->setValue(5);
    settings_slider_2->setTickInterval(1);
    connect(settings_slider_1, &QSlider::valueChanged, this, [this](int value){
        settings_label_2->setText(QString(" %1 ").arg(value));
    });
    connect(settings_slider_2, &QSlider::valueChanged, this, [this](int value){
        settings_label_4->setText(QString(" %1 ").arg(value));
    });

    dashboard_container = new QWidget();
    dashboard_layout = new QGridLayout();
    settings_layout = new QGridLayout();
    button_layout = new QHBoxLayout();
    button_layout_2 = new QHBoxLayout();
    std::vector<QString> labels = {"Gas sensor: ", "Speech: ", "Magnetometer: "};
    for(int i = 0; i < labels.size(); i++){
        sensor_label = new QLabel(labels[i]);
        sensor_label->setStyleSheet(R"(
            QLabel {
                color: white;
                font-size: 14px;
                padding: 15px;
                border: 1.5px solid gray;
                font-family: Consolas;
            }
        )");
        dashboard_layout->addWidget(sensor_label, i, 0);
    }
    dashboard_layout->addWidget(gas_label, 0, 1);
    dashboard_layout->addWidget(speech_label, 1, 1);
    dashboard_layout->addWidget(magnetometer_label, 2, 1);

    button_layout->addWidget(microphone_button);
    button_layout->addWidget(clear_button);
    button_layout->addWidget(estop_button);
    //button_layout->addWidget(restart_button);
    button_layout->setContentsMargins(0, 0, 0, 0);
    button_layout->setSpacing(0);

    button_layout_2->addWidget(controller_button);
    button_layout_2->addWidget(restart_button);
    button_layout_2->setContentsMargins(0, 0, 0, 0);
    button_layout_2->setSpacing(0);

    settings_layout->addWidget(settings_label_1, 0, 0);
    settings_layout->addWidget(settings_slider_1, 0, 1);
    settings_layout->addWidget(settings_label_2, 0, 2);
    settings_layout->addWidget(settings_label_3, 1, 0);
    settings_layout->addWidget(settings_slider_2, 1, 1);
    settings_layout->addWidget(settings_label_4, 1, 2);

    connect(microphone_button, &QPushButton::clicked, this, [this](){ emit buttonChanged(microphone_button->isChecked()); });
    connect(clear_button, &QPushButton::clicked, this, [this](){
        gas_label->setText("No data");
        speech_label->setText("No data");
        magnetometer_label->setText("No data");
    });
    connect(estop_button, &QPushButton::pressed, this, [this](){ emit estopCalled(); });
    connect(restart_button, &QPushButton::clicked, this, [this](){
        QMessageBox::StandardButton reply;
        reply = QMessageBox::warning(this, "Restart relay", "<div align='center'>Are you sure you want to remotely restart the relay?<br>Connection will be forcibly closed.<br>MAKE SURE THE RELAY WAS LAUNCHED, NOT RAN</div>", QMessageBox::Yes | QMessageBox::Cancel);
        if(reply == QMessageBox::Yes)
            emit restartCalled();
    });
    connect(controller_button, &QPushButton::clicked, this, [this](){ emit controllerCalled(); });
    QVBoxLayout* temp = new QVBoxLayout();
    temp->addLayout(dashboard_layout);
    temp->addLayout(button_layout);
    temp->addLayout(button_layout_2);
    temp->addLayout(settings_layout);
    dashboard_container->setLayout(temp);
    dashboard_container->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    //dashboard_container->resize(480, 200);

    // 3D MODEL VIEWER
    model = new ModelWidget(this);
    right_layout->addWidget(model);
    right_layout->addWidget(dashboard_container);

    connect(this, &MainWindow::modelUpdated, model, &ModelWidget::updateState);

    main_layout->addWidget(left_container, 2);
    main_layout->addLayout(right_layout, 1);
}

void MainWindow::setCamPorts(int num_cams, std::vector<std::string> cam_names){
    for(int i = 0; i < subsections.size(); i++){
        subsections[i]->setAvailableDevices(num_cams, cam_names);
    }
}

void MainWindow::updateFrame(int id, std::vector<unsigned char> data, bool bypass){
    cv::Mat frame, image;
    if(bypass){
        image = cv::Mat(720, 1280, CV_8UC3, data.data());
    }
    else{
        image = cv::imdecode(data, cv::IMREAD_COLOR);
    }
    cv::cvtColor(image, frame, cv::COLOR_BGR2RGB);
    int sub_id = -1;
    std::vector<int> sub_ids = {};
    for(auto it = cam_map.begin(); it != cam_map.end(); it++){
        if(it->second == id){
            sub_id = it->first;
            sub_ids.push_back(it->first);
        }
    }
    std::vector<float> thermal_floats;
    {
        std::lock_guard<std::mutex> lock(thermal_mutex);
        thermal_floats = thermal_data;
    }
    cv::Mat thermal = cv::Mat(8, 8, CV_32F);
    std::memcpy(thermal.data, thermal_floats.data(), thermal_floats.size()*sizeof(float));
    if(sub_id != -1){
        for(int i = 0; i < sub_ids.size(); i++){
            subsections[sub_ids[i]]->updateFrame(frame.clone(), thermal.clone(), data);
        }
    }
}

template<typename T> void MainWindow::updateDashbord(int index, T data){
    if(index == 0){
        if constexpr(std::is_same_v<T, int>)
            gas_label->setText(QString("%1 ppm").arg(data));
    }
    else if(index == 1){
        if constexpr(std::is_same_v<T, QString>){
            //speech_label->setText(QString("%1 %2").arg(speech_label->text()).arg(data));
            speech_label->setText(data);
        }
    }
    else if(index == 2){
        if constexpr(std::is_same_v<T, QVector3D>)
            magnetometer_label->setText(QString("X: %1\nY: %2\nZ: %3").arg(data.x(), 0, 'f', 2).arg(data.y(), 0, 'f', 2).arg(data.z(), 0, 'f', 2));
    }
    else
        qCritical() << "MainWindow::updateDashboard | Invalid data index: out of bounds";
}

// SIGNAL INTERCEPT
void MainWindow::closeEvent(QCloseEvent* event) {
    emit windowClosing();

    qInfo() << "Closing main window...";

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for(int i = 0; i < subsections.size(); i++){
        subsections[i]->destroy();
    }

    model->destroy();
    event->accept();

    qInfo() << "Bye";
}

void MainWindow::resizeEvent(QResizeEvent* event){
    QWidget::resizeEvent(event);
    emit windowResized(event->size(), event->oldSize());
}

void MainWindow::updateState(std::vector<float> data){
    BasePacket model_state;
    std::memcpy(&model_state, data.data(), sizeof(BasePacket));
    updateDashbord(0, (int)model_state.gas_ppm);
    updateDashbord(2, QVector3D(model_state.magnetometer_x, model_state.magnetometer_y, model_state.magnetometer_z));
    emit modelUpdated(model_state);
}

void MainWindow::updateThermal(std::vector<float> data){
    std::lock_guard<std::mutex> lock(thermal_mutex);
    thermal_data = data;
}

// --- ROTAS stream handler ---
RTPStreamHandler::RTPStreamHandler(int port, std::string address, PayloadType type, QObject *parent) : QObject(parent){
    stream = new Stream;
    stream->ssrc = 0;
    stream->seq_num = 0 & 0xFFFF;
    stream->timestamp = 0;
    stream->payload_type = type;
    stream->port = port;
    stream->address = address;

    // --- UDP Socket init ---
    // -- send --
    send_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    send_socket_address.sin_family = AF_INET;
    send_socket_address.sin_port = htons(port + 1);
    inet_pton(AF_INET, stream->address.c_str(), &send_socket_address.sin_addr);
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
    shutdown(recv_socket, SHUT_RDWR);
    close(send_socket);
    close(recv_socket);
}

template <typename T> void RTPStreamHandler::sendPacket(std::vector<T> data, int marker, int delay){
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

        if(sendto(send_socket, (const char*)packet.data(), packet.size(), 0, (struct sockaddr*)&send_socket_address, socket_address_size) < 0){
            int error = errno;
            qWarning() << "RTPStreamHandler::sendPacket | Socket error: " << error << " " << strerror(error);
        }
        if(delay != 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }
}

void RTPStreamHandler::recvPacket(){
    std::vector<std::vector<char>> fragments;
    std::vector<char> packet, buffer(MAX_PACKET_SIZE);
    int i = 0, num_fragments = -1, ssrc = -1;
    do{
        int bytes_received = recvfrom(recv_socket, buffer.data(), MAX_PACKET_SIZE, 0, (struct sockaddr*)&recv_socket_address, &socket_address_size);
        if(bytes_received < 0){
            int error = errno;
            if(error != EAGAIN && error != EWOULDBLOCK)
                qCritical() << "RTPStreamHandler::recvPacket | Socket error: " << error << " " << strerror(error);
            return;
        }
        else if(bytes_received < sizeof(RTPHeader)) {
            if(bytes_received != 0)
                qCritical() << "RTPStreamHandler::recvPacket | Invalid RTP header: incomplete packet, size: " << bytes_received;
            return;
        }

        RTPHeader* header = new RTPHeader;
        std::memcpy(header, buffer.data(), sizeof(RTPHeader));
        packet.resize(bytes_received - sizeof(RTPHeader));
        std::memcpy(packet.data(), buffer.data() + sizeof(RTPHeader), bytes_received - sizeof(RTPHeader));

        if((header->seq & FRAGMENTATION_FLAG) == 0){
            ssrc = header->ssrc;
            break;
        }
        else if(i == 0){
            num_fragments = header->m;
            ssrc = header->ssrc;
            fragments.resize(num_fragments);
        }
        else if(ssrc != header->ssrc){
            qWarning() << "RTPStreamHandler::recvPacket | Fragmentation error: different ssrc, previous packet dropped";
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

    std::vector<uchar> data(packet.size());
    std::memcpy(data.data(), packet.data(), packet.size());
    /*
    if(stream->payload_type == PayloadType::ROS2_ARRAY && floatCallback){
        std::vector<float> data(packet.size() / sizeof(float));
        std::memcpy(data.data(), packet.data(), packet.size());
        floatCallback(data);
    }
    else if((stream->payload_type == PayloadType::VIDEO_MJPEG || stream->payload_type == PayloadType::AUDIO_PCM) && ucharCallback){
        ucharCallback(data);
    }*/

    if(ucharCallback)
        ucharCallback(data);
}

// --- Universal ---
AppHandler::AppHandler(int port, QObject* parent) : QObject(parent){
    qInfo() << "Starting GUI...";

    window = new MainWindow;
    window->setWindowTitle("GUI - beta");
    window->resize(1440, 800);
    QObject::connect(window, &MainWindow::windowClosing, [this](){ this->destroy(); });
    this->port = port;

    qInfo() << "Initializing base channel...";

    base_channel = new SocketStruct;
    base_channel->target_socket = new RTPStreamHandler(port, CLIENT_IP, PayloadType::ROS2_ARRAY);
    base_channel->target_socket->setUCharCallback([this](std::vector<uchar> data){
        if(data.size() < sizeof(BasePacket)+sizeof(float)){
            qCritical() << "AppHandler::baseUCharCallback | Invalid payload: incomplete payload, size: " << data.size();
            return;
        }
        std::vector<float> float_data((sizeof(BasePacket)/sizeof(float))+1), thermal(64);
        std::vector<std::string> string_data;
        int offset = sizeof(BasePacket) + sizeof(float);
        std::memcpy(float_data.data(), data.data(), offset);
        for(int i = 0; i < static_cast<int>(float_data[0]); i++){
            int str_size;
            std::memcpy(&str_size, data.data()+offset, sizeof(int));
            std::vector<char> str(str_size);
            std::memcpy(str.data(), data.data()+offset+sizeof(int), str_size);
            string_data.push_back(std::string(str.begin(), str.end()));
            offset += sizeof(int) + str_size;
        }
        std::memcpy(thermal.data(), data.data()+offset, 64*sizeof(float));
        {
            std::lock_guard<std::mutex> lock(base_channel->data_mutex);
            base_channel->float_data = float_data;
            base_channel->string_data = string_data;
        }
        window->updateState(std::vector<float>(float_data.begin()+1, float_data.end()));
        window->updateThermal(thermal);
    });
    base_channel->is_recv_running.store(true);
    base_channel->is_send_running.store(true);

    qInfo() << "Initializing PortAudio (stderr silenced)...";

    int stderr_backup = -1;
    int dev_null = -1;
    fflush(stderr);
    stderr_backup = dup(STDERR_FILENO);
    dev_null = open("/dev/null", O_WRONLY);
    if(dev_null != -1 && stderr_backup != -1){
        dup2(dev_null, STDERR_FILENO);
        close(dev_null);
    }
    Pa_Initialize();
    Pa_OpenDefaultStream(&stream, 0, 1, paInt16, SAMPLE_RATE, AUDIO_BUFFER_SIZE, nullptr, nullptr);
    if(stderr_backup != -1) {
        fflush(stderr);
        dup2(stderr_backup, STDERR_FILENO);
        close(stderr_backup);
    }

    qInfo() << "Initializing audio channel...";

    audio_channel = new SocketStruct;
    audio_channel->target_socket = new RTPStreamHandler(port + 2, CLIENT_IP, PayloadType::AUDIO_PCM);
    audio_channel->target_socket->setUCharCallback([this](std::vector<uchar> data){
        std::vector<opus_int16> output(AUDIO_BUFFER_SIZE);
        int frames = opus_decode(opus_decoder, data.data(), data.size(), output.data(), output.size(), 0);
        {
            std::lock_guard<std::mutex> lock(vosk_mutex);
            audio_queue.push(output);
            while(audio_queue.size() > 50) {
                audio_queue.pop();
            }
        }
        Pa_WriteStream(stream, output.data(), frames);
    });
    opus_decoder = opus_decoder_create(SAMPLE_RATE, 1, &pa_error);

    qInfo() << "Initializing controller channel...";

    controller_channel = new SocketStruct;
    controller_channel->target_socket = new RTPStreamHandler(port + 4, CLIENT_IP, PayloadType::ROS2_ARRAY);
    controller_channel->is_recv_running.store(false);
    controller_channel->is_send_running.store(true);

    qInfo() << "Initializing vosk model...";

    // - remove the first two "/" to skip vosk init -
    /*
    vosk_set_log_level(-1);
    vosk_model = vosk_model_new(VOSK_MODEL_PATH);
    vocab_json = nlohmann::json(VOSK_VOCAB).dump();
    //vosk_recognizer = vosk_recognizer_new_grm(vosk_model, SAMPLE_RATE, vocab_json.c_str());
    //*/
    // ??????????????????????
    //vosk_recognizer = vosk_recognizer_new(vosk_model, SAMPLE_RATE);

    connect(window, &MainWindow::buttonChanged, this, [this](bool is_pressed){ is_audio_active.store(is_pressed); });
    qRegisterMetaType<std::map<int, int>>("std::map<int,int>");
    connect(window, &MainWindow::selectionChanged, this, [this](std::map<int,int> cam_map){
        for(int i = 0; i < video_channels.size(); i++){
            video_channels[i]->is_active.store(false);
        }
        for(int i = 0; i < cams.size(); i++){
            cams[i]->is_active.store(false);
        }
        for(auto it = cam_map.begin(); it != cam_map.end(); it++){
            if(it->second >= 0 && it->second < video_channels.size())
                video_channels[it->second]->is_active.store(true);
            else if(it->second >= 0)
                cams[it->second-num_cams]->is_active.store(true);
        }
    });

    connect(window, &MainWindow::estopCalled, this, [this](){
        qInfo() << "E-Stop called";
        base_channel->target_socket->sendPacket(std::vector<int>{0, -1});
    });
    connect(window, &MainWindow::controllerCalled, this, [this](){
        qInfo() << "Controller restart called";
        controller_channel->is_active.store(false);
        controller->destroy();
        controller = new Controller(1000);
        controller_channel->is_active.store(true);
    });
    connect(window, &MainWindow::resolutionChanged, this, [this](int id, int width, int height){
        std::lock_guard<std::mutex> lock(video_mutex);
        video_channels[id]->int_data = std::vector<int>{width, height};
    });
    connect(window, &MainWindow::restartCalled, this, [this](){
        qInfo() << "Relay restart called";
        base_channel->target_socket->sendPacket(std::vector<int>{0, -2});
    });

    qInfo() << "Setup complete";
}

AppHandler::~AppHandler(){
    qInfo() << "Closing program...";

    base_channel->target_socket->sendPacket(std::vector<int>{0, 1});
    base_channel->is_recv_running.store(false);
    base_channel->is_send_running.store(false);
    base_channel->target_socket->destroy();
    if(base_channel->send_thread.joinable())
        base_channel->send_thread.join();
    if(base_channel->recv_thread.joinable())
        base_channel->recv_thread.join();

    qInfo() << "Base channel closed";

    is_audio_active.store(false);
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

    if(vosk_thread.joinable())
        vosk_thread.join();
    if(vosk_recognizer)
        vosk_recognizer_free(vosk_recognizer);
    if(vosk_model)
        vosk_model_free(vosk_model);

    for(int i = 0; i < cams.size(); i++){
        cams[i]->is_active.store(false);
        cams[i]->is_send_running.store(false);
        if(cams[i]->cam_thread.joinable())
            cams[i]->cam_thread.join();
        cams[i]->cap->release();
    }

    qInfo() << "Vosk model closed";

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
    num_cams = 0;
    local_cams = 0;
    local_cam_ports = {2};

    for(int i = 0; i < local_cams; i++){
        CamStruct* cam = new CamStruct;
        cam->cap = new cv::VideoCapture(local_cam_ports[i], cv::CAP_V4L2);
        cam->cap->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
        cam->cap->set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cam->cap->set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cams.push_back(std::move(cam));
        cams[i]->is_active.store(false);
        cams[i]->is_send_running.store(true);
        cams[i]->cam_thread = std::thread([i, this](){
            while(cams[i]->is_send_running.load()){
                if(!cams[i]->is_active.load()){
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                cv::Mat frame;
                (*cams[i]->cap) >> frame;
                //frame = undistortFisheyeApprox(frame);
                std::vector<unsigned char> buffer(frame.data, frame.data + frame.total()*frame.elemSize());
                window->updateFrame(i+num_cams, buffer, true);
                std::this_thread::sleep_for(std::chrono::milliseconds(33));
            }
        });
    }

    qInfo() << "Awaiting relay connection...";

    std::vector<std::string> cam_names;

    // - remove the first two "/" for connectionless debugging -
    ///*
    base_channel->target_socket->recvPacket();
    base_channel->target_socket->sendPacket(std::vector<int>{0, 0});
    {
        std::lock_guard<std::mutex> lock(base_channel->data_mutex);
        if(base_channel->float_data.empty() || base_channel->float_data[0] < 0){
            qCritical() << "AppHandler::baseSendPacket | Handshake failed: missing data or invalid marker";
            return;
        }
        num_cams = (int)base_channel->float_data[0];
        cam_names = base_channel->string_data;
    }
    for(int i = 0; i < local_cams; i++){
        cam_names.push_back("Local " + std::to_string(i+1));
    }
    window->setCamPorts(num_cams+local_cams, cam_names);
    //*/
    //window->setCamPorts(num_cams+local_cams, {"Local 1"});

    qInfo() << "Connection established. Received " << num_cams << " video sources and using " << local_cams << " locals\nStarting base channel...";

    base_channel->recv_thread = std::thread([this](){
        while(base_channel->is_recv_running.load()){
            base_channel->target_socket->recvPacket();
        }
    });

    qInfo() << "Starting video channels...";

    for(int i = 0; i < num_cams; i++){
        SocketStruct* video_socket = new SocketStruct;
        video_socket->target_socket = new RTPStreamHandler(port + (2 * i) + 6, CLIENT_IP, PayloadType::VIDEO_MJPEG);
        video_channels.push_back(std::move(video_socket));
    }
    for(int i = 0; i < video_channels.size(); i++){
        video_channels[i]->is_active.store(false);
        video_channels[i]->is_send_running.store(true);
        video_channels[i]->is_recv_running.store(true);
        video_channels[i]->target_socket->setUCharCallback([this, i](std::vector<uchar> data){ window->updateFrame(i, data); });
        video_channels[i]->recv_thread = std::thread([i, this](){
            while(video_channels[i]->is_recv_running.load()){
                video_channels[i]->target_socket->recvPacket();
            }
        });
        video_channels[i]->send_thread = std::thread([i, this](){
            while(video_channels[i]->is_send_running.load()){
                std::vector<int> data;
                {
                    std::lock_guard<std::mutex> lock(video_mutex);
                    data = video_channels[i]->int_data;
                }
                if(data.empty())
                    video_channels[i]->target_socket->sendPacket(std::vector<int>{0, (int)video_channels[i]->is_active.load()});
                else if(data.size() == 2){
                    video_channels[i]->target_socket->sendPacket(std::vector<int>{1, data[0], data[1]});
                    std::lock_guard<std::mutex> lock(video_mutex);
                    video_channels[i]->int_data.clear();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
    }

    qInfo() << "Starting audio channel...";

    audio_channel->is_recv_running.store(true);
    audio_channel->is_send_running.store(true);
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

    qInfo() << "Starting controller channel...";

    controller = new Controller(1000);
    controller_channel->is_active.store(true);
    controller_channel->send_thread = std::thread([this](){
        while(controller_channel->is_send_running.load()){
            if(!controller_channel->is_active.load()){
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            std::vector<int> data = controller->readState();
            controller_channel->target_socket->sendPacket(data);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    qInfo() << "Starting vosk model...";

    is_audio_active.store(false);
    vosk_thread = std::thread([this](){
        while(audio_channel->is_recv_running.load()){
            if(!is_audio_active.load() || !vosk_model){
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            std::vector<opus_int16> buffer;
            {
                std::lock_guard<std::mutex> lock(vosk_mutex);
                if(!audio_queue.empty()){
                    buffer = std::move(audio_queue.front());
                    audio_queue.pop();
                }
            }
            if(!buffer.empty()){
                const char* audio_bytes = reinterpret_cast<const char*>(buffer.data());
                int audio_size = buffer.size() * sizeof(opus_int16);
                if(vosk_recognizer_accept_waveform(vosk_recognizer, audio_bytes, audio_size)){
                    const char* result = vosk_recognizer_result(vosk_recognizer);
                    std::string text = parseJSON(std::string(result), "\"text\"");
                    const std::string prefix = "the ";
                    if(text.rfind(prefix, 0) == 0)
                        text = text.substr(prefix.length());
                    if(!text.empty())
                        window->updateDashbord(1, QString::fromStdString(text));
                }
            }
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    window->show();

    qInfo() << "Program init complete";
}

// --- (Optional) console logs ---
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
