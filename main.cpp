/*
    ROBOTEC 2025

    DISCLAIMER:
    APP WILL ONLY RUN WITH THE FOLLOWING DEPENDENCIES:
    - QT6 + WIDGETS + 3D + RANDOM STUFF I DONT REMEMBER
    - OPENCV (binaries)
    - PORTAUDIO (vcpkg)
    - OPUS (vcpkg)
    - XINPUT (windows sdk)

    MAY OR MAY NOT WORK WITHOUT THEM (you can comment them in CMakeLists.txt and code lines themselves)
*/

#include "mainwindow.h"
#include <QApplication>
#include <cstdio>

// --- tests n stuff ---
/*
#define AUDIO_PORT 8001
#define VIDEO_PORT 8002
#define SAMPLE_RATE2 48000
#define CHANNELS 1
#define FRAME_SIZE 960

SOCKET audioSocket, videoSocket;
//struct sockaddr_in serverAddr;
sockaddr_in audioAddr, videoAddr;
OpusDecoder* opusDecoder;
std::vector<opus_int16> audioBuffer(FRAME_SIZE);
// Audio callback
int audioCallback(const void* input, void* output, unsigned long frameCount, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {
    qDebug() << "audio start";
    SOCKET* sock = static_cast<SOCKET*>(userData);
    unsigned char recvBuffer[4000];
    int recvLen = recvfrom(*sock, (char*)recvBuffer, sizeof(recvBuffer), 0, NULL, NULL);
    if (recvLen > sizeof(uint32_t)) {
        opus_decode(opusDecoder, recvBuffer, recvLen, (opus_int16*)output, FRAME_SIZE, 0);
    }
    qDebug() << "audio end";
    return paContinue;
}

void videoStream() {
    videoSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    videoAddr.sin_family = AF_INET;
    videoAddr.sin_port = htons(VIDEO_PORT);
    videoAddr.sin_addr.s_addr = INADDR_ANY;
    if(::bind(videoSocket, (struct sockaddr*)&videoAddr, sizeof(videoAddr)) == SOCKET_ERROR) {
        qDebug() << "Failed to bind video socket: " << WSAGetLastError();
    }
    qDebug() << "start video thread";
    cv::Mat frame = cv::imread("../../assets/404.png", 0);
    window->updateFrame(frame);
    frame.release();
    uint32_t expectedSeqNum = 0;
    sockaddr_in senderAddr;
    int senderAddrSize = sizeof(senderAddr);
    while (true) {
        qDebug() << "video start";
        //unsigned char recvBuffer[65536];
        std::vector<uchar> recvBuffer(65536);
        int recvLen = recvfrom(videoSocket, (char*)recvBuffer.data(), recvBuffer.size(), 0, (sockaddr*)&senderAddr, &senderAddrSize);
        if(recvLen == SOCKET_ERROR) {
            int error = WSAGetLastError();
            qDebug() << "recvfrom() failed: " << error;
            continue;
        }
        qDebug() << "video mid";
        if (recvLen > sizeof(uint32_t)) {
            qDebug() << "size: " << recvBuffer.size();
            frame = cv::imdecode(recvBuffer, cv::IMREAD_COLOR);
            if (!frame.empty()) {
                //cv::imshow("Video Stream", frame);
                //cv::waitKey(1);
                window->updateFrame(frame);
            }
            //expectedSeqNum = seqNum + 1;
        }
        qDebug() << "video end";
    }
}
*/
/*
    WSAData wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    std::thread videoThread = std::thread(videoStream);

    audioSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    audioAddr.sin_family = AF_INET;
    audioAddr.sin_port = htons(AUDIO_PORT);
    audioAddr.sin_addr.s_addr = INADDR_ANY;
    ::bind(audioSocket, (struct sockaddr*)&audioAddr, sizeof(audioAddr));

    int error;
    opusDecoder = opus_decoder_create(SAMPLE_RATE2, CHANNELS, &error);

    Pa_Initialize();
    PaStream* stream;
    Pa_OpenDefaultStream(&stream, 0, 1, paInt16, SAMPLE_RATE2, FRAME_SIZE, audioCallback, &audioSocket);
    Pa_StartStream(stream);

    QObject::connect(window, &MainWindow::windowClosing, [stream, &videoThread](){
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        opus_decoder_destroy(opusDecoder);
        closesocket(audioSocket);
        closesocket(videoSocket);
        WSACleanup();
        videoThread.join();
    });
*/

/*
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

// WORK IN PROGRESS OK ?😭😭😭😭😭
// 80 420 300 300

ConsoleWindow *console = nullptr;

/*
cv::Mat detectShape(const cv::Mat& inputFrame) {
    cv::Mat frame = inputFrame.clone();
    int frameHeight = frame.rows;
    int frameWidth = frame.cols;
    int quadrantWidth = frameWidth;
    int quadrantHeight = frameHeight;
    cv::Mat quadrantROI = inputFrame.clone();
    cv::Mat grayQuadrant;
    cv::cvtColor(quadrantROI, grayQuadrant, cv::COLOR_BGR2GRAY);
    cv::Mat threshQuadrant;
    cv::threshold(grayQuadrant, threshQuadrant, 50, 255, cv::THRESH_BINARY_INV);
    std::vector<std::vector<cv::Point>> quadrantContours;
    cv::findContours(threshQuadrant, quadrantContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Rect roi;
    bool roiFound = false;
    double dists = DBL_MAX;
    for(const auto& contour : quadrantContours) {
        double area = cv::contourArea(contour);
        if(area < 1000) continue;
        cv::Rect temp = cv::boundingRect(contour);
        double dis = std::sqrt((temp.x*temp.x)+((inputFrame.rows-temp.y)*(inputFrame.rows-temp.y)));
        if(dis < dists){
            dists = dis;
            roi = temp;
        }
    }
    int padding = 5;
    roi.x = max(0, roi.x - padding);
    roi.y = max(0, roi.y - padding);
    roi.width = min(inputFrame.cols - roi.x, roi.width + 2*padding);
    roi.height = min(inputFrame.rows - roi.y, roi.height + 2*padding);
    cv::rectangle(frame, roi, cv::Scalar(0, 255, 255), 2);
    cv::Mat roiMat = frame(roi);
    cv::Mat gray;
    cv::cvtColor(roiMat, gray, cv::COLOR_BGR2GRAY);
    cv::Mat thresh;
    cv::threshold(gray, thresh, 200, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat morphed;
    cv::morphologyEx(thresh, morphed, cv::MORPH_OPEN, kernel);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, gray.rows/8, 100, 30, gray.rows/4, gray.rows/2);
    cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1) * 255;
    for(size_t i = 0; i < circles.size(); i++) {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        cv::circle(mask, center, radius, cv::Scalar(0), 2);
    }
    cv::Mat masked;
    cv::bitwise_and(morphed, mask, masked);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(masked, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double minArea = 10.0;
    std::vector<std::vector<cv::Point>> filteredContours;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);

        if (area > minArea) {
            cv::Rect boundRect = cv::boundingRect(contour);
            double aspectRatio = (double)boundRect.width / boundRect.height;
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);
            double hullArea = cv::contourArea(hull);
            double solidity = area / hullArea;
            if (aspectRatio > 0.2 && aspectRatio < 5.0 && solidity > 0.5) {
                filteredContours.push_back(contour);
            }
        }
    }
    qDebug() << "final: " << filteredContours.size();
    if (!filteredContours.empty()) {
        cv::Point center(gray.cols/2, gray.rows/2);
        double minDistance = DBL_MAX;
        int bestContourIdx = -1;
        for (size_t i = 0; i < filteredContours.size(); i++) {
            cv::Moments m = cv::moments(filteredContours[i]);
            if (m.m00 != 0) {
                cv::Point centerOfMass(m.m10/m.m00, m.m01/m.m00);
                double distance = cv::norm(centerOfMass - center);
                if (distance < minDistance) {
                    minDistance = distance;
                    bestContourIdx = i;
                }
            }
        }
        if (bestContourIdx >= 0) {
            cv::Rect boundingBox = cv::boundingRect(filteredContours[bestContourIdx]);
            boundingBox.x += roi.x;
            boundingBox.y += roi.y;
            cv::rectangle(frame, boundingBox, cv::Scalar(0, 255, 0), 2);
        }
    }
    return frame;
}*/

/*
cv::Mat detectShapeGPU(const cv::Mat& input_frame) {
    cv::cuda::GpuMat gpu_frame(input_frame);
    cv::cuda::GpuMat gpu_gray, gpu_thresh;

    cv::cuda::cvtColor(gpu_frame, gpu_gray, cv::COLOR_BGR2GRAY);
    cv::cuda::threshold(gpu_gray, gpu_thresh, 50, 255, cv::THRESH_BINARY_INV);

    cv::Mat thresh;
    gpu_thresh.download(thresh);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Rect roi;
    double best_dist = DBL_MAX;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 1000) continue;
        cv::Rect temp = cv::boundingRect(contour);
        double dist = std::sqrt(temp.x * temp.x + std::pow(input_frame.rows - temp.y, 2));
        if (dist < best_dist) {
            best_dist = dist;
            roi = temp;
        }
    }
    int padding = 5;
    roi.x = max(0, roi.x - padding);
    roi.y = max(0, roi.y - padding);
    roi.width = min(input_frame.cols - roi.x, roi.width + 2 * padding);
    roi.height = min(input_frame.rows - roi.y, roi.height + 2 * padding);

    // Crop the ROI on GPU
    cv::cuda::GpuMat gpu_roi = gpu_frame(roi);

    cv::cuda::GpuMat gpu_gray_roi;
    cv::cuda::cvtColor(gpu_roi, gpu_gray_roi, cv::COLOR_BGR2GRAY);
    cv::cuda::GpuMat gpu_thresh_roi;
    cv::cuda::threshold(gpu_gray_roi, gpu_thresh_roi, 200, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Ptr<cv::cuda::Filter> morph_filter = cv::cuda::createMorphologyFilter(
        cv::MORPH_OPEN, gpu_thresh_roi.type(), kernel);
    cv::cuda::GpuMat gpu_morphed;
    morph_filter->apply(gpu_thresh_roi, gpu_morphed);

    // Download morphed result to CPU for HoughCircles
    cv::Mat gray_roi, morphed;
    gpu_gray_roi.download(gray_roi);
    gpu_morphed.download(morphed);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray_roi, circles, cv::HOUGH_GRADIENT, 1, gray_roi.rows / 8, 100, 30, gray_roi.rows / 4, gray_roi.rows / 2);
    cv::Mat mask = cv::Mat::ones(gray_roi.size(), CV_8UC1) * 255;
    for (const auto& circle : circles) {
        cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
        int radius = cvRound(circle[2]);
        cv::circle(mask, center, radius, cv::Scalar(0), 2);
    }
    cv::Mat masked;
    cv::bitwise_and(morphed, mask, masked);
    std::vector<std::vector<cv::Point>> filtered_contours;
    std::vector<std::vector<cv::Point>> all_contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(masked, all_contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : all_contours) {
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
        cv::Point center(gray_roi.cols / 2, gray_roi.rows / 2);
        double min_distance = DBL_MAX;
        int best_contour_idx = -1;

        for (size_t i = 0; i < filtered_contours.size(); ++i) {
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
            cv::Mat output_frame = input_frame.clone();
            cv::rectangle(output_frame, bounding_box, cv::Scalar(0, 255, 0), 2);
            return output_frame;
        }
    }
    return input_frame.clone();
}
*/

// might be the goat fr
cv::Mat detectShape(cv::Mat input_frame){
    cv::Mat frame = input_frame, gray_frame, thresh_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_frame, thresh_frame, 50, 255, cv::THRESH_BINARY_INV);
    std::vector<std::vector<cv::Point>> sector_contours;
    cv::findContours(thresh_frame, sector_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Vec3f> circles2;
    cv::HoughCircles(gray_frame, circles2, cv::HOUGH_GRADIENT, 1, gray_frame.rows/8, 100, 50, gray_frame.rows/8, gray_frame.rows/4);
    double min_dis = DBL_MAX;
    cv::Vec3f circle_sector;
    for(int i = 0; i < circles2.size(); i++){
        cv::Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
        int radius = cvRound(circles2[i][2]);
        cv::circle(input_frame, center, radius, cv::Scalar(0), 10);
        double dis = center.x*center.x + (frame.rows - center.y)*(frame.rows - center.y);
        //cv::rectangle(thresh_frame, temp, cv::Scalar(255, 0, 255), 2);
        if(dis < min_dis){
            min_dis = dis;
            circle_sector = circles2[i];
        }
    }
    cv::circle(frame, cv::Point(cvRound(circle_sector[0]), cvRound(circle_sector[1])), cvRound(circle_sector[2]), cv::Scalar(255, 0, 0), 2);
    cv::Mat mask2 = cv::Mat::zeros(frame.size(), CV_8UC1), res;
    cv::circle(mask2, cv::Point(cvRound(circle_sector[0]), cvRound(circle_sector[1])), cvRound(circle_sector[2]), cv::Scalar(255), -1);
    gray_frame.copyTo(res, mask2);

    cv::Rect bounding_box = cv::boundingRect(mask2);
    cv::Mat final = res(bounding_box);

    cv::Rect task_sector;
    min_dis = DBL_MAX;
    //for(const auto& contour : quadrant_contours) {
    for(int i = 0; i < sector_contours.size(); i++){
        double area = cv::contourArea(sector_contours[i]);
        if(area < 1000) continue;
        cv::Rect temp = cv::boundingRect(sector_contours[i]);
        double dis = temp.x*temp.x + (frame.rows - temp.y)*(frame.rows - temp.y);
        //cv::rectangle(thresh_frame, temp, cv::Scalar(255, 0, 255), 2);
        if(dis < min_dis){
            min_dis = dis;
            task_sector = temp;
        }
    }
    //cv::imshow("test", thresh_frame);
    cv::rectangle(frame, task_sector, cv::Scalar(0, 255, 255), 2);

    cv::Mat task_mat = gray_frame(task_sector);
    cv::Mat task_thresh, morphed;
    //cv::cvtColor(task_mat, task_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(task_mat, task_thresh, 200, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(task_thresh, morphed, cv::MORPH_OPEN, kernel);
    std::vector<cv::Vec3f> circles;
    //cv::HoughCircles(task_mat, circles, cv::HOUGH_GRADIENT, 1, task_mat.rows/8, 100, 50, task_mat.rows/8, task_mat.rows/3);
    //cv::Mat mask = cv::Mat::ones(task_mat.size(), CV_8UC1) * 255;
    cv::HoughCircles(final, circles, cv::HOUGH_GRADIENT, 1, final.rows/8, 100, 50, final.rows/8, final.rows/3);
    cv::Mat mask = cv::Mat::ones(final.size(), CV_8UC1) * 255;

    qDebug() << "circles: " << circles.size();
    for(int i = 0; i < circles.size(); i++){
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]) + 5;
        cv::circle(mask, center, radius, cv::Scalar(0), 10);
        //center.x += task_sector.x;
        //center.y += task_sector.y;
        center.x += bounding_box.x;
        center.y += bounding_box.y;
        cv::circle(frame, center, radius, cv::Scalar(255, 0, 255), 2);
    }

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
    cv::circle(mask_roi, cv::Point(cvRound(circles[0][0]), cvRound(circles[0][1])), cvRound(circles[0][2])-10, cv::Scalar(255), -1);
    final.copyTo(roi, mask_roi);
    cv::Rect bounding_box2 = cv::boundingRect(mask_roi);
    cv::Mat finalfinal = roi(bounding_box2);
    cv::threshold(roi, ahorasi, 200, 255, cv::THRESH_BINARY);
    cv::findContours(ahorasi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

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
                center.x += bounding_box.x;
                center.y += bounding_box.y;
                double distance = cv::norm(center_of_mass - center);
                if (distance < min_distance){
                    min_distance = distance;
                    best_contour_idx = i;
                }

                cv::Rect bounding_box3 = cv::boundingRect(filtered_contours[i]);
                bounding_box3.x += bounding_box.x - 10;
                bounding_box3.y += bounding_box.y - 10;
                bounding_box3.width += 20;
                bounding_box3.height += 20;
                cv::rectangle(frame, bounding_box3, cv::Scalar(255, 255, 0), 2);

            }
        }

        if (best_contour_idx >= 0) {
            cv::Rect bounding_box3 = cv::boundingRect(filtered_contours[best_contour_idx]);
            //bounding_box.x += task_sector.x - 10;
            //bounding_box.y += task_sector.y - 10;
            bounding_box3.x += bounding_box.x - 10;
            bounding_box3.y += bounding_box.y - 10;
            bounding_box3.width += 20;
            bounding_box3.height += 20;
            cv::rectangle(frame, bounding_box3, cv::Scalar(0, 255, 0), 2);
        }

    }
    return frame;
}


int main(int argc, char* argv[]){
    QApplication app(argc, argv);

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
    /*
    cv::Mat frame = cv::imread("../../assets/full_crate_test.png");
    //cv::Mat frame = cv::imread("../../assets/cam3.jpg");
    cv::Mat frame2= detectShape(frame);
    if(!frame2.empty())
        cv::imshow("res", frame2);
    */

    WSAData wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
    AppHandler* app_handler = new AppHandler(8000);
    app_handler->init();

    return app.exec();
}
