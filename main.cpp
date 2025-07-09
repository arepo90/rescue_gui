#include <QApplication>
#include "mainwindow.h"

/*
    TODO:
    - Pass settings to relay (and publish them)
    - Test thermal implementation
    - Check if launched before crash (on relay)
    - Fix frame scaling for different resolutions
*/

// Qt3D testing
/*
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // Create main 3D window
    Qt3DExtras::Qt3DWindow view;

    // Root entity
    Qt3DCore::QEntity *rootEntity = new Qt3DCore::QEntity;

    // Camera
    Qt3DRender::QCamera *camera = view.camera();
    camera->lens()->setPerspectiveProjection(45.0f, 16.0f/9.0f, 0.1f, 1000.0f);
    camera->setPosition(QVector3D(0, 0, 5));
    camera->setViewCenter(QVector3D(0, 0, 0));

    // For camera controls
    Qt3DExtras::QOrbitCameraController *camController = new Qt3DExtras::QOrbitCameraController(rootEntity);
    camController->setLinearSpeed(50.0f);
    camController->setLookSpeed(180.0f);
    camController->setCamera(camera);

    // Create a mesh component
    Qt3DRender::QMesh *mesh = new Qt3DRender::QMesh(rootEntity);
    // Replace with your OBJ file path
    //mesh->setSource(QUrl::fromLocalFile("../../assets/body_nobands.obj"));
    mesh->setSource(QUrl::fromLocalFile(mesh_addresses[0]));


    // Create material component
    Qt3DExtras::QPhongMaterial *material = new Qt3DExtras::QPhongMaterial(rootEntity);
    material->setDiffuse(QColor(QRgb(0x665423)));

    // Create transform component
    Qt3DCore::QTransform *transform = new Qt3DCore::QTransform;
    transform->setScale(1.0f);

    // Create entity and add components
    Qt3DCore::QEntity *modelEntity = new Qt3DCore::QEntity(rootEntity);
    modelEntity->addComponent(mesh);
    modelEntity->addComponent(material);
    modelEntity->addComponent(transform);

    // Set root entity of the scene
    view.setRootEntity(rootEntity);

    // Show window
    view.show();

    return app.exec();
}
*/

// Circle detection, may not be necessary
/*
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
    int rad_checks = 16;

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
*/

// Original shape detection
/*
cv::Mat SubsectionWidget::Filters::detectShape(cv::Mat frame, int corner){
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
    cv::Mat final_roi = frame_roi.clone(), final_thresh;

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
*/

cv::Mat placeText(std::string text, cv::Mat frame){
    int temp = 0;
    cv::Size text_size = cv::getTextSize(text,  cv::FONT_HERSHEY_SIMPLEX, 3, 3, &temp);
    cv::putText(frame, text, cv::Point((frame.cols - text_size.width)/2, (frame.rows + text_size.height)/2),  cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255, 0, 0), 3);
    return frame;
}

// std::vector<QString> shape_buttons{ "Upper left", "Upper right", "Lower left", "Lower right" };
cv::Mat detectShape(cv::Mat frame, int corner = 0, bool mode = false){
    double scale = 1.0, min_dis = DBL_MAX;
    cv::Mat gray_frame, gray_resized, inv_thresh, inv_task_sector, task_sector;
    std::vector<cv::Vec3f> circles;
    std::vector<std::vector<cv::Point>> contours, shapes;
    cv::Rect sector;

    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    cv::resize(gray_frame, gray_resized, cv::Size(), 1.0/scale, 1.0/scale, cv::INTER_AREA);
    cv::threshold(gray_resized, inv_thresh, 50, 255, cv::THRESH_BINARY_INV);

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
            }
        }
        inv_task_sector = inv_thresh(sector);
    }

    cv::bitwise_not(inv_task_sector, task_sector);
    cv::findContours(task_sector, shapes, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
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
        if(aspect_ratio < 0.75 || aspect_ratio > 1.25 || solidity < 0.75) continue;
        cv::Rect contour = cv::boundingRect(shapes[i]);
        cv::Point shape_center(contour.x + contour.width/2, contour.y + contour.height/2);
        double dis = cv::norm(shape_center - sector_center);
        if(dis < min_dis){
            min_dis = dis;
            shape = shapes[i];
        }
    }
    if(!shape.empty()){
        cv::Rect box = cv::boundingRect(shape), final;
        final.x = box.x + sector.x - 10;
        final.y = box.y + sector.y - 10;
        final.width = box.width + 20;
        final.height = box.height + 20;

        cv::rectangle(frame, final, cv::Scalar(0, 255, 0), 5);
    }
    return frame;
}

#include <netdb.h>

int main(int argc, char* argv[]){
    QApplication app(argc, argv);

#ifdef _WIN32
    qInfo() << "Hi Windows";
    WSAData wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
#else
    qInfo() << "--- Hi Linux ---";
#endif

/*
    std::string hostname = "robotec-rescue-jetson.local", port = "12345";
    struct addrinfo hints = {}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    int status = getaddrinfo(hostname.c_str(), port.c_str(), &hints, &res);
    if(status != 0){
        qDebug() << "getaddrinfo error: " << gai_strerror(status);
    }
    qDebug() << "got: " << res->ai_addr << " " << res->ai_addrlen;
    */

    AppHandler* app_handler = new AppHandler(8000);
    app_handler->init();

    return app.exec();
}
