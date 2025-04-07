#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "nofree/bag_merge.h"
#include "util/cereal_archive_helper.hpp"
#include "spdlog/fmt/bundled/color.h"
#include "spdlog/spdlog.h"
#include "util/status.hpp"
#include "util/utils_tpl.hpp"
#include "filesystem"
#include "util/tqdm.h"

typedef struct {
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<std::vector<cv::Point3f>> object_points;
    cv::Size image_size;
} cv_calib_struct;


std::vector<std::string> splitTopics(const std::string& str) {
    std::vector<std::string> result;
    std::istringstream iss(str);
    std::string token;
    while (iss >> token) {
        result.push_back(token);
    }
    return result;
}

void writeCameraIntrinsicsYaml(
    const std::string& filename,
    const cv::Size& image_size,
    const cv::Mat& camera_matrix,  // 3x3
    const cv::Mat& dist_coeffs     // n x 1 or 1 x n
) {
    YAML::Emitter out;
    out << YAML::BeginMap;

    out << YAML::Key << "Intrinsics" << YAML::Value << YAML::BeginMap;

    out << YAML::Comment("polymorphic pointers point to a base class and also point to objects of derived classes");
    out << YAML::Comment("don't care about this field");
    out << YAML::Key << "polymorphic_id" << YAML::Value << 2147483649;

    out << YAML::Comment("the camera model type, supported models are:");
    out << YAML::Comment("(1) pinhole_brown_t2: k1, k2, k3, t1, t2");
    out << YAML::Comment("(2)  pinhole_fisheye: k1, k2, k3, k4");
    out << YAML::Key << "polymorphic_name" << YAML::Value << "pinhole_brown_t2";

    out << YAML::Key << "ptr_wrapper" << YAML::Value << YAML::BeginMap;
    out << YAML::Comment("don't care about this field");
    out << YAML::Key << "id" << YAML::Value << 2147483649;
    out << YAML::Key << "data" << YAML::Value << YAML::BeginMap;

    // img_width and height
    out << YAML::Key << "img_width" << YAML::Value << image_size.width;
    out << YAML::Key << "img_height" << YAML::Value << image_size.height;

    // Focal length (fx, fy)
    out << YAML::Comment("fx, fy");
    out << YAML::Comment("we have to admit that these intrinsic parameters are poorly calibrated due to our");
    out << YAML::Comment("careless operation, however, these intrinsics would be refined in ikalibr");
    out << YAML::Key << "focal_length" << YAML::Value <<  YAML::BeginSeq
        << camera_matrix.at<double>(0, 0) << camera_matrix.at<double>(1, 1)
        << YAML::EndSeq;

    // Principal point (cx, cy)
    out << YAML::Comment("cx, cy");
    out << YAML::Key << "principal_point" << YAML::Value << YAML::BeginSeq
        << camera_matrix.at<double>(0, 2) << camera_matrix.at<double>(1, 2)
        << YAML::EndSeq;

    // Distortion parameters
    out << YAML::Key << "disto_param" << YAML::Value << YAML::BeginSeq;
    for (int i = 0; i < dist_coeffs.cols * dist_coeffs.rows; ++i) {
        out << dist_coeffs.at<double>(i);
    }
    out << YAML::EndSeq;

    // Close all nested maps
    out << YAML::EndMap; // data
    out << YAML::EndMap; // ptr_wrapper
    out << YAML::EndMap; // Intrinsics
    out << YAML::EndMap; // root

    // Write to file
    std::ofstream fout(filename);
    fout << out.c_str();
    fout.close();
}


void calibrateCameraForTopic(const std::string& topic, const std::vector<std::vector<cv::Point2f>>& image_points,
    const std::vector<std::vector<cv::Point3f>>& object_points, const cv::Size& image_size, const std::string output_path) {
    if (image_points.size() < 10) {
        ROS_WARN("Not enough valid chessboard images for topic: %s", topic.c_str());
        return;
    }

    cv::Mat camera_matrix, dist_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double error = cv::calibrateCamera(object_points, image_points, image_size, camera_matrix, dist_coeffs, rvecs, tvecs);

    // 输出 YAML 文件
    std::string file_name_topic = topic;
    file_name_topic.erase(0,1);
    std::replace(file_name_topic.begin(), file_name_topic.end(), '/', '-');
    std::string filename = output_path + file_name_topic + "-intri.yaml"; // 去掉 `/` 避免文件名问题
    writeCameraIntrinsicsYaml(filename, image_size, camera_matrix, dist_coeffs);
    ROS_INFO("Calibration for %s saved to %s", topic.c_str(), filename.c_str());
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ikalibr_camera_intri_calib");

    try {
        ns_ikalibr::ConfigSpdlog();
        
        ns_ikalibr::PrintIKalibrLibInfo();

        // load parameters
        std::string iBagPath = ns_ikalibr::GetParamFromROS<std::string>(
            "/ikalibr_camera_intri_calib/input_bag_path"
        );

        std::string output_path = ns_ikalibr::GetParamFromROS<std::string>(
            "/ikalibr_camera_intri_calib/output_path"
        );

        std::string camera_topics = ns_ikalibr::GetParamFromROS<std::string>(
            "/ikalibr_camera_intri_calib/camera_topics"
        );
        std::vector<std::string> camera_topics_vec = splitTopics(camera_topics);

        std::string target_type = ns_ikalibr::GetParamFromROS<std::string>(
            "/ikalibr_camera_intri_calib/target_type"
        );

        int target_cols = ns_ikalibr::GetParamFromROS<int>(
            "/ikalibr_camera_intri_calib/targetCols"
        );
        target_cols--;
        int target_rows = ns_ikalibr::GetParamFromROS<int>(
            "/ikalibr_camera_intri_calib/targetRows"
        );
        target_rows--;
        int square_size = ns_ikalibr::GetParamFromROS<int>(
            "/ikalibr_camera_intri_calib/squareSize"
        );

    
        if (!std::filesystem::exists(iBagPath)) {
            throw ns_ikalibr::Status(ns_ikalibr::Status::ERROR, "the bag path not exists!!! '{}'",
                                        iBagPath);
        } else {
            spdlog::info("the path of rosbag: '{}'", iBagPath);
        }
        
        std::map<std::string, cv_calib_struct> cv_calib_struct_map;

        std::vector<cv::Point3f> objp;
    
        for (int i = 0; i < target_rows; i++) {
            for (int j = 0; j < target_cols; j++) {
                objp.push_back(cv::Point3f(j, i, 0));
            }
        }
    
        rosbag::Bag bag;
        bag.open(iBagPath, rosbag::bagmode::Read);
        rosbag::View view(bag, rosbag::TopicQuery(camera_topics_vec));
    
        spdlog::info("Extracting chessboardcorners...");
        auto extract_bar = std::make_shared<tqdm>();
        int bar_cnt = 0;
        int failed_cnt = 0;
        for (const rosbag::MessageInstance& m : view) {
            extract_bar->progress(bar_cnt++, view.size());
            sensor_msgs::Image::ConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
            if (!img_msg) continue;
    
            std::string topic = m.getTopic();
            cv::Mat image;
            cv::Mat gray;
            try {
                image = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            } catch (cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                continue;
            }
    
            cv_calib_struct_map[topic].image_size = image.size();
    
            std::vector<cv::Point2f> corners;
            bool found = cv::findChessboardCorners(gray, cv::Size(target_cols, target_rows), corners, 
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
            if (found) {
                cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));

                cv_calib_struct_map[topic].image_points.push_back(corners);
                cv_calib_struct_map[topic].object_points.push_back(objp);

                cv::drawChessboardCorners(image, cv::Size(target_cols, target_rows), corners, found);
            } else {
                failed_cnt++;
            }
        }
        extract_bar->finish();
        bag.close();

        std::string extract_rate = std::to_string(view.size()-failed_cnt) + "/" + std::to_string(view.size());
        spdlog::info("Extracting finished. [{}]", extract_rate);
    
        spdlog::info("Calibrating cameras...");
        auto calib_bar = std::make_shared<tqdm>();
        bar_cnt = 0;
        for (const auto& entry : cv_calib_struct_map) {
            calib_bar->progress(bar_cnt++, cv_calib_struct_map.size());
            calibrateCameraForTopic(entry.first, 
                                    entry.second.image_points,
                                    entry.second.object_points, 
                                    entry.second.image_size, 
                                    output_path
                                );
        }
        calib_bar->finish();
    } catch (const ns_ikalibr::IKalibrStatus &status) {
        // if error happened, print it
        static const auto FStyle = fmt::emphasis::italic | fmt::fg(fmt::color::green);
        static const auto WECStyle = fmt::emphasis::italic | fmt::fg(fmt::color::red);
        switch (status.flag) {
            case ns_ikalibr::Status::FINE:
                // this case usually won't happen
                spdlog::info(fmt::format(FStyle, "{}", status.what));
                break;
            case ns_ikalibr::Status::WARNING:
                spdlog::warn(fmt::format(WECStyle, "{}", status.what));
                break;
            case ns_ikalibr::Status::ERROR:
                spdlog::error(fmt::format(WECStyle, "{}", status.what));
                break;
            case ns_ikalibr::Status::CRITICAL:
                spdlog::critical(fmt::format(WECStyle, "{}", status.what));
                break;
        }
    } catch (const std::exception &e) {
        // an unknown exception not thrown by this program
        static const auto WECStyle = fmt::emphasis::italic | fmt::fg(fmt::color::red);
        spdlog::critical(fmt::format(WECStyle, "unknown error happened: '{}'", e.what()));
    }

    ros::shutdown();
    return 0;
}