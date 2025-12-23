#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vector>

using namespace cv;
using namespace std::chrono_literals;

int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("opencv_webcam_pub");

    // Publisher 생성 (/image_raw/compressed)
    auto publisher = node->create_publisher<sensor_msgs::msg::CompressedImage>("/image_raw/compressed", 10);

    // 카메라 열기 (V4L2)
    VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        RCLCPP_ERROR(node->get_logger(), "Failed to open the camera.");
        return 1;
    }
 
    // 카메라 설정
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30); 

    RCLCPP_INFO(node->get_logger(), "Webcam Publisher Started!");

    while (rclcpp::ok())
    {
        Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        // JPEG 압축
        std::vector<uchar> buf;
        std::vector<int> params = {IMWRITE_JPEG_QUALITY, 80};
        imencode(".jpg", frame, buf, params);

        // 메시지 생성
        auto msg = std::make_shared<sensor_msgs::msg::CompressedImage>();
        msg->format = "jpeg";
        msg->data = std::move(buf);
        msg->header.stamp = node->now();
        msg->header.frame_id = "camera_link";

        publisher->publish(*msg);

        rclcpp::sleep_for(33ms);
    }

    cap.release();
    rclcpp::shutdown();
    return 0;
}