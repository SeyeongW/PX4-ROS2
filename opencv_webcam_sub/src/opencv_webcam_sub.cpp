#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp> // Header 사용을 위해 추가
#include <cv_bridge/cv_bridge.h>
#include <vector>

using namespace cv;
using std::placeholders::_1;

class WebcamSubscriber : public rclcpp::Node
{
public:
    WebcamSubscriber() : Node("opencv_webcam_sub")
    {
        // CompressedImage 구독자 생성 (/perception/debug_image/compressed)
        subscriber_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/perception/debug_image/compressed", 
            10,
            std::bind(&WebcamSubscriber::imageCallback, this, _1));

        // OpenCV 창 생성
        namedWindow("Webcam Subscriber", WINDOW_AUTOSIZE);
        RCLCPP_INFO(this->get_logger(), "Webcam Subscriber Started.");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscriber_;

    void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
    {
        try
        {
            // CompressedImage 데이터를 OpenCV Mat으로 변환 (imdecode)
            std::vector<uchar> buf(msg->data.begin(), msg->data.end());
            Mat image = imdecode(buf, IMREAD_COLOR);

            if (image.empty())
            {
                RCLCPP_WARN(this->get_logger(), "Received empty image!");
                return;
            }

            // 이미지 표시 (OpenCV GUI)
            imshow("Webcam Subscriber", image);
            
            // 키 입력 대기 (1ms). ESC(27) 누르면 종료
            if (waitKey(1) == 27) 
            {
                RCLCPP_INFO(this->get_logger(), "ESC pressed. Shutting down...");
                rclcpp::shutdown();
            }
        }
        catch (const cv::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "OpenCV Error: %s", e.what());
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WebcamSubscriber>();
    rclcpp::spin(node);
    
    // 종료 처리
    destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}