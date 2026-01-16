#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <vector>

using namespace cv;
using std::placeholders::_1;

class WebcamSubscriber : public rclcpp::Node
{
public:
    WebcamSubscriber() : Node("opencv_webcam_sub")
    {
        // [Key Solution] Use SensorDataQoS instead of integer '10'
        // SensorDataQoS sets reliability to 'Best Effort' automatically.
        rclcpp::SensorDataQoS qos_profile;

        subscriber_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/perception/yolo_result/compressed", 
            qos_profile, 
            std::bind(&WebcamSubscriber::imageCallback, this, _1));

        namedWindow("Webcam Subscriber", WINDOW_AUTOSIZE);
        RCLCPP_INFO(this->get_logger(), "Node initialized. Waiting for image data...");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscriber_;

    void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
    {
        // Debug log to confirm callback execution
        // RCLCPP_INFO(this->get_logger(), "Image received. Size: %zu", msg->data.size());

        try
        {
            std::vector<uchar> buf(msg->data.begin(), msg->data.end());
            Mat image = imdecode(buf, IMREAD_COLOR);

            if (image.empty())
            {
                RCLCPP_WARN(this->get_logger(), "Decoding failed: Empty image");
                return;
            }

            imshow("Webcam Subscriber", image);
            
            // waitKey is essential for OpenCV UI to update
            int key = waitKey(1);
            if (key == 27) // ESC key
            {
                RCLCPP_INFO(this->get_logger(), "ESC pressed. Exiting...");
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
    destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}