#include <segment_module.hpp>

#define GLIM_ROS2
#include <glim/util/ros_cloud_converter.hpp>

namespace glim
{

    std::vector<GenericTopicSubscription::Ptr> ImageSegmentationModule::create_subscriptions()
    {
        const auto rgbd_sub = std::make_shared<TopicSubscription<realsense2_camera_msgs::msg::RGBD>>(rgbd_topic, [this](const RGBDConstPtr msg)
                                                                                                     { RgbdCallback(msg); });
        return {rgbd_sub};
    }
}
