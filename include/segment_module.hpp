#include <deque>
#include <mutex>
#include <atomic>
#include <thread>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <glk/texture_opencv.hpp>
#include <vector>

#include <glim/util/logging.hpp>
#include <glim/util/concurrent_vector.hpp>
#include <glim/util/config.hpp>
#include <glim/preprocess/callbacks.hpp>
#include <glim/odometry/callbacks.hpp>
#include <glim/util/extension_module.hpp>
#include <glim/mapping/callbacks.hpp>

#include <implot.h>
#include <guik/viewer/light_viewer.hpp>
#include <realsense2_camera_msgs/msg/rgbd.hpp>
#include <trtyolo/infer/trtyolo.hpp>
#include <glim/util/extension_module_ros2.hpp>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <semantic_type.hpp>

using ExtensionModuleBase = glim::ExtensionModuleROS2;
using RGBDConstPtr = realsense2_camera_msgs::msg::RGBD::ConstSharedPtr;

namespace glim
{

    struct RGBDStruct
    {
        cv::Mat rgb_image;
        cv::Mat depth_image;
        double fx;
        double fy;
        double cx;
        double cy;
    };

    class ImageSegmentationModule : public ExtensionModuleBase
    {
    public:
        ImageSegmentationModule();
        ~ImageSegmentationModule();

        std::vector<std::string> GenerateLabels(const std::string &label_file);
        void RgbdCallback(const RGBDConstPtr &msg);
        void SegmentationTask();
        void ComputeMask(const trtyolo::SegmentRes &result, int i, int w, int h, cv::Mat &bool_mask);
        void HandleSegmentation(struct RGBDStruct &rgbd_image, const trtyolo::SegmentRes &result);
        void AddMaskTo2DImage(cv::Mat &image, const cv::Mat &bool_mask, std::array<int, 4> xyxy, const trtyolo::Box &box, std::string label_text, int im_w, int im_h);
        void UpdateVoxels(struct RGBDStruct &rgbd_image, cv::Mat &bool_mask, int cls, float score, std::array<int, 4> xyxy, int im_w, int im_h);
        void Segment();
        void GlobalmapOnUpdateSubmaps(const std::vector<SubMap::Ptr> &updated_submaps);
        void GlobalmapOnInsertSubmaps(const SubMap::ConstPtr &submap);
        std::vector<Eigen::Vector4f> GenerateColorMap(int totalClasses);

    private:
        std::vector<GenericTopicSubscription::Ptr> create_subscriptions() override;

        double window_size;
        ConcurrentVector<struct RGBDStruct> input_image_queue;
        std::vector<struct RGBDStruct> rgbd_images_list;

        std::unique_ptr<trtyolo::SegmentModel> detector;
        std::atomic_bool kill_switch;
        std::thread thread;

        int color_mode;

        std::vector<Eigen::Vector3f> segmented_cloud;
        std::vector<Eigen::Vector4f> segmented_cloud_colors;

        gtsam_points::PointCloudCPU::Ptr latest_lidar_cloud;
        std::shared_ptr<spdlog::logger> logger;
        std::vector<std::string> labels;
        std::vector<cv::Mat> bool_masks;
        std::string label_path, model_path, rgbd_topic;
        std::mutex pose_mutex;
        std::mutex cloud_mutex;
        Eigen::Isometry3d T_sensor_camera;
        Eigen::Isometry3d T_camera_sensor;
        Eigen::Isometry3d latest_pose = Eigen::Isometry3d::Identity();

        std::mutex voxel_mutex;
        std::unordered_map<VoxelIndex, SemanticVoxel, VoxelHash> voxel_map;
        // Submaps
        std::mutex submap_mutex;
        std::vector<SubMap::ConstPtr> submaps;
        std::vector<Eigen::Isometry3d> submap_poses;

        std::vector<Eigen::Vector4f> colorMap;

        std::deque<SegmentedKeyframe> keyframe_buffer;
        std::mutex buffer_mutex;

        std::unordered_map<int, std::vector<Eigen::Vector4f>> submap_semantic_colors;
        std::unordered_map<int, std::vector<Eigen::Vector4f>> submap_rgb_colors;

        void UpdateAllSubmapColors();

        // Store the colored OpenGL buffers for each submap ID
        std::map<int, std::shared_ptr<glk::PointCloudBuffer>> submap_render_buffers;
        std::mutex render_buffer_mutex;

        float voxel_resolution = 0.1f;
        const float MIN_Z = 0.28f;
        const float FOV_H = 69.0f * M_PI / 180.0f;
        const float FOV_V = 42.0f * M_PI / 180.0f;

        bool draw_segmentation;
        float mask_threshold;

        std::unordered_map<int, std::vector<VoxelIndex>> submap_voxel_indices;
        void RefreshConsensusColors();
    };
}