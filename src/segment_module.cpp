#include <segment_module.hpp>

#include <util/config_segmentation.hpp>
#include <cv_bridge/cv_bridge.h>
#include <glk/pointcloud_buffer_pcl.hpp>
#include <glim/util/convert_to_string.hpp>

namespace fs = std::filesystem;

namespace glim
{

    ImageSegmentationModule::ImageSegmentationModule() : logger(create_module_logger("img_seg"))
    {
        logger->info("Starting");
        const std::string config_path = GlobalConfigExt::get_config_path("config_segmentation");
        glim::Config config(config_path);
        rgbd_topic = config.param<std::string>("segmentation", "rgbd_topic", "realsense2_camera_msgs/msg/RGBD");
        model_path = config.param<std::string>("segmentation", "model_path", "../model/yolov26n-seg.engine");
        label_path = config.param<std::string>("segmentation", "label_path", "../labels/labels.txt");
        mask_threshold = config.param<float>("segmentation", "mask_threshold", 0.50);
        T_sensor_camera = config.param<Eigen::Isometry3d>("segmentation", "T_sensor_camera", Eigen::Isometry3d::Identity());
        T_camera_sensor = config.param<Eigen::Isometry3d>("segmentation", "T_camera_sensor", Eigen::Isometry3d::Identity());
        logger->info("T_sensor_camera={}", convert_to_string(T_sensor_camera));

        trtyolo::InferOption option;
        option.enableSwapRB();
        window_size = 10.0;
        detector = std::make_unique<trtyolo::SegmentModel>(model_path, option);
        labels = GenerateLabels(label_path);

        colorMap = GenerateColorMap(80);
        logger->info("Finished start up.");

        glim::OdometryEstimationCallbacks::on_new_frame.add([this](const EstimationFrame::ConstPtr &frame)
                                                            {
            std::lock_guard<std::mutex> lock(pose_mutex);
            latest_pose = frame->T_world_sensor(); });

        glim::PreprocessCallbacks::on_downsampling_finished.add(
            [this](gtsam_points::PointCloudCPU::Ptr &points)
            {
                std::lock_guard<std::mutex> lock(cloud_mutex);
                latest_lidar_cloud = points;
            });

        using std::placeholders::_1;
        glim::GlobalMappingCallbacks::on_insert_submap.add(std::bind(&ImageSegmentationModule::GlobalmapOnInsertSubmaps, this, _1));
        glim::GlobalMappingCallbacks::on_update_submaps.add(std::bind(&ImageSegmentationModule::GlobalmapOnUpdateSubmaps, this, _1));

        kill_switch = false;
        thread = std::thread([this]
                             { SegmentationTask(); });
    }

    ImageSegmentationModule::~ImageSegmentationModule()
    {
        kill_switch = true;
        thread.join();
    }

    std::vector<std::string> ImageSegmentationModule::GenerateLabels(const std::string &label_file)
    {
        std::ifstream file(label_file);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open labels file: " + label_file);
        }

        std::vector<std::string> labels;
        std::string label;
        while (std::getline(file, label))
        {
            labels.emplace_back(label);
        }
        return labels;
    }

    void ImageSegmentationModule::RefreshConsensusColors()
    {
        std::vector<std::pair<int, std::vector<Eigen::Vector4f>>> updates_needed;
        auto start = std::chrono::high_resolution_clock::now();
        {
            std::lock_guard<std::mutex> lock(render_buffer_mutex);
            std::lock_guard<std::mutex> voxel_lock(voxel_mutex);

            for (auto &[id, indices] : submap_voxel_indices)
            {
                auto &sem_colors = submap_semantic_colors[id];
                bool submap_changed = false;

                for (size_t i = 0; i < indices.size(); ++i)
                {
                    const auto &center_idx = indices[i];
                    std::unordered_map<int, int> neighborhood_votes;
                    bool voxel_exists = false;

                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        for (int dy = -1; dy <= 1; ++dy)
                        {
                            for (int dz = -1; dz <= 1; ++dz)
                            {

                                VoxelIndex neighbor_idx;
                                neighbor_idx.x = center_idx.x + dx;
                                neighbor_idx.y = center_idx.y + dy;
                                neighbor_idx.z = center_idx.z + dz;

                                auto it = voxel_map.find(neighbor_idx);
                                if (it != voxel_map.end())
                                {
                                    if (dx == 0 && dy == 0 && dz == 0)
                                        voxel_exists = true;

                                    int weight = (dx == 0 && dy == 0 && dz == 0) ? 2 : 1;

                                    for (const auto &[class_id, prob] : it->second.class_probs)
                                    {
                                        neighborhood_votes[class_id] += (prob * weight);
                                    }
                                }
                            }
                        }
                    }

                    if (voxel_exists && !neighborhood_votes.empty())
                    {

                        double max_prob = -1.0;
                        int smoothed_dominant_class = -1;

                        for (const auto &[class_id, prob] : neighborhood_votes)
                        {
                            if (prob > max_prob)
                            {
                                max_prob = prob;
                                smoothed_dominant_class = class_id;
                            }
                        }

                        if (smoothed_dominant_class != -1)
                        {
                            Eigen::Vector4f new_color = colorMap[smoothed_dominant_class];

                            if (sem_colors[i] != new_color)
                            {
                                sem_colors[i] = new_color;
                                submap_changed = true;
                            }
                        }
                    }
                }

                if (submap_changed && color_mode == 1)
                {
                    updates_needed.push_back({id, sem_colors});
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            logger->info("Consensus update took {:.2f} ms", std::chrono::duration<double, std::milli>(end - start).count());
        }

        if (!updates_needed.empty())
        {
            auto viewer = guik::LightViewer::instance();
            if (viewer)
            {
                viewer->invoke([this, updates = std::move(updates_needed)]()
                               {
                std::lock_guard<std::mutex> lock(render_buffer_mutex);
                for (const auto& [id, new_colors] : updates)
                {
                    auto it = submap_render_buffers.find(id);
                    if (it != submap_render_buffers.end())
                    {
                        it->second->add_color(new_colors);
                    }
                } });
            }
        }
    }

    void ImageSegmentationModule::ComputeMask(const trtyolo::SegmentRes &result, int i, int w, int h, cv::Mat &bool_mask)
    {
        cv::Mat float_mask(result.masks[i].height, result.masks[i].width, CV_32FC1, (void *)result.masks[i].data.data());
        cv::resize(float_mask, float_mask, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        cv::threshold(float_mask, bool_mask, 0.5, 255, cv::THRESH_BINARY);
    }

    void ImageSegmentationModule::SegmentationTask()
    {

        while (!kill_switch)
        {
            auto rgbd_images = input_image_queue.get_all_and_clear();

            if (rgbd_images.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            auto &rgbd_image = rgbd_images.back();
            trtyolo::Image segmentation_image(
                rgbd_image.rgb_image.data,
                rgbd_image.rgb_image.cols,
                rgbd_image.rgb_image.rows);
            try
            {
                auto start = std::chrono::high_resolution_clock::now();

                auto result = detector->predict(segmentation_image);

                auto mid = std::chrono::high_resolution_clock::now();

                HandleSegmentation(rgbd_image, result);

                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double, std::milli> infer_time = mid - start;
                std::chrono::duration<double, std::milli> vis_time = end - mid;

                logger->info("Inference: {:.2f} ms | Vis+Vox: {:.2f} ms | Objects: {}",
                             infer_time.count(), vis_time.count(), result.num);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Program error: " << e.what() << std::endl;
            }
        }
    }
    /**
    * @brief visualise the image and adds the segmented PC.
     */
    void ImageSegmentationModule::HandleSegmentation(struct RGBDStruct &rgbd_image, const trtyolo::SegmentRes &result)
    {
        auto viewer = guik::LightViewer::instance();
        // viewer->register_ui_callback("segmentation_selection", [this]
        //                              { draw_selection(); });

        Eigen::Isometry3d T_world_sensor;
        if (rgbd_image.rgb_image.empty())
        {
            logger->error("Image is empty! Cannot draw boxes.");
            return; // or throw an error
        }

        int im_h = rgbd_image.rgb_image.rows;
        int im_w = rgbd_image.rgb_image.cols;
        auto image_copy = rgbd_image.rgb_image.clone();

        for (size_t i = 0; i < result.num; ++i)
        {
            const trtyolo::Box &box = result.boxes[i];
            int cls = result.classes[i];
            float score = result.scores[i];
            std::string &label = labels[cls];
            std::string label_text = label + " " + cv::format("%.3f", score);

            std::array<int, 4> xyxy = box.xyxy();
            int w = std::max(xyxy[2] - xyxy[0] + 1, 1);
            int h = std::max(xyxy[3] - xyxy[1] + 1, 1);

            cv::Mat bool_mask;
            ComputeMask(result, i, w, h, bool_mask);
            AddMaskTo2DImage(image_copy, bool_mask, xyxy, box, label_text, im_w, im_h);
            if (score > mask_threshold)
            {
                UpdateVoxels(rgbd_image, bool_mask, cls, score, xyxy, im_w, im_h);
            }
        }

        if (viewer)
        {
            viewer->register_ui_callback("segmentation_color_toggle", [this]
                                         {

            
            std::vector<const char *> color_modes = {"RGB", "Segmentation"};                      
            if (ImGui::Combo("ColorMode", &color_mode, color_modes.data(), color_modes.size())) {
                UpdateAllSubmapColors();
            } });
        }
        viewer->invoke([img = std::move(image_copy)]() mutable
                       {
            auto texture = glk::create_texture(img);
            guik::LightViewer::instance()->update_image("segmentation_overlay", texture); });

        // if (!segmented_cloud.empty())
        // {
        //     viewer->invoke([cloud = segmented_cloud,
        //                     colors = segmented_cloud_colors]() mutable
        //                    {
        // auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(cloud);
        // cloud_buffer->add_color(colors);
        // guik::LightViewer::instance()->update_drawable(
        //     "segmented_cloud", cloud_buffer, guik::VertexColor()); });
        // }
    }

    void ImageSegmentationModule::AddMaskTo2DImage(cv::Mat &image, const cv::Mat &bool_mask, std::array<int, 4> xyxy, const trtyolo::Box &box, std::string label_text, int im_w, int im_h)
    {
        int base_line;
        try
        {
            cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);

            cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(251, 81, 163), 2, cv::LINE_AA);
            cv::rectangle(image, cv::Point(box.left, box.top - label_size.height), cv::Point(box.left + label_size.width, box.top), cv::Scalar(125, 40, 81), -1);
            cv::putText(image, label_text, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1);
        }
        catch (cv::Exception &e)
        {
            const char *err_msg = e.what();
            std::cout << "exception caught: " << err_msg << std::endl;
        }

        int x1 = std::max(0, xyxy[0]);
        int y1 = std::max(0, xyxy[1]);
        int x2 = std::min(im_w, xyxy[2] + 1);
        int y2 = std::min(im_h, xyxy[3] + 1);

        cv::Mat mask_image = cv::Mat::zeros(image.size(), CV_8UC1);

        int src_x_offset = std::max(0, -xyxy[0]);
        int src_y_offset = std::max(0, -xyxy[1]);

        int target_w = x2 - x1;
        int target_h = y2 - y1;
        if (target_w <= 0 || target_h <= 0)
            return;

        if (src_x_offset + target_w > bool_mask.cols)
            target_w = bool_mask.cols - src_x_offset;
        if (src_y_offset + target_h > bool_mask.rows)
            target_h = bool_mask.rows - src_y_offset;
        if (target_w <= 0 || target_h <= 0)
            return;

        cv::Rect target_rect(x1, y1, target_w, target_h);
        cv::Rect source_rect(src_x_offset, src_y_offset, target_w, target_h);

        cv::Mat color_image(image.size(), image.type(), cv::Scalar(251, 81, 163));

        cv::Mat masked_color_image;
        cv::bitwise_and(color_image, color_image, masked_color_image, mask_image);

        cv::addWeighted(image, 1.0, masked_color_image, 0.5, 0, image);
    }

    /**
    * Updates voxels labels and features. 
     */
    void ImageSegmentationModule::UpdateVoxels(struct RGBDStruct &rgbd_image, cv::Mat &bool_mask, int cls, float score, std::array<int, 4> xyxy, int im_w, int im_h)
    {
        static const Eigen::Matrix3f R_ros_to_cv = (Eigen::Matrix3f() << 0, -1, 0, 0, 0, -1, 1, 0, 0).finished();
        const float depth_scale = 0.001f;

        struct VoxelUpdate
        {
            VoxelIndex index;
            Eigen::Vector4f color;
            int class_id;
            float score;
        };

        std::vector<float> mask_depths;
        int x_offset = std::max(0, (int)xyxy[0]);
        int y_offset = std::max(0, (int)xyxy[1]);

        for (int r = 0; r < bool_mask.rows; ++r)
        {
            for (int c = 0; c < bool_mask.cols; ++c)
            {
                if (bool_mask.at<uchar>(r, c) > 0)
                {
                    int u = x_offset + c;
                    int v = y_offset + r;
                    if (u >= rgbd_image.depth_image.cols || v >= rgbd_image.depth_image.rows)
                        continue;

                    float z = rgbd_image.depth_image.at<uint16_t>(v, u) * depth_scale;
                    if (z > 0.1f && z <= 6.0f)
                    {
                        mask_depths.push_back(z);
                    }
                }
            }
        }

        if (mask_depths.empty())
            return;

        std::nth_element(mask_depths.begin(), mask_depths.begin() + mask_depths.size() / 2, mask_depths.end());
        float median_z = mask_depths[mask_depths.size() / 2];

        Eigen::Isometry3d T_world_camera;
        {
            std::lock_guard<std::mutex> lock(pose_mutex);
            T_world_camera = latest_pose;
        }

        Eigen::Isometry3f T_combined = (T_world_camera * T_sensor_camera.cast<double>()).cast<float>();

        float depth_tolerance = 0.5f;

        std::vector<VoxelUpdate> local_updates;

        const float inv_voxel_res = 1.0f / voxel_resolution;
        for (int r = 0; r < bool_mask.rows; ++r)
        {
            for (int c = 0; c < bool_mask.cols; ++c)
            {
                if (bool_mask.at<uchar>(r, c) > 0)
                {
                    int u = x_offset + c;
                    int v = y_offset + r;
                    if (u >= rgbd_image.depth_image.cols || v >= rgbd_image.depth_image.rows)
                        continue;

                    float z = rgbd_image.depth_image.at<uint16_t>(v, u) * depth_scale;

                    if (z <= 0.1f || z > 6.0f || std::abs(z - median_z) > depth_tolerance)
                        continue;

                    float x = (u - rgbd_image.cx) * z / rgbd_image.fx;
                    float y = (v - rgbd_image.cy) * z / rgbd_image.fy;

                    Eigen::Vector3f pt_cam(z, -x, -y);

                    Eigen::Vector3f pt_world = T_combined * pt_cam;

                    VoxelIndex v_idx;

                    v_idx.x = static_cast<int>(std::floor(pt_world.x() * inv_voxel_res));
                    v_idx.y = static_cast<int>(std::floor(pt_world.y() * inv_voxel_res));
                    v_idx.z = static_cast<int>(std::floor(pt_world.z() * inv_voxel_res));

                    cv::Vec3b bgr = rgbd_image.rgb_image.at<cv::Vec3b>(v, u);

                    Eigen::Vector4f rgb4f(
                        bgr[2] / 255.0f,
                        bgr[1] / 255.0f,
                        bgr[0] / 255.0f,
                        1.0f);

                    local_updates.push_back({v_idx, rgb4f, cls, score});
                    // segmented_cloud.push_back(pt_world);
                    // segmented_cloud_colors.push_back(rgb4f);
                }
            }
        }
        auto start = std::chrono::high_resolution_clock::now();
        // Baysian Filter
        if (!local_updates.empty())
        {
            std::lock_guard<std::mutex> voxel_lock(voxel_mutex);
            const double prior = 1.0 / 80.0;

            for (const auto &update : local_updates)
            {
                auto &voxel = voxel_map[update.index];

              
                if (voxel.class_probs.find(update.class_id) == voxel.class_probs.end())
                {
                    voxel.class_probs[update.class_id] = prior;
                }

                double old_belief = voxel.class_probs[update.class_id];

      
                double new_measurement = static_cast<double>(update.score);

        
                double unnormalized_prob = (new_measurement * old_belief) / prior;

                voxel.class_probs[update.class_id] = unnormalized_prob;

                double sum = 0.0;
                for (const auto &[cls_id, prob] : voxel.class_probs)
                {
                    sum += prob;
                }
                for (auto &[cls_id, prob] : voxel.class_probs)
                {
                    prob /= sum;
                }

                voxel.color = (voxel.color * voxel.total_points + update.color) / (voxel.total_points + 1);
                voxel.total_points++;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        logger->info("Bayesian update took {:.2f} ms", std::chrono::duration<double, std::milli>(end - start).count());
    }
    /**
    * @brief ROS callback.
     */
    void ImageSegmentationModule::RgbdCallback(const RGBDConstPtr &msg)
    {
        struct RGBDStruct rgbd_image;
        rgbd_image.rgb_image = cv_bridge::toCvCopy(msg->rgb, "bgr8")->image;
        rgbd_image.depth_image = cv_bridge::toCvCopy(msg->depth, "16UC1")->image;
        rgbd_image.fx = msg->rgb_camera_info.k[0];
        rgbd_image.fy = msg->rgb_camera_info.k[4];
        rgbd_image.cx = msg->rgb_camera_info.k[2];
        rgbd_image.cy = msg->rgb_camera_info.k[5];
        if (rgbd_image.rgb_image.cols != 640 || rgbd_image.rgb_image.rows != 640)
        {
            float scale_x = 640.0f / rgbd_image.rgb_image.cols;
            float scale_y = 640.0f / rgbd_image.rgb_image.rows;

            cv::Mat resized_rgb, resized_depth;

            cv::resize(rgbd_image.rgb_image, resized_rgb, cv::Size(640, 640));
            rgbd_image.rgb_image = resized_rgb;

            cv::resize(rgbd_image.depth_image, resized_depth, cv::Size(640, 640), 0, 0, cv ::INTER_NEAREST);
            rgbd_image.depth_image = resized_depth;

            rgbd_image.fx = rgbd_image.fx * scale_x;
            rgbd_image.fy = rgbd_image.fy * scale_y;
            rgbd_image.cx = rgbd_image.cx * scale_x;
            rgbd_image.cy = rgbd_image.cy * scale_y;
        }

        input_image_queue.push_back(rgbd_image);
    }

    std::vector<Eigen::Vector4f> ImageSegmentationModule::GenerateColorMap(int totalClasses)
    {
        std::vector<Eigen::Vector4f> colorMap(totalClasses);

        for (int i = 0; i < totalClasses; ++i)
        {
            // Use the Golden Ratio conjugate to spread colors evenly around the color wheel
            double hue = fmod(i * 0.618033988749895, 1.0) * 360.0;
            double s = 0.85; // High saturation
            double l = 0.50; // Balanced brightness

            // HSL to RGB conversion math
            double c = (1.0 - std::abs(2.0 * l - 1.0)) * s;
            double x = c * (1.0 - std::abs(fmod(hue / 60.0, 2.0) - 1.0));
            double m = l - c / 2.0;

            double r = 0, g = 0, b = 0;
            if (hue < 60)
            {
                r = c;
                g = x;
                b = 0;
            }
            else if (hue < 120)
            {
                r = x;
                g = c;
                b = 0;
            }
            else if (hue < 180)
            {
                r = 0;
                g = c;
                b = x;
            }
            else if (hue < 240)
            {
                r = 0;
                g = x;
                b = c;
            }
            else if (hue < 300)
            {
                r = x;
                g = 0;
                b = c;
            }
            else
            {
                r = c;
                g = 0;
                b = x;
            }

            // Direct float assignment (No scaling needed!)
            float x_ = static_cast<float>(r + m);
            float y_ = static_cast<float>(g + m);
            float z_ = static_cast<float>(b + m);

            colorMap[i] << x_, y_, z_, 1.0f;
        }
        return colorMap;
    }

    /**
     * @brief Submap pose update callback. Updates the pose and calls the RefreshConsensusColors() to change the semantics of the points
     */
    void ImageSegmentationModule::GlobalmapOnUpdateSubmaps(
        const std::vector<SubMap::Ptr> &updated_submaps)
    {
        auto start = std::chrono::high_resolution_clock::now();
        RefreshConsensusColors();

        std::vector<std::pair<std::string,
                              std::pair<std::shared_ptr<glk::PointCloudBuffer>, Eigen::Isometry3d>>>
            updates;
        {
            std::lock_guard<std::mutex> lock(render_buffer_mutex);
            for (const auto &submap : updated_submaps)
            {
                auto it = submap_render_buffers.find(submap->id);
                if (it != submap_render_buffers.end())
                    updates.push_back({"semantic_submap_" + std::to_string(submap->id),
                                       {it->second, submap->T_world_origin}});
            }
        }

        auto mid = std::chrono::high_resolution_clock::now();
        guik::LightViewer::instance()->invoke([updates = std::move(updates)]()
                                              {
        for (const auto &[name, buf_pose] : updates)
            guik::LightViewer::instance()->update_drawable(
                name, buf_pose.first,
                guik::VertexColor().set_model_matrix(
                    buf_pose.second.cast<float>().matrix())); });

        auto end = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief New submap insertion callback. Where the map fussion happens using NN-search with voxels
     */
    void ImageSegmentationModule::GlobalmapOnInsertSubmaps(const glim::SubMap::ConstPtr &submap)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto cloud = submap->frame;
        Eigen::Isometry3f T_world_submap = submap->T_world_origin.cast<float>();

        std::vector<Eigen::Vector3f> temporary_points;
        std::vector<VoxelIndex> active_voxels;
        std::vector<VoxelIndex> temporary_voxel_indices;

        temporary_points.reserve(cloud->num_points);
        temporary_voxel_indices.reserve(cloud->num_points);

        std::vector<int> voxel_density_counts;
        voxel_density_counts.reserve(cloud->num_points);
        active_voxels.reserve(temporary_points.size());

        {
            std::lock_guard<std::mutex> voxel_lock(voxel_mutex);
            for (size_t i = 0; i < cloud->num_points; ++i)
            {
                Eigen::Vector3f pt_submap = cloud->points[i].cast<float>().head<3>();
                Eigen::Vector3f pt_world = T_world_submap * pt_submap;

                VoxelIndex v_idx;
                v_idx.x = static_cast<int>(std::floor(pt_world.x() / voxel_resolution));
                v_idx.y = static_cast<int>(std::floor(pt_world.y() / voxel_resolution));
                v_idx.z = static_cast<int>(std::floor(pt_world.z() / voxel_resolution));

                // Check if this voxel exists in our global map
                auto it = voxel_map.find(v_idx);
                if (it != voxel_map.end() && it->second.total_points > 0)
                {
                    temporary_points.push_back(pt_submap);
                    temporary_voxel_indices.push_back(v_idx);
                    voxel_density_counts.push_back(it->second.total_points);
                }
            }
        }
        
        if (temporary_points.empty())
            return;

        double discard_percentage = 0.30;

        std::vector<int> sorted_counts = voxel_density_counts;
        std::sort(sorted_counts.begin(), sorted_counts.end());

        size_t cutoff_index = static_cast<size_t>(sorted_counts.size() * discard_percentage);

        cutoff_index = std::min(cutoff_index, sorted_counts.size() - 1);
        int density_threshold = sorted_counts[cutoff_index];

        logger->debug("Filtering edge noise. Discarding bottom {}%. Voxel density threshold set to: {} points.",
                      discard_percentage * 100.0, density_threshold);

        std::vector<Eigen::Vector3f> colored_points;
        std::vector<Eigen::Vector4f> semantic_colors;
        std::vector<Eigen::Vector4f> rgb_colors;

        colored_points.reserve(temporary_points.size());
        semantic_colors.reserve(temporary_points.size());
        rgb_colors.reserve(temporary_points.size());

        {
            std::lock_guard<std::mutex> voxel_lock(voxel_mutex);
            for (size_t i = 0; i < temporary_points.size(); ++i)
            {

                if (voxel_density_counts[i] <= density_threshold)
                {
                    continue;
                }

                const auto &v_idx = temporary_voxel_indices[i];
                auto &voxel = voxel_map[v_idx];

                colored_points.push_back(temporary_points[i]);
                semantic_colors.push_back(colorMap[voxel.get_dominant_class()]);
                rgb_colors.push_back(voxel.color);
                active_voxels.push_back(v_idx);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        logger->info("Submap insertion took {:.2f} ms", std::chrono::duration<double, std::milli>(end - start).count());
        if (!colored_points.empty())
        {

            {
                std::lock_guard<std::mutex> lock(render_buffer_mutex);
                submap_semantic_colors[submap->id] = semantic_colors;
                submap_rgb_colors[submap->id] = rgb_colors;
                submap_voxel_indices[submap->id] = active_voxels;
            }

            auto viewer = guik::LightViewer::instance();
            if (!viewer)
                return;

            viewer->invoke([this,
                            id = submap->id,
                            T_w_o = submap->T_world_origin,
                            points = std::move(colored_points)]() mutable
                           {
            auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(points);
            
            switch (color_mode)
            {
            case 0: cloud_buffer->add_color(submap_rgb_colors[id]); break;
            case 1: cloud_buffer->add_color(submap_semantic_colors[id]); break;
            }
       
            {
                std::lock_guard<std::mutex> lock(render_buffer_mutex);
                submap_render_buffers[id] = cloud_buffer;
            }

            std::string drawable_name = "semantic_submap_" + std::to_string(id);
            guik::LightViewer::instance()->update_drawable(
                drawable_name, 
                cloud_buffer, 
                guik::VertexColor().set_model_matrix(T_w_o.cast<float>().matrix())
            ); });
        }
    }

    /**
     * @brief Updates the submap buffers for visualistion. 
     */
    void ImageSegmentationModule::UpdateAllSubmapColors()
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto viewer = guik::LightViewer::instance();
        if (!viewer)
            return;

        viewer->invoke([this]()
                       {
        std::lock_guard<std::mutex> lock(render_buffer_mutex);
        
        for (auto& [id, buffer] : submap_render_buffers) {
            switch (color_mode)
            {
            case 0:
                buffer->add_color(submap_rgb_colors[id]);
                break;
            
            case 1:
                buffer->add_color(submap_semantic_colors[id]);
                break;
            }
        } });
        auto end = std::chrono::high_resolution_clock::now();
        logger->info("Submap update took {:.2f} ms", std::chrono::duration<double, std::milli>(end - start).count());
    }
}
