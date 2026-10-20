#include <unordered_map>
#include <Eigen/Core>
#include <deque>
#include <map>

struct SemanticVoxel {        
    std::unordered_map<int, double> class_probs;
    int total_points = 0;
    Eigen::Vector4f color;

    int get_dominant_class() const {
        int best_class = -1;
        int max_votes = -1;
        for (const auto& [cls, prob] : class_probs) {
            if (prob > max_votes) {
                max_votes = prob;
                best_class = cls;
            }
        }
        return best_class;
    }
};

// SemanticVoxel CreateVoxel(const std::vector<Eigen::Vector3f> & maskCloud);


struct VoxelIndex {
    int x, y, z;
    bool operator==(const VoxelIndex& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};


struct VoxelHash {
    std::size_t operator()(const VoxelIndex& k) const {
        return ((std::hash<int>()(k.x) * 73856093) ^ 
                (std::hash<int>()(k.y) * 19349663) ^ 
                (std::hash<int>()(k.z) * 83492791));
    }
};


struct SegmentedKeyframe {
    Eigen::Isometry3d T_world_sensor;
    std::vector<cv::Mat> masks;
    std::vector<int> classes;
    std::vector<std::array<int, 4>> bboxes;
    float fx, fy, cx, cy;
    int im_w, im_h;
    cv::Mat image;
};