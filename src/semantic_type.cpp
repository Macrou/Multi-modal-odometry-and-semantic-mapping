#include <semantic_type.hpp>


SemanticVoxel CreateVoxel(const std::vector<Eigen::Vector3f> &maskCloud, Eigen::Vector4f color, std::unordered_map<int, int> class_counts)
{
    size_t size = maskCloud.size();

    size_t idx10 = size / 10;
    size_t idx90 = (9 * size) / 10;

    std::vector<float> x_coords, y_coords, z_coords;

    x_coords.reserve(size);
    y_coords.reserve(size);
    z_coords.reserve(size);

    for (const auto &pt : maskCloud)
    {cloud
        x_coords.push_back(pt.x);
        y_coords.push_back(pt.y);
        z_coords.push_back(pt.z);
    }

    auto getPercentiles = [&](std::vector<float> &coords, float &p10, float &p90)
    {
        std::nth_element(coords.begin(), coords.begin() + idx10, coords.end());
        p10 = coords[idx10];

        std::nth_element(coords.begin(), coords.begin() + idx90, coords.end());
        p90 = coords[idx90];
    };

    float x10, x90, y10, y90, z10, z90;
    getPercentiles(x_coords, x10, x90);
    getPercentiles(y_coords, y10, y90);
    getPercentiles(z_coords, z10, z90);

    struct SemanticVoxel voxel;
    Eigen::Vector3f p10(x10, y10, z10);
    Eigen::Vector3f p90(x90, y90, z90);
    voxel.center = (p10 + p90) / 2.0f;
    float m = 1.5f;
    voxel.size = m * (p90 - p10).cwiseAbs().maxCoeff();

    voxel.color = color;

    class_counts = class_counts;

    return voxel;
}

