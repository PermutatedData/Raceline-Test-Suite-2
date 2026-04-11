#include <iostream>
#include <filesystem>
#include <vector>
#include <set>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>

#include "json.hpp"
#include "delaunator.hpp"

// I'm sure there's a constant somewhere here
#ifndef M_PI
#define M_PI 3.1415926535
#endif

#define MIN_SPACING 1
#define MAX_SPACING 25

#define TEST_CASE 4

// Change as necessary
const std::string ABS_FILE_PATH = "C:/Richard's Programming/CMR Stuff/Raceline-Testing-2/";

std::vector<Eigen::Vector2d> left;
std::vector<Eigen::Vector2d> right;

nlohmann::json get_data() {
    std::ifstream file(ABS_FILE_PATH + "test_configs.json");
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file");
    }
    
    nlohmann::json j;
    file >> j;

    return j[TEST_CASE - 1];
}

void load_csv(const std::string& filename) {
    std::ifstream file(ABS_FILE_PATH + "tracks/" + filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file");
    }

    std::vector<Eigen::Vector2d> points;

    // Apparently, reading after EOF will still not break anything
    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string x_str, y_str, type;

        if (!std::getline(ss, x_str, ',')) continue;
        if (!std::getline(ss, y_str, ',')) continue;
        if (!std::getline(ss, type, ',')) continue;

        double x = std::stod(x_str);
        double y = std::stod(y_str);

        // Emplace back: another GPT thing?
        if(type.compare("left") == 0) {
            left.emplace_back(x, y);
        } 
        else if(type.compare("right") == 0) {
            right.emplace_back(x, y);
        }
        else {

        }
        points.emplace_back(x, y);
    }
}

std::vector<Eigen::Vector2d> generate_midpoints(const std::vector<Eigen::Vector2d>& left_cones, const std::vector<Eigen::Vector2d>& right_cones) {
    std::vector<Eigen::Vector2d> midpoints;
    std::set<std::pair<size_t, size_t>> bad_edges;    

    std::vector<Eigen::Vector2d> all_cones = left_cones;
    all_cones.insert(all_cones.end(), right_cones.begin(), right_cones.end());

    // Apparently, C++ prematurely optimizes bools into bits instead of chars. Not a problem for me
    std::vector<bool> is_left(left_cones.size(), true);
    is_left.insert(is_left.end(), right_cones.size(), false);

    // For whatever reason, Delaunator inputs direct list of coords: [x0, y0, x1, y1...] 
    // Why not list of points? Like: [(x0, y0), (x1, y1)...]
    std::vector<double> coords;
    coords.reserve((left_cones.size() + right_cones.size()) * 2);

    for (const auto& p : all_cones) {
        coords.push_back(p.x());
        coords.push_back(p.y());
    }

    // All good here. Delaunator is cooked?

    delaunator::Delaunator d(coords);

    for(size_t i = 0; i < d.triangles.size(); i += 3) {
        size_t simplex[3] = { d.triangles[i], d.triangles[i + 1], d.triangles[i + 2] };

        for (int j = 0; j < 3; j++) {
            size_t index_a = simplex[j];
            size_t index_b = simplex[(j + 1) % 3];

            std::pair<size_t, size_t> edge = std::minmax(index_a, index_b);

            if(bad_edges.count(edge) != 0) {
                continue;
            }

            if(is_left[index_a] == is_left[index_b]) {
                bad_edges.insert(edge);
                continue;
            }

            Eigen::Vector2d point_a = all_cones[index_a];
            Eigen::Vector2d point_b = all_cones[index_b];

            double width = (point_a - point_b).norm();

            // If an edge is too small or too larger, the entire Delaunay triangle is guaranteed to be outside the track. Skip entirely 
            if(MIN_SPACING > width || width > MAX_SPACING) {
                bad_edges.insert(edge);
                break;
            }

            midpoints.push_back((point_a + point_b) / 2);
            break;
        }
    }

    // Is throwing errors here good or bad?

    return midpoints;
}

// Change start position to (0, 0) for actual code
std::vector<Eigen::Vector2d> order_midpoints(const std::vector<Eigen::Vector2d>& points, const Eigen::Vector2d& start) {
    // """
    // start_pos: starting index
    
    // returns:
    //     path: ordered points
    // """
    
    // ordered = []
    
    // n = len(edges)
    // visited_indices = [] # Rather than constantly appending to list, a fixed array of correct size could be used. Interesting

    // edge_points = indices_to_points(edges) # Numpy magic. Works with not just list of indices, but list of edge indices
    // midpoints = (edge_points[:,0] + edge_points[:,1]) / 2

    // current = start_pos

    // while len(visited_indices) < n:
    //     dist = np.linalg.norm(midpoints - current, axis=1)
    //     dist[visited_indices] = np.inf
        
    //     nearest_index = np.argmin(dist)
    //     nearest = midpoints[nearest_index]

    //     visited_indices.append(nearest_index)
    //     ordered.append(nearest)
        
    //     current = nearest

    // return np.array(ordered)

    size_t len = points.size();

    std::vector<Eigen::Vector2d> ordered_pts;
    ordered_pts.reserve(len);

    std::vector<bool> visited(len, false);

    // hmm...
    Eigen::Vector2d current = start;

    while(ordered_pts.size() < len) {
        double min_dist = std::numeric_limits<double>::max();
        size_t nearest_index = std::numeric_limits<size_t>::max();

        for(size_t i = 0; i < len; i++) {
            if(visited[i]) {
                continue;
            }

            double dist = (points[i] - current).norm();

            if(dist < min_dist) {
                min_dist = dist;
                nearest_index = i;
            }
        }

        // Throw error?
        if (nearest_index == std::numeric_limits<size_t>::max()) {
            break;
        }

        ordered_pts.push_back(points[nearest_index]);
        visited[nearest_index] = true;

        current = points[nearest_index];
    }

    return ordered_pts;
}

int main() {
    // Compiler points to ucrt64 bin. Ok I think?
    // std::cout << std::filesystem::current_path() << std::endl;

    // Irritatingly, I have to use the absolute file path
    // std::ifstream f("C:\\Richard's Programming\\CMR Stuff\\driverless\\driverless_ws\\src\\delaunay\\delaunay.cpp");
    
    // if (!f.is_open()) {
    //     throw std::runtime_error("Oh dear");
    // }

    auto entry = get_data();

    std::string file = entry["file"];
    Eigen::Vector2d start(entry["position"][0], entry["position"][1]);
    double heading_rads = entry["heading"].get<double>() * M_PI / 180.0;
    (void) heading_rads; // ?

    load_csv(file);
    // Not sure how to order in place
    // GPT says compiler may optimize this. Interesting
    std::vector<Eigen::Vector2d> vec = order_midpoints(generate_midpoints(left, right), start);
    
    // TODO: display

    return 0;
}