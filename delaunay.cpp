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
// Note that matplotlibcpp doesn't appear to be maintained any longer and thus has with modern Python and C++
// Some lines of the h file have thus been commented out
#include "matplotlibcpp.h"

// I'm sure there's a constant somewhere here
#ifndef M_PI
#define M_PI 3.1415926535
#endif

#define MIN_SPACING 1
#define MAX_SPACING 25

#define TEST_CASE 5

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

std::vector<size_t> get_delaunator_triangles(
    const std::vector<Eigen::Vector2d>& left_cones, 
    const std::vector<Eigen::Vector2d>& right_cones,
    const std::vector<Eigen::Vector2d>& all_cones
) {
    // For whatever reason, Delaunator inputs direct list of coords: [x0, y0, x1, y1...] 
    // Why not list of points? Like: [(x0, y0), (x1, y1)...]
    std::vector<double> coords;
    coords.reserve((left_cones.size() + right_cones.size()) * 2);

    for (const auto& p : all_cones) {
        coords.push_back(p.x());
        coords.push_back(p.y());
    }

    delaunator::Delaunator d(coords);

    return d.triangles;
}

// ima modify filtered_triangles because why the hell not
std::vector<Eigen::Vector2d> generate_midpoints(
    const std::vector<Eigen::Vector2d>& left_cones, 
    const std::vector<Eigen::Vector2d>& right_cones, 
    const std::vector<Eigen::Vector2d>& all_cones,
    const std::vector<size_t>& triangles,
    std::vector<size_t>& filtered_triangles
) {
    std::vector<Eigen::Vector2d> midpoints;
    std::set<std::pair<size_t, size_t>> bad_edges;    

    // Apparently, C++ prematurely optimizes bools into bits instead of chars. Not a problem for me
    std::vector<bool> is_left(left_cones.size(), true);
    is_left.insert(is_left.end(), right_cones.size(), false);

    for(size_t i = 0; i < triangles.size(); i += 3) {
        size_t simplex[3] = { triangles[i], triangles[i + 1], triangles[i + 2] };

        for (int j = 0; j < 3; j++) {
            // Indices work as expected; coords fed into Delaunator are the weird ones
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

            // If an edge is too small or too large, the entire Delaunay triangle is guaranteed to be outside the track. Skip entirely 
            if(MIN_SPACING > width || width > MAX_SPACING) {
                bad_edges.insert(edge);
                break;
            }

            midpoints.push_back((point_a + point_b) / 2);

            filtered_triangles.push_back(simplex[0]);
            filtered_triangles.push_back(simplex[1]);
            filtered_triangles.push_back(simplex[2]);

            break;
        }
    }

    // Is throwing errors here good or bad for the car?

    return midpoints;
}

// Change start position to (0, 0) for actual code
std::vector<Eigen::Vector2d> order_midpoints(
    const std::vector<Eigen::Vector2d>& points, 
    const Eigen::Vector2d& start
) {
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

        // Throw error, or fail gracefully?
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
    
    // if (!f.is_open()) {
    //     throw std::runtime_error("Oh dear");
    // }

    auto entry = get_data();

    std::string file = entry["file"];
    Eigen::Vector2d start(entry["position"][0], entry["position"][1]);
    double heading_rads = entry["heading"].get<double>() * M_PI / 180.0;

    load_csv(file);

    std::vector<Eigen::Vector2d> all_cones(left);
    all_cones.insert(all_cones.end(), right.begin(), right.end());

    // GPT says compiler may optimize vector setting. Interesting
    std::vector<size_t> triangles = get_delaunator_triangles(left, right, all_cones);

    std::vector<size_t> filtered_triangles;
    // Not sure how to order in place
    std::vector<Eigen::Vector2d> vec = order_midpoints(generate_midpoints(left, right, all_cones, triangles, filtered_triangles), start);

    std::vector<double> left_x;
    std::vector<double> left_y;

    for(const auto& point: left){
        left_x.push_back(point.x());
        left_y.push_back(point.y());
    }

    std::vector<double> right_x;
    std::vector<double> right_y;

    for(const auto& point: right){
        right_x.push_back(point.x());
        right_y.push_back(point.y());
    }

    std::vector<double> mid_x;
    std::vector<double> mid_y;

    for(const auto& point: vec){
        mid_x.push_back(point.x());
        mid_y.push_back(point.y());
    }

    matplotlibcpp::plot(left_x, left_y, "bo-");
    matplotlibcpp::plot(right_x, right_y, {{"color", "gold"}, {"linestyle", "-"}, {"marker", "o"}});
    matplotlibcpp::plot(mid_x, mid_y, {{"color", "forestgreen"}, {"linestyle", "-"}, {"marker", "x"}});

    // Not very efficient, but oh well
    std::vector<double> x_tri(4);
    std::vector<double> y_tri(4);

    for (size_t i = 0; i < filtered_triangles.size(); i += 3) {
        size_t vertex_1 = filtered_triangles[i];
        size_t vertex_2 = filtered_triangles[i + 1];
        size_t vertex_3 = filtered_triangles[i + 2];

        x_tri[0] = all_cones[vertex_1].x();
        y_tri[0] = all_cones[vertex_1].y();

        x_tri[1] = all_cones[vertex_2].x();
        y_tri[1] = all_cones[vertex_2].y();

        x_tri[2] = all_cones[vertex_3].x();
        y_tri[2] = all_cones[vertex_3].y();

        x_tri[3] = x_tri[0];
        y_tri[3] = y_tri[0];

        matplotlibcpp::plot(x_tri, y_tri, "rx--");
    }

    std::vector<double> wrapper_x = {start.x()};
    std::vector<double> wrapper_y = {start.y()};

    std::vector<double> wrapper_x_dir = {std::cos(heading_rads)};
    std::vector<double> wrapper_y_dir = {std::sin(heading_rads)};

    matplotlibcpp::quiver(wrapper_x, wrapper_y, wrapper_x_dir, wrapper_y_dir);

    matplotlibcpp::axis("equal");

    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}