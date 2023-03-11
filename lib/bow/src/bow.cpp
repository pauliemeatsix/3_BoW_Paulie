#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../include/bow.hpp"

using namespace cv;
using namespace std;

// Setters methods
void ipb::BowDictionary::set_max_iterations(int max_iter) {
  max_iterations_value = max_iter;
  // repeat_();
  repeat_kmeans();
}
void ipb::BowDictionary::set_size(int size) {
  size_value = size;
  repeat_kmeans();
}
void ipb::BowDictionary::set_descriptors(std::vector<cv::Mat> descriptors) {
  descriptors_sift = std::move(descriptors);
  repeat_kmeans();
}
void ipb::BowDictionary::set_params(int max_iter, int size,
                                    std::vector<cv::Mat> descriptors) {
  max_iterations_value = max_iter;
  descriptors_sift = std::move(descriptors);
  size_value = size;
  repeat_kmeans();
}
