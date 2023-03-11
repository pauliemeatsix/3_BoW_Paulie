#include "../include/bow.hpp"
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
using namespace cv;
using namespace std;

cv::Mat ipb::stackMatrices(const std::vector<cv::Mat> &descriptors) {
  cv::Mat vector_concat;
  cv::vconcat(descriptors, vector_concat);
  std::cout << "okay " << std::endl;

  return vector_concat;
}

cv::Mat ipb::kMeans(const std::vector<cv::Mat> &descriptors, int k,
                    int max_iter) {
  Mat centers;
  Mat labels;
  cv::Mat v_concat = stackMatrices(descriptors);

  cv::kmeans(
      v_concat, k, labels,
      TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 100000, 0.0001),
      max_iter, KMEANS_PP_CENTERS, centers);

  return centers;
}