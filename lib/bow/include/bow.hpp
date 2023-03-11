#ifndef BOW_HPP_
#define BOW_HPP_

// #include "../../lib/serialization/include/convert_dataset.hpp"
// #include <../serialization/include/convert_dataset.hpp>
#include "../../serialization/include/convert_dataset.hpp"
#include "../../serialization/include/serialize.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <execution>
#include <filesystem>
#include <functional>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <thread>
#include <vector>

namespace ipb {
cv::Mat stackMatrices(const std::vector<cv::Mat> &descriptors);
cv::Mat kMeans(const std::vector<cv::Mat> &descriptors, int k, int max_iter);

class BowDictionary {
private:
  BowDictionary() = default;
  ~BowDictionary() = default;

  // BowDictionary(const BowDictionary& other);
  // BowDictionary& operator=(const BowDictionary& other);

  // static BowDictionary *instancePtr;

  // Getters methods
  int max_iterations_value{};
  int size_value{};
  std::vector<cv::Mat> descriptors_sift;
  cv::Mat vocabulary_dictionary;

  void repeat_kmeans() {
    std::cout << "Repeating kemans clustering " << std::endl;
    vocabulary_dictionary =
        ipb::kMeans(descriptors_sift, size_value, max_iterations_value);
    std::cout << "New centers: " << vocabulary_dictionary << std::endl;
  }

public:
  BowDictionary(const BowDictionary &obj) = delete; // delete copy const
                                                    //
  // static BowDictionary *getInstance() { return instancePtr; }

  void operator=(const BowDictionary &) = delete;
  BowDictionary(BowDictionary &&other) = delete;
  BowDictionary &operator=(BowDictionary &&other) = delete;

  // Getters methods
  int max_iterations() const { return max_iterations_value; }
  int size() const { return size_value; }; // number of centroids / codewords
  std::vector<cv::Mat> descriptors() const { return descriptors_sift; };
  cv::Mat vocabulary() const { return vocabulary_dictionary; };

  int total_features() const {
    return descriptors_sift.size();
  }; // number of input features
  bool empty() { return vocabulary_dictionary.empty(); };

  // Setters methods
  void set_max_iterations(int max_iter);
  void set_size(int size);
  void set_descriptors(std::vector<cv::Mat> descriptors);
  void set_params(int max_iter, int size, std::vector<cv::Mat> descriptors);

  static BowDictionary &GetInstance() {
    static BowDictionary instance;
    return instance;
  }
};

} // namespace ipb
#endif