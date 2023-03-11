#ifndef BOVW_PROJECT_HPP_
#define BOVW_PROJECT_HPP_

#include "../lib/bow/include/bow.hpp"
#include <imagebrowser/image_browser.hpp>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <serialization/convert_dataset.hpp>

class PlaceRecognizer {
public:
  PlaceRecognizer();
  explicit PlaceRecognizer(const std::string &dictionary_path,
                           const std::string &bin_dataset_path,
                           const std::string &raw_dataset_path);
  ~PlaceRecognizer(){};

  void find_places(const std::string &image_path);
  void find_places(cv::Mat image);

private:
  float cosine_distance(ipb::Histogram h1, ipb::Histogram h2);

  std::vector<std::filesystem::path> raw_images_;
  ipb::BowDictionary &dictionary_ = ipb::BowDictionary::GetInstance();
  cv::Ptr<cv::SiftFeatureDetector> detector_ =
      cv::SiftFeatureDetector::create();
  cv::Ptr<cv::SiftDescriptorExtractor> extractor_ =
      cv::SiftDescriptorExtractor::create();
  std::vector<ipb::Histogram> histograms_;
  std::vector<int> bin_frequency_;
  cv::Mat descriptors_;
  std::vector<cv::KeyPoint> keypoints_;
};

#endif