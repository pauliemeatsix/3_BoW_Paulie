
#include <cstdlib>

#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "../include/convert_dataset.hpp"
#include "../include/serialize.hpp"

#include "../include/html_writer.hpp"
#include "../include/image_browser.hpp"
#include "bow.hpp"

#include <cstddef>
#include <cstdio>

#include <iostream>
#include <regex>
#include <sstream>
#include <string>

using namespace std;

int main() {

  // Homework 3

  // const image_browser::ScoredImage img1{"../web_app/data/000000.png", 0.98};
  // const image_browser::ScoredImage img2{"../web_app/data/000100.png", 0.96};
  // const image_browser::ScoredImage img3{"../web_app/data/000200.png", 0.88};

  // const image_browser::ScoredImage img4{"../web_app/data/000300.png", 0.87};
  // const image_browser::ScoredImage img5{"../web_app/data/000400.png", 0.80};
  // const image_browser::ScoredImage img6{"../web_app/data/000500.png", 0.79};

  // const image_browser::ScoredImage img7{"../web_app/data/000600.png", 0.76};
  // const image_browser::ScoredImage img8{"../web_app/data/000700.png", 0.75};
  // const image_browser::ScoredImage img9{"../web_app/data/000800.png", 0.20};
  // const image_browser::ImageRow triad_1{img1, img2, img3};
  // const image_browser::ImageRow triad_2{img4, img5, img6};
  // const image_browser::ImageRow triad_3{img7, img8, img9};

  // const std::string title = "Image Browser";
  // const std::string stylesheet = "../web_app/style.css";
  // std::vector<image_browser::ImageRow> images_rows = {triad_1, triad_2,
  //                                                     triad_3};

  // image_browser::CreateImageBrowser(title, stylesheet, images_rows);

  // Homework 5 onlyu

  // Serialize
  // cv::Mat lenna = cv::imread("../data/lenna.png", cv::IMREAD_GRAYSCALE);
  // ipb::serialization::Serialize(lenna, "../data/lenna.bin");

  // // Convert Data Set
  std::filesystem::path img_path = "../dataset/raw_imgs";
  std::filesystem::path bin_path = "../bin/";
  ipb::serialization::sifts::ConvertDataset(img_path);

  std::vector<cv::Mat> loaded_bins =
      ipb::serialization::sifts::LoadDataset(bin_path);

  // std::cout << loaded_bins.size() << std::endl;

  // cv::Mat concat;
  // cv::vconcat(loaded_bins, concat);

  // std::cout << concat.size << std::endl;

  // Deserialize
  // cv::Mat lenna_bin = ipb::serialization::Deserialize(
  //     "../dataset/sifts_bin/imageCompressedCam0_0000000.bin");

  int max_iter = 40;
  int K = 5;
  cv::Mat Clustered = ipb::kMeans(loaded_bins, K, max_iter);
  std::cout << Clustered << std::endl;

  ipb::BowDictionary &dict = ipb::BowDictionary::GetInstance();
  dict.set_params(30, 15, loaded_bins);

  return 0;
}
