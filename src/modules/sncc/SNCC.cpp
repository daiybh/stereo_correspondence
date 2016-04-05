/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   SNCC.cpp
 * Author: user
 * 
 * Created on 4. dubna 2016, 10:52
 */

#include "SNCC.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include <limits>
#include <c++/4.9/limits>

namespace yuri {
    namespace sncc {

        IOTHREAD_GENERATOR(SNCC)
        MODULE_REGISTRATION_BEGIN("sncc")
        REGISTER_IOTHREAD("sncc", SNCC)
        MODULE_REGISTRATION_END()
        SNCC::SNCC(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters) :
        core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>(log_, parent, 2, std::string("sncc")) {
            IOTHREAD_INIT(parameters)
                    //set_supported_formats({core::raw_format::rgba32});
            supported_formats_.push_back(core::raw_format::y8);
        }

        core::Parameters SNCC::configure() {
            core::Parameters p = base_type::configure();
            p.set_description("SNCC Disparity computation");
            p["num_disparities"]["Number of disparities"]=16;
            return p;
        }

        float SNCC::getPixel(unsigned char *image, int x, int y, int width, int height) {
            if (x < 0 || x >= width) {
                return 0;
            }
            if (y < 0 || y >= height) {
                return 0;
            }
            return float(image[y * width + x]);
        }
        
        float SNCC::getPixel(float *image, int x, int y, int width, int height) {
            if (x < 0 || x >= width) {
                return 0;
            }
            if (y < 0 || y >= height) {
                return 0;
            }
            return float(image[y * width + x]);
        }

        unsigned char* SNCC::computeDisparity(unsigned char *left, unsigned char *right, int width, int height, int maxDisparity, int avgWindow) {
            float *means_left, *means_right;
            float *sds_left, *sds_right;

            means_left = new float[width * height];
            means_right = new float[width * height];
            sds_left = new float[width * height];
            sds_right = new float[width * height];

            float ***patches_left, ***patches_right;
            patches_left = new float**[height];
            patches_right = new float**[height];
            for (int i = 0; i < height; i++) {
                patches_left[i] = new float*[width];
                patches_right[i] = new float*[width];
                for (int j = 0; j < width; j++) {
                    patches_left[i][j] = new float[9];
                    patches_right[i][j] = new float[9];
                }
            }

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    patches_left[h][w][0] = getPixel(left, w - 1, h - 1, width, height);
                    patches_left[h][w][1] = getPixel(left, w, h - 1, width, height);
                    patches_left[h][w][2] = getPixel(left, w + 1, h - 1, width, height);
                    patches_left[h][w][3] = getPixel(left, w - 1, h, width, height);
                    patches_left[h][w][4] = getPixel(left, w, h, width, height);
                    patches_left[h][w][5] = getPixel(left, w + 1, h, width, height);
                    patches_left[h][w][6] = getPixel(left, w - 1, h + 1, width, height);
                    patches_left[h][w][7] = getPixel(left, w, h + 1, width, height);
                    patches_left[h][w][8] = getPixel(left, w + 1, h + 1, width, height);

                    patches_right[h][w][0] = getPixel(right, w - 1, h - 1, width, height);
                    patches_right[h][w][1] = getPixel(right, w, h - 1, width, height);
                    patches_right[h][w][2] = getPixel(right, w + 1, h - 1, width, height);
                    patches_right[h][w][3] = getPixel(right, w - 1, h, width, height);
                    patches_right[h][w][4] = getPixel(right, w, h, width, height);
                    patches_right[h][w][5] = getPixel(right, w + 1, h, width, height);
                    patches_right[h][w][6] = getPixel(right, w - 1, h + 1, width, height);
                    patches_right[h][w][7] = getPixel(right, w, h + 1, width, height);
                    patches_right[h][w][8] = getPixel(right, w + 1, h + 1, width, height);
                }
            }
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float sum_left = 0.0;
                    float sum_right = 0.0;
                    for (int p = 0; p < 9; p++) {
                        sum_left += patches_left[h][w][p];
                        sum_right += patches_right[h][w][p];
                    }
                    means_left[h * width + w] = sum_left / 9.0;
                    means_right[h * width + w] = sum_right / 9.0;

                    sum_left = 0.0;
                    sum_right = 0.0;
                    for (int p = 0; p < 9; p++) {
                        sum_left += pow(patches_left[h][w][p] - means_left[h * width + w], 2);
                        sum_right += pow(patches_right[h][w][p] - means_right[h * width + w], 2);
                    }
                    sds_left[h * width + w] = sqrt(sum_left / 9.0);
                    sds_right[h * width + w] = sqrt(sum_right / 9.0);
                }
            }
            float *SNCCMap;

            SNCCMap = new float[width * height * maxDisparity];
            for (int disp = 0; disp < maxDisparity; disp++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        if ((w - disp) >= 0) {
                            float dotProduct = 0.0;
                            for (int p = 0; p < 9; p++) {
                                dotProduct += (patches_left[h][w][p] + patches_right[h][w - disp][p]);
                            }
                            float rho = ((dotProduct / 9.0) / 9.0 - means_left[h * width + w] * means_right[h * width + w - disp]) / (sds_left[h * width + w] * sds_right[h * width + w - disp]);
                            SNCCMap[disp * width * height + h * width + w] = rho;
                        }
                    }
                }
            }
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    delete [] patches_left[i][j];
                    delete [] patches_right[i][j];
                }
                delete [] patches_left[i];
                delete [] patches_right[i];
            }
            delete [] patches_left;
            delete [] patches_right;

            float *SNCCMap2;
            int winAvg2 = int(floor(avgWindow / 2.0));
            float winAvgSize = avgWindow*avgWindow;
            SNCCMap2 = new float[width * height * maxDisparity];
            for (int disp = 0; disp < maxDisparity; disp++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        float sum = 0.0;
                        for (int wy = h - winAvg2; wy <= (h + winAvg2); wy++) {
                            for (int wx = w - winAvg2; wx <= (w + winAvg2); wx++) {
                                sum += getPixel(&SNCCMap[disp * width * height], wx, wy, width, height);
                            }
                        }
                        SNCCMap2[disp * width * height + h * width + w] = sum / winAvgSize;
                    }
                }
            }
            unsigned char *disparity;
            disparity = new unsigned char[width * height];
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float min=-std::numeric_limits<float>::infinity();
                    int dmin=0;
                    for (int disp = 0; disp < maxDisparity; disp++) {
                        if(SNCCMap2[disp * width * height + h * width + w] > min){
                            min=SNCCMap2[disp * width * height + h * width + w];
                            dmin=disp;
                        }
                    }
                    disparity[h*width+w]=dmin;
                }
            }
            int coef=256/maxDisparity;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    disparity[h*width+w]*=coef;
                }
            }
            delete [] SNCCMap;
            delete [] SNCCMap2;
            return disparity;
        }

        std::vector<core::pFrame> SNCC::do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames) {
            converter_left = std::make_shared<core::Convert>(log, get_this_ptr(), core::Convert::configure());
            converter_right = std::make_shared<core::Convert>(log, get_this_ptr(), core::Convert::configure());
            core::pRawVideoFrame left_frame = std::dynamic_pointer_cast<core::RawVideoFrame>(converter_left->convert_to_cheapest(std::get<0>(frames), supported_formats_));
            core::pRawVideoFrame right_frame = std::dynamic_pointer_cast<core::RawVideoFrame>(converter_right->convert_to_cheapest(std::get<1>(frames), supported_formats_));
            size_t w = left_frame->get_width();
            size_t h = left_frame->get_height();
            unsigned char* disparity=computeDisparity(PLANE_RAW_DATA(left_frame,0),PLANE_RAW_DATA(right_frame,0),w,h,num_disparities,7);
            core::pRawVideoFrame output = core::RawVideoFrame::create_empty(core::raw_format::y8,{static_cast<dimension_t> (w), static_cast<dimension_t> (h)},
            disparity,w * h * sizeof (unsigned char));
            return {output};
        }

        bool SNCC::set_param(const core::Parameter& param) {
            if (assign_parameters(param)
                    (num_disparities, "num_disparities"))
                return true;
            return core::MultiIOFilter::set_param(param);
        }

        SNCC::~SNCC() noexcept {
        }
    }
}
