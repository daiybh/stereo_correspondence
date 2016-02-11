/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   OpenCVStereoCalib.h
 * Author: user
 *
 * Created on 9. únor 2016, 15:18
 */

#ifndef OPENCVSTEREOCALIB_H
#define OPENCVSTEREOCALIB_H

#include "yuri/core/thread/SpecializedMultiIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
namespace yuri{
namespace opencvstereocalib{
class OpenCVStereoCalib: public core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>{
    using base_type = core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>;
public:
    IOTHREAD_GENERATOR_DECLARATION
    static core::Parameters configure();
    OpenCVStereoCalib(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
    virtual ~OpenCVStereoCalib() noexcept;
private:
    virtual std::vector<core::pFrame> do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames) override;
    virtual bool set_param(const core::Parameter& param) override;
    void calibrate(cv::Size imageSize);
    bool calibrated;
    std::vector<std::vector<cv::Point2f>> left_found_points;
    std::vector<std::vector<cv::Point2f>> right_found_points;
    int frames_processed;
    unsigned int target_pairs;
};
}
}
#endif /* OPENCVSTEREOCALIB_H */

