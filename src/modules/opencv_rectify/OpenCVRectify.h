/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   OpenCVRectify.h
 * Author: user
 *
 * Created on 11. února 2016, 11:38
 */

#ifndef OPENCVRECTIFY_H
#define OPENCVRECTIFY_H
#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace yuri{
namespace opencvrectify{
class OpenCVRectify: public core::SpecializedIOFilter<core::RawVideoFrame> {
    using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
public:
    IOTHREAD_GENERATOR_DECLARATION
    static core::Parameters configure();
    OpenCVRectify(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
    virtual ~OpenCVRectify() noexcept;
private:
    virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
    virtual bool set_param(const core::Parameter& param) override;
    std::string map_path;
    bool left;
    cv::Mat mat1;
    cv::Mat mat2;
};
}
}
#endif /* OPENCVRECTIFY_H */

