/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   OpenCVSGBM.h
 * Author: user
 *
 * Created on 14. února 2016, 13:32
 */

#ifndef OPENCVSGBM_H
#define OPENCVSGBM_H

#include "yuri/core/thread/SpecializedMultiIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"

namespace yuri{
namespace opencvsgbm{
class OpenCVSGBM: public core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>{
    using base_type = core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>;
public:
    IOTHREAD_GENERATOR_DECLARATION
    static core::Parameters configure();
    OpenCVSGBM(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
    virtual ~OpenCVSGBM() noexcept;
private:
    virtual std::vector<core::pFrame> do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames) override;
    virtual bool set_param(const core::Parameter& param) override;
    cv::Ptr<cv::StereoSGBM> sgbm;
};
}
}

#endif /* OPENCVSGBM_H */

