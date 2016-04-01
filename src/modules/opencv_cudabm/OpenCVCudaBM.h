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
#include "opencv2/cudastereo.hpp"
#include "yuri/core/thread/Convert.h"

namespace yuri{
namespace opencvcudabm{
class OpenCVCudaBM: public core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>{
    using base_type = core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>;
public:
    IOTHREAD_GENERATOR_DECLARATION
    static core::Parameters configure();
    OpenCVCudaBM(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
    virtual ~OpenCVCudaBM() noexcept;
private:
    virtual std::vector<core::pFrame> do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames) override;
    virtual bool set_param(const core::Parameter& param) override;
    cv::Ptr<cv::cuda::StereoBM> bm;
    std::shared_ptr<core::Convert> convert;
    int num_disparities;
    int window_size;
};
}
}

#endif /* OPENCVSGBM_H */

