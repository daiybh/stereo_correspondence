/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   SNCC.h
 * Author: user
 *
 * Created on 4. dubna 2016, 10:52
 */

#ifndef CUDASNCC_H
#define CUDASNCC_H

#include "yuri/core/thread/SpecializedMultiIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/thread/Convert.h"

namespace yuri{
namespace cuda_sncc{
class CudaSNCC: public core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>{
    using base_type = core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>;
public:
    IOTHREAD_GENERATOR_DECLARATION
    static core::Parameters configure();
    CudaSNCC(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
    virtual ~CudaSNCC() noexcept;
private:
    virtual std::vector<core::pFrame> do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames) override;
    virtual bool set_param(const core::Parameter& param) override;
    std::shared_ptr<core::Convert> converter_left;
    std::shared_ptr<core::Convert> converter_right;
    std::vector<format_t>	supported_formats_;
    int num_disparities;
    int filter_width;
    int filter_height;
};
}
}

#endif /* SNCC_H */

