/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   CudaASW.h
 * Author: user
 *
 * Created on 26. února 2016, 12:20
 */

#ifndef CUDAASW_H
#define CUDAASW_H

#include "yuri/core/thread/SpecializedMultiIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"

namespace yuri{
namespace cudaasw{
class CudaASW: public core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>{
    using base_type = core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>;
public:
    IOTHREAD_GENERATOR_DECLARATION
    static core::Parameters configure();
    CudaASW(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
    virtual ~CudaASW() noexcept;
private:
    virtual std::vector<core::pFrame> do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames) override;
    virtual bool set_param(const core::Parameter& param) override;
    int num_disparities;
    int window_size;
};
}
}

#endif /* CUDAASW_H */

