/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   OpenCVSGBM.cpp
 * Author: user
 * 
 * Created on 14. února 2016, 13:32
 */

#include "CudaASW.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"

namespace yuri{
namespace cudaasw{
    
IOTHREAD_GENERATOR(CudaASW)
    MODULE_REGISTRATION_BEGIN("cuda_asw")
    REGISTER_IOTHREAD("cuda_asw",CudaASW)
    MODULE_REGISTRATION_END()
CudaASW::CudaASW(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters):
core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>(log_, parent, 1, std::string("cuda_asw")){
    IOTHREAD_INIT(parameters)
    //set_supported_formats({core::raw_format::rgba32});
}

core::Parameters CudaASW::configure(){
    core::Parameters p = base_type::configure();
    p.set_description("Cuda Iterative ASW");
    return p;
}

std::vector<core::pFrame> CudaASW::do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames){
    core::pRawVideoFrame output = std::get<0>(frames);
    
    return {output};
}

bool CudaASW::set_param(const core::Parameter& param){
    
    return core::MultiIOFilter::set_param(param);
}

CudaASW::~CudaASW() noexcept{
}
}
}
