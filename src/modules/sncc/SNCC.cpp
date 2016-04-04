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

namespace yuri{
namespace sncc{
    
IOTHREAD_GENERATOR(SNCC)
    MODULE_REGISTRATION_BEGIN("sncc")
    REGISTER_IOTHREAD("sncc",SNCC)
    MODULE_REGISTRATION_END()
SNCC::SNCC(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters):
core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>(log_, parent, 2, std::string("sncc")){
    IOTHREAD_INIT(parameters)
    //set_supported_formats({core::raw_format::rgba32});
}

core::Parameters SNCC::configure(){
    core::Parameters p = base_type::configure();
    p.set_description("SNCC Disparity computation");
    return p;
}

std::vector<core::pFrame> SNCC::do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames){
    return {std::get<0>(frames)};
}

bool SNCC::set_param(const core::Parameter& param){
    if (assign_parameters(param))
		return true;
    return core::MultiIOFilter::set_param(param);
}

SNCC::~SNCC() noexcept{
}
}
}
