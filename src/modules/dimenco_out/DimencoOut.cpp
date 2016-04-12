/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DimencoOut.cpp
 * Author: user
 * 
 * Created on 12. dubna 2016, 13:50
 */

#include "DimencoOut.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/RawVideoFrame.h"

namespace yuri{
namespace dimencoout{

    IOTHREAD_GENERATOR(DimencoOut)

MODULE_REGISTRATION_BEGIN("dimenco_out")
		REGISTER_IOTHREAD("dimenco_out",DimencoOut)
MODULE_REGISTRATION_END()
            
core::Parameters DimencoOut::configure(){
    core::Parameters p = base_type::configure();
    p.set_description("Output to Dimenco display");
    return p;
}

DimencoOut::~DimencoOut(){
    
}

DimencoOut::DimencoOut(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("dimenco_out")){
    IOTHREAD_INIT(parameters)
    set_supported_formats({core::raw_format::rgb24});
}

core::pFrame DimencoOut::do_special_single_step(core::pRawVideoFrame frame){
    int height = frame->get_height();
    int width = frame->get_width();
    return frame;
}

bool DimencoOut::set_param(const core::Parameter& param){
    if (assign_parameters(param))
		return true;
    return base_type::set_param(param);
}
    
}
}
