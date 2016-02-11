/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   OpenCVRectify.cpp
 * Author: user
 * 
 * Created on 11. února 2016, 11:38
 */

#include "OpenCVRectify.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
namespace yuri{
namespace opencvrectify{
    
IOTHREAD_GENERATOR(OpenCVRectify)

MODULE_REGISTRATION_BEGIN("opencv_rectify")
		REGISTER_IOTHREAD("opencv_rectify",OpenCVRectify)
MODULE_REGISTRATION_END()

core::Parameters OpenCVRectify::configure(){
    core::Parameters p = base_type::configure();
    p.set_description("OpenCV Rectification");
    p["map_file"]["Path to file with undistortion maps"]="./undistort.yml";
    return p;
}
OpenCVRectify::OpenCVRectify(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("magnify")){
    IOTHREAD_INIT(parameters)
    set_supported_formats({core::raw_format::bgr24});
}


}
}
