/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   OpenCVStereoCalib.cpp
 * Author: user
 * 
 * Created on 9. únor 2016, 15:18
 */

#include <opencv2/calib3d.hpp>

#include "OpenCVStereoCalib.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "opencv2/imgproc/imgproc.hpp"

namespace yuri{
namespace opencvstereocalib{
    IOTHREAD_GENERATOR(OpenCVStereoCalib)
    MODULE_REGISTRATION_BEGIN("opencv_stereo_calib")
    REGISTER_IOTHREAD("opencv_stereo_calib",OpenCVStereoCalib)
    MODULE_REGISTRATION_END()
            
core::Parameters OpenCVStereoCalib::configure(){
    core::Parameters p = base_type::configure();
    p.set_description("OpenCV Stereo Calibration");
    return p;
}
    
OpenCVStereoCalib::OpenCVStereoCalib(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters):
base_type(log_,parent,std::string("magnify")){
    IOTHREAD_INIT(parameters)
    set_supported_formats({core::raw_format::rgba32});
}

OpenCVStereoCalib::~OpenCVStereoCalib()noexcept{
    
}

core::pFrame OpenCVStereoCalib::do_special_single_step(core::pRawVideoFrame frame){
    const size_t width = frame->get_width();
    const size_t height = frame->get_height();
    cv::Mat in_mat(height,width,CV_8UC4,PLANE_RAW_DATA(frame,0));
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(in_mat,cv::Size(7,5),corners);
    cv::drawChessboardCorners(in_mat,cv::Size(7,5),corners,found);
    core::pRawVideoFrame output = core::RawVideoFrame::create_empty(core::raw_format::rgba32,
                                            {static_cast<dimension_t>(in_mat.cols), static_cast<dimension_t>(in_mat.rows)},
											in_mat.data,
											in_mat.total() * in_mat.elemSize());
    return output;
}

bool OpenCVStereoCalib::set_param(const core::Parameter& param){
    return true;
}
}
}
