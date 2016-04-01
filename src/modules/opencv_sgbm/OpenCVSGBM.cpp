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

#include <opencv2/calib3d.hpp>

#include "OpenCVSGBM.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
namespace yuri{
namespace opencvsgbm{
    
IOTHREAD_GENERATOR(OpenCVSGBM)
    MODULE_REGISTRATION_BEGIN("opencv_sgbm")
    REGISTER_IOTHREAD("opencv_sgbm",OpenCVSGBM)
    MODULE_REGISTRATION_END()
OpenCVSGBM::OpenCVSGBM(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters):
core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>(log_, parent, 1, std::string("opencv_sgbm")){
    IOTHREAD_INIT(parameters)
    //set_supported_formats({core::raw_format::rgba32});
    sgbm = cv::StereoSGBM::create(min_disparity,num_disparities,window_size);
    sgbm->setP1(8*3*window_size*window_size);
    sgbm->setP2(32*3*window_size*window_size);
    sgbm->setMinDisparity(min_disparity);
    sgbm->setNumDisparities(num_disparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(300);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
}

core::Parameters OpenCVSGBM::configure(){
    core::Parameters p = base_type::configure();
    p.set_description("OpenCV SGBM");
    p["min_disparity"]["Min disparity"]=0;
    p["num_disparities"]["Number of disparities"]=16;
    p["window_size"]["Window size"]=3;
    
    return p;
}

std::vector<core::pFrame> OpenCVSGBM::do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames){
    core::pRawVideoFrame left_frame = std::get<0>(frames);
    core::pRawVideoFrame right_frame = std::get<1>(frames);
    const size_t width = left_frame->get_width();
    const size_t height = left_frame->get_height();
    cv::Mat disp, disp8,left_mat,right_mat;
    
    left_mat=cv::Mat(height,width,CV_8UC3,PLANE_RAW_DATA(left_frame,0));
    right_mat=cv::Mat(height,width,CV_8UC3,PLANE_RAW_DATA(right_frame,0));
    
    sgbm->compute(left_mat,right_mat,disp);
    disp.convertTo(disp8, CV_8U);
    core::pRawVideoFrame output = core::RawVideoFrame::create_empty(core::raw_format::g8,
                                            {static_cast<dimension_t>(disp8.cols), static_cast<dimension_t>(disp8.rows)},
											disp8.data,
											disp8.total() * disp8.elemSize());
    return {output};
}

bool OpenCVSGBM::set_param(const core::Parameter& param){
    if (assign_parameters(param)
			(min_disparity,"min_disparity")
                        (num_disparities,"num_disparities")
                        (window_size,"window_size"))
		return true;
    return core::MultiIOFilter::set_param(param);
}

OpenCVSGBM::~OpenCVSGBM() noexcept{
}
}
}