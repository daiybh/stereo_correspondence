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

#include "OpenCVCudaBM.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "opencv2/imgproc.hpp"
namespace yuri{
namespace opencvcudabm{
    
IOTHREAD_GENERATOR(OpenCVCudaBM)
    MODULE_REGISTRATION_BEGIN("opencv_cudabm")
    REGISTER_IOTHREAD("opencv_cudabm",OpenCVCudaBM)
    MODULE_REGISTRATION_END()
OpenCVCudaBM::OpenCVCudaBM(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters):
core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>(log_, parent, 2, std::string("opencv_cudabm")){
    IOTHREAD_INIT(parameters)
    //set_supported_formats({core::raw_format::rgba32});
    
    bm = cv::cuda::createStereoBM(num_disparities,window_size);
}

core::Parameters OpenCVCudaBM::configure(){
    core::Parameters p = base_type::configure();
    p.set_description("OpenCV Cuda BM");
    p["num_disparites"]["Number of disparities"]=16;
    p["window_size"]["Window size"]=3;
    return p;
}

std::vector<core::pFrame> OpenCVCudaBM::do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames){
    core::pRawVideoFrame left_frame = std::get<0>(frames);
    core::pRawVideoFrame right_frame = std::get<1>(frames);
    const size_t width = left_frame->get_width();
    const size_t height = left_frame->get_height();
    cv::Mat left_mat,right_mat,left,right;
    cv::cuda::GpuMat d_left,d_right;
    convert = std::make_shared<core::Convert>(log, get_this_ptr(), core::Convert::configure());
    left_mat=cv::Mat(height,width,CV_8UC3,PLANE_RAW_DATA(left_frame,0));
    right_mat=cv::Mat(height,width,CV_8UC3,PLANE_RAW_DATA(right_frame,0));
    cv::cvtColor(left_mat, left, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_mat, right, cv::COLOR_BGR2GRAY);
    cv::Mat disp8(left_mat.size(), CV_8U);
    cv::cuda::GpuMat d_disp(left_mat.size(), CV_8U);
    d_left.upload(left);
    d_right.upload(right);
    bm->compute(d_left,d_right,d_disp);
    d_disp.download(disp8);
    cv::Mat out;
    cv::medianBlur(disp8,out,5);
    int coef=256/num_disparities;
    out = coef*out;
    core::pRawVideoFrame map = core::RawVideoFrame::create_empty(core::raw_format::y8,
                                            {static_cast<dimension_t>(out.cols), static_cast<dimension_t>(out.rows)},
											out.data,
											out.total() * out.elemSize());
    core::pRawVideoFrame output = std::dynamic_pointer_cast<core::RawVideoFrame>(convert->convert_to_cheapest(map, {std::get<0>(frames)->get_format()}));
    return {output,left_frame};
}

bool OpenCVCudaBM::set_param(const core::Parameter& param){
    if (assign_parameters(param)
                        (num_disparities,"num_disparities")
                        (window_size,"window_size"))
		return true;
    return core::MultiIOFilter::set_param(param);
}

OpenCVCudaBM::~OpenCVCudaBM() noexcept{
}
}
}
