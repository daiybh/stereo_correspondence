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
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
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
    p["left"]["True if frame from the left camera, false if from the right one"]=true;
    return p;
}

OpenCVRectify::~OpenCVRectify(){
    
}

OpenCVRectify::OpenCVRectify(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("opencv_rectify")){
    IOTHREAD_INIT(parameters)
    set_supported_formats({core::raw_format::bgr24});
    
    cv::FileStorage fs;
    fs.open(map_path, cv::FileStorage::READ);
    if(left){
        fs["L1"] >> mat1;
        fs["L2"] >> mat2;
    }else{
        fs["R1"] >> mat1;
        fs["R2"] >> mat2;
    }
    fs.release();
    log[log::info] << "Map loaded, width: "<<mat1.cols; 
    
}

core::pFrame OpenCVRectify::do_special_single_step(core::pRawVideoFrame frame){
    int height = frame->get_height();
    int width = frame->get_width();
    cv::Mat frame_mat(height,width,CV_8UC3,PLANE_RAW_DATA(frame,0));
    
    cv::Mat new_mat;
    cv::remap(frame_mat, new_mat, mat1, mat2, cv::INTER_LINEAR);
    log[log::info]<< new_mat.rows;
    
    core::pRawVideoFrame output = core::RawVideoFrame::create_empty(core::raw_format::bgr24,
                                            {static_cast<dimension_t>(new_mat.cols), static_cast<dimension_t>(new_mat.rows)},
											new_mat.data,
											new_mat.total() * new_mat.elemSize());
    return output;
}

bool OpenCVRectify::set_param(const core::Parameter& param){
    if (assign_parameters(param)
			(map_path,"map_file")
                        (left,"left"))
		return true;
    return base_type::set_param(param);
}

}
}
