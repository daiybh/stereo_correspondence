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
#include <opencv2/imgproc.hpp>

#include "OpenCVStereoCalib.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "yuri/core/frame/RawAudioFrame.h"

namespace yuri{
namespace opencvstereocalib{
    IOTHREAD_GENERATOR(OpenCVStereoCalib)
    MODULE_REGISTRATION_BEGIN("opencv_stereo_calib")
    REGISTER_IOTHREAD("opencv_stereo_calib",OpenCVStereoCalib)
    MODULE_REGISTRATION_END()
            
core::Parameters OpenCVStereoCalib::configure(){
    core::Parameters p = base_type::configure();
    p.set_description("OpenCV Stereo Calibration");
    p["calibration_frames"]["Number of frames to calibrate with"]=10;
    p["frame_delay"]["Every n-th frame will be saved"]=10;
    p["chessboard_x"]["Number of horizontal corners"]=9;
    p["chessboard_y"]["Number of vertical corners"]=6;
    p["path"]["Path to which export the files"]="./";
    return p;
}
    
OpenCVStereoCalib::OpenCVStereoCalib(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters):
core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>(log_, parent, 2, std::string("opencv_stereo_calib")),
calibrated(false),frames_processed(0),path("./"){
    IOTHREAD_INIT(parameters)
    //set_supported_formats({core::raw_format::rgba32});
    log[log::info]<< "Target pairs: "<<target_pairs;
}

OpenCVStereoCalib::~OpenCVStereoCalib()noexcept{
    
}

std::vector<core::pFrame> OpenCVStereoCalib::do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames){
//    log[log::info]  << std::get<0>(frames)->get_format();
    core::pRawVideoFrame left_frame = std::get<0>(frames);
    core::pRawVideoFrame right_frame = std::get<1>(frames);
    
    const size_t width = left_frame->get_width();
    const size_t height = left_frame->get_height();
    //log[log::info] << width;
    cv::Mat left_mat;
    cv::Mat right_mat;

    if(left_frame->get_format() == core::raw_format::yuyv422){
        log[log::debug]<< "Converting";
        cv::Mat left_yuv(height,width,CV_8UC2,PLANE_RAW_DATA(left_frame,0));
        cv::Mat right_yuv(height,width,CV_8UC2,PLANE_RAW_DATA(right_frame,0));
        cv::cvtColor(left_yuv,left_mat,CV_YUV2GRAY_YUYV);
        cv::cvtColor(right_yuv,right_mat,CV_YUV2GRAY_YUYV);
    }else{
        //log[log::debug]<< "Not converting";
        cv::Mat left_rgb=cv::Mat(height,width,CV_8UC3,PLANE_RAW_DATA(left_frame,0));
        cv::Mat right_rgb=cv::Mat(height,width,CV_8UC3,PLANE_RAW_DATA(right_frame,0));
        cv::cvtColor(left_rgb, left_mat, CV_RGB2GRAY);
        cv::cvtColor(right_rgb, right_mat, CV_RGB2GRAY);
    }
    
    std::vector<cv::Point2f> left_corners;
    std::vector<cv::Point2f> right_corners;
    bool found_left = cv::findChessboardCorners(left_mat,cv::Size(chessboard_x,chessboard_y),left_corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    bool found_right = cv::findChessboardCorners(right_mat,cv::Size(chessboard_x,chessboard_y),right_corners,cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    //log[log::info] << found << " Found :" <<corners.size()<<" corners";
    
    if(found_left && found_right && !calibrated && left_found_points.size() < target_pairs && (frames_processed%frame_delay)==0){
        cv::cornerSubPix(left_mat, left_corners, cv::Size(11,11), cv::Size(-1,-1),
                         cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                      30, 0.01));
        cv::cornerSubPix(right_mat, right_corners, cv::Size(11,11), cv::Size(-1,-1),
                         cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                      30, 0.01));
        left_found_points.push_back(left_corners);
        right_found_points.push_back(right_corners);
        log[log::info] << "Storing corners - "<<left_found_points.size() ;
    }
    if(left_found_points.size() == target_pairs && !calibrated){
        log[log::info] << "Calibrating now...";
        cv::Size imageSize(left_mat.cols,left_mat.rows);
        calibrate(imageSize);
        log[log::info] << "Calibration finished";
        calibrated = true;
    }
    cv::Mat out_left;
    cv::Mat out_right;
    cv::cvtColor(left_mat,out_left,CV_GRAY2RGB);
    cv::cvtColor(right_mat,out_right,CV_GRAY2RGB);
    if(found_left && found_right){
        cv::drawChessboardCorners(out_left,cv::Size(chessboard_x,chessboard_y),left_corners,found_left);
        cv::drawChessboardCorners(out_right,cv::Size(chessboard_x,chessboard_y),right_corners,found_right);
    }
    
    core::pRawVideoFrame output_left = core::RawVideoFrame::create_empty(core::raw_format::rgb24,
                                            {static_cast<dimension_t>(out_left.cols), static_cast<dimension_t>(out_left.rows)},
											out_left.data,
											out_left.total() * out_left.elemSize());
    core::pRawVideoFrame output_right = core::RawVideoFrame::create_empty(core::raw_format::rgb24,
                                            {static_cast<dimension_t>(out_right.cols), static_cast<dimension_t>(out_right.rows)},
											out_right.data,
											out_right.total() * out_right.elemSize());
    frames_processed++;
    return {output_left, output_right};
}

void OpenCVStereoCalib::calibrate(cv::Size imageSize){
    std::vector<std::vector<cv::Point3f> > objectPoints;
    cv::Size boardSize(chessboard_x,chessboard_y);
    objectPoints.resize(target_pairs);
    for(unsigned int i=0;i<target_pairs;i++){
        for(int j=0;j<boardSize.height;j++)
            for(int k=0;k<boardSize.width;k++)
                objectPoints[i].push_back(cv::Point3f(k*1.0, j*1.0, 0));
    }
    cv::Mat leftCameraMatrix,rightCameraMatrix,leftDistCoefs,rightDistCoefs;
    leftCameraMatrix = cv::initCameraMatrix2D(objectPoints,left_found_points,imageSize,0);
    rightCameraMatrix = cv::initCameraMatrix2D(objectPoints,right_found_points,imageSize,0);
    cv::Mat R, T, E, F;
    
    double rms = stereoCalibrate(objectPoints, left_found_points, right_found_points,
                    leftCameraMatrix, leftDistCoefs,
                    rightCameraMatrix, rightDistCoefs,
                    imageSize, R, T, E, F,
                    cv::CALIB_FIX_ASPECT_RATIO +
                    cv::CALIB_ZERO_TANGENT_DIST +
                    cv::CALIB_USE_INTRINSIC_GUESS +
                    cv::CALIB_SAME_FOCAL_LENGTH +
                    cv::CALIB_RATIONAL_MODEL +
                    cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5,
                    cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 100, 1e-5) );
    log[log::info] << "Error: " << rms;
    
    cv::FileStorage fs(path+"intrinsics.yml", cv::FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << leftCameraMatrix << "D1" << leftDistCoefs <<
            "M2" << rightCameraMatrix << "D2" << rightDistCoefs;
        fs.release();
    }
    else
        log[log::error] << "Error: can not save the intrinsic parameters\n";
    
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validRoi[2];

    stereoRectify(leftCameraMatrix, leftDistCoefs,
                  rightCameraMatrix, rightDistCoefs,
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  cv::CALIB_ZERO_DISPARITY, 0, imageSize, &validRoi[0], &validRoi[1]);

    fs.open(path+"extrinsics.yml", cv::FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        log[log::error] << "Error: can not save the extrinsic parameters\n";
    
    cv::Mat leftUndistortMap[2];
    cv::Mat rightUndistortMap[2];
    initUndistortRectifyMap(leftCameraMatrix, leftDistCoefs, R1, P1, imageSize, CV_16SC2, leftUndistortMap[0], leftUndistortMap[1]);
    initUndistortRectifyMap(rightCameraMatrix, rightDistCoefs, R2, P2, imageSize, CV_16SC2, rightUndistortMap[0], rightUndistortMap[1]);
    
    fs.open(path+"undistort.yml", cv::FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "L1" << leftUndistortMap[0] << "L2" << leftUndistortMap[1] << "R1" << rightUndistortMap[0] << "R2" << rightUndistortMap[1];
        fs.release();
    }
    else
        log[log::error] << "Error: can not save the undistort maps\n";
}

bool OpenCVStereoCalib::set_param(const core::Parameter& param){
    if (assign_parameters(param)
			(target_pairs,"calibration_frames")
                        (chessboard_x,"chessboard_x")
                        (chessboard_y,"chessboard_y")
                        (frame_delay,"frame_delay")
                        (path,"path"))
		return true;
    return core::MultiIOFilter::set_param(param);
}
}
}
