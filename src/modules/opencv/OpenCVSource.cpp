/*
 * OpenCVSource.cpp
 *
 *  Created on: 1. 2. 2015
 *      Author: neneko
 */

#include "OpenCVSource.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/raw_frame_types.h"
namespace yuri {
namespace opencv {

core::Parameters OpenCVSource::configure()
{
	auto p = IOThread::configure();
	p.set_description("Source from OpenCV ViceoCapture");
	p["index"]["Camera index"]=0;
        p["width"]["Capture width"]=640;
        p["height"]["Capture height"]=480;
	p["path"]["Source path. Overrides device index, if specified."]="";
	return p;
}

IOTHREAD_GENERATOR(OpenCVSource)

OpenCVSource::OpenCVSource(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters)
:IOThread(log_, parent, 0, 1, "opencv_source"),device_index_(0)
{
	IOTHREAD_INIT(parameters)
        
	if (device_path_.empty()) capture_.open(device_index_);
	else capture_.open(device_path_);
        capture_.set(CV_CAP_PROP_FRAME_HEIGHT,height);
        capture_.set(CV_CAP_PROP_FRAME_WIDTH,width);
	if (!capture_.isOpened()) {
		throw exception::InitializationFailed("Failed to open video device");
	}
}

OpenCVSource::~OpenCVSource() noexcept
{

}

void OpenCVSource::run()
{
	while (still_running()) {
		cv::Mat mat;
		capture_ >> mat;
		if (mat.isContinuous()) {
			auto frame = core::RawVideoFrame::create_empty(core::raw_format::bgr24,
					{static_cast<dimension_t>(mat.cols), static_cast<dimension_t>(mat.rows)},
					mat.data,
					mat.total() * mat.elemSize());
			push_frame(0, frame);
		}
	}
}
bool OpenCVSource::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(device_index_, "index")
			(device_path_, "path")
                        (height,"height")
                        (width,"width"))
		return true;
	return IOThread::set_param(param);
}


}
}

