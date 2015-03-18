/*!
 * @file 		FlyCap.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		02.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "FlyCap.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
namespace yuri {
namespace flycap {


IOTHREAD_GENERATOR(FlyCap)

MODULE_REGISTRATION_BEGIN("flycap")
		REGISTER_IOTHREAD("flycap",FlyCap)
MODULE_REGISTRATION_END()

core::Parameters FlyCap::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("FlyCap");
	p["resolution"]["Capture resolution"]=resolution_t{1280, 960};
	p["format"]["Capture format (Y8, Y16, RGB, YUV)"]="Y8";
	p["fps"]["Capture framerate"]=30;
	p["index"]["Index of camera to use"]=0;
	p["serial"]["Serial number of camera to user (overrides index)"]=0;
	p["keep_format"]["Keep currently format (skips setting of format)"]=false;
	return p;
}

namespace {
using namespace core::raw_format;

struct cmp_resolution {
bool operator()(const resolution_t& a, const resolution_t& b) const
{
	return (a.width < b.width) || ((a.width==b.width) && (a.height < b. height));
}
};
const std::map<resolution_t, std::map<format_t, FlyCapture2::VideoMode>, cmp_resolution> video_modes = {
		{{160,120}, {
				{yuv444, FlyCapture2::VIDEOMODE_160x120YUV444}},
		},
		{{320,240}, {
				{yuyv422, FlyCapture2::VIDEOMODE_320x240YUV422}},
		},
		{{640, 480}, {
				{y8, FlyCapture2::VIDEOMODE_640x480Y8},
				{y16, FlyCapture2::VIDEOMODE_640x480Y16},
				{rgb24, FlyCapture2::VIDEOMODE_640x480RGB},
				{yuv411, FlyCapture2::VIDEOMODE_640x480YUV411},
				{yuyv422, FlyCapture2::VIDEOMODE_640x480YUV422}},
		},
		{{800, 600}, {
				{y8, FlyCapture2::VIDEOMODE_800x600Y8},
				{y16, FlyCapture2::VIDEOMODE_800x600Y16},
				{rgb24, FlyCapture2::VIDEOMODE_800x600RGB},
				{yuyv422, FlyCapture2::VIDEOMODE_800x600YUV422}},
		},
		{{1024, 768}, {
				{y8, FlyCapture2::VIDEOMODE_1024x768Y8},
				{y16, FlyCapture2::VIDEOMODE_1024x768Y16},
				{rgb24, FlyCapture2::VIDEOMODE_1024x768RGB},
				{yuyv422, FlyCapture2::VIDEOMODE_1024x768YUV422}},
		},
		{{1280, 960}, {
				{y8, FlyCapture2::VIDEOMODE_1280x960Y8},
				{y16, FlyCapture2::VIDEOMODE_1280x960Y16},
				{rgb24, FlyCapture2::VIDEOMODE_1280x960RGB},
				{yuyv422, FlyCapture2::VIDEOMODE_1280x960YUV422}},
		},
		{{1600, 1200}, {
				{y8, FlyCapture2::VIDEOMODE_1600x1200Y8},
				{y16, FlyCapture2::VIDEOMODE_1600x1200Y16},
				{rgb24, FlyCapture2::VIDEOMODE_1600x1200RGB},
				{yuyv422, FlyCapture2::VIDEOMODE_1600x1200YUV422}},
		},

};

const std::map<size_t, FlyCapture2::FrameRate> frame_rates = {
		{15, FlyCapture2::FRAMERATE_15},
		{30, FlyCapture2::FRAMERATE_30},
		{60, FlyCapture2::FRAMERATE_60},
		{120, FlyCapture2::FRAMERATE_120},
		{240, FlyCapture2::FRAMERATE_240},
};

FlyCapture2::VideoMode get_mode(resolution_t res, format_t fmt)
{
	auto it = video_modes.find(res);
	if (it == video_modes.end()) return FlyCapture2::NUM_VIDEOMODES;
	auto it2 = it->second.find(fmt);
	if (it2 == it->second.end()) return FlyCapture2::NUM_VIDEOMODES;
	return it2->second;
}

FlyCapture2::FrameRate get_fps(size_t fps)
{
	auto it = frame_rates.find(fps);
	if (it == frame_rates.end()) return FlyCapture2::NUM_FRAMERATES;
	return it->second;
}
}


FlyCap::FlyCap(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("flycap")),resolution_(resolution_t{1280,960}),
format_(core::raw_format::y8),fps_(30),index_(0),serial_(0),keep_format_(false),
shutdown_delay_(100_ms)
{
	IOTHREAD_INIT(parameters)

	auto mode = get_mode(resolution_, format_);
	if (mode == FlyCapture2::NUM_VIDEOMODES) {
		throw exception::InitializationFailed("Unsupported video format");
	}
	auto fps = get_fps(fps_);
	if (fps == FlyCapture2::NUM_FRAMERATES) {
		throw exception::InitializationFailed("Unsupported framerate");
	}
	FlyCapture2::BusManager busMgr;
	unsigned int numCameras;
	auto error = busMgr.GetNumOfCameras(&numCameras);
	if (error != FlyCapture2::PGRERROR_OK)
	{
		throw exception::InitializationFailed("Failed to query cameras");
	}

	log[log::info] << "Number of cameras detected: " << numCameras;


	FlyCapture2::PGRGuid guid;
	if (!serial_) {
		error = busMgr.GetCameraFromIndex( index_, &guid );
		if (error != FlyCapture2::PGRERROR_OK)
		{
			throw exception::InitializationFailed("Failed to get camera with index " + std::to_string(index_));
		}
	} else {
		error = busMgr.GetCameraFromSerialNumber( serial_, &guid );
		if (error != FlyCapture2::PGRERROR_OK)
		{
			throw exception::InitializationFailed("Failed to get camera with serial " + std::to_string(serial_));
		}
	}

	// Connect to a camera
	error = camera_.Connect( &guid );
	if (error != FlyCapture2::PGRERROR_OK)
	{
		throw exception::InitializationFailed("Failed to connect to camera");
	}

	FlyCapture2::Format7Info f7info;
	bool sup;
	camera_.GetFormat7Info(&f7info, &sup);
	if (!sup) {
		log[log::info] << "format7 not supported";
	} else {
		log[log::info] << "Format7: max. res.:" << f7info.maxWidth << "x" << f7info.maxHeight;
	}



	// Get the camera information
	FlyCapture2::CameraInfo cam_info;
	error = camera_.GetCameraInfo( &cam_info );
	if (error != FlyCapture2::PGRERROR_OK)
	{
		camera_.Disconnect();
		// It takes a while to shutdown background process in FlyCapture2...
		sleep(shutdown_delay_);
		throw exception::InitializationFailed("Failed to query camera info");
	}

	log[log::info] << "Connected to " << cam_info.modelName << ", from "
			<< cam_info.vendorName << ", serial number: " << cam_info.serialNumber;

	if (!keep_format_) {

		error = camera_.SetVideoModeAndFrameRate(
				mode, fps);
		if (error != FlyCapture2::PGRERROR_OK)
		{
			camera_.Disconnect();
			// It takes a while to shutdown background process in FlyCapture2...
			sleep(shutdown_delay_);
			throw exception::InitializationFailed("Failed to set resolution");
		}
	}
	error = camera_.StartCapture();
	if (error != FlyCapture2::PGRERROR_OK) {
		camera_.Disconnect();
		// It takes a while to shutdown background process in FlyCapture2...
		sleep(shutdown_delay_);
		throw exception::InitializationFailed("Failed to start capture");
	}

}

FlyCap::~FlyCap() noexcept
{
}

void FlyCap::run()
{
	while(still_running()) {
		FlyCapture2::Image image;
		camera_.RetrieveBuffer(&image);
		auto frame = core::RawVideoFrame::create_empty(core::raw_format::y8, resolution_t{1280, 960}, image.GetData(), image.GetDataSize());
		push_frame(0, std::move(frame));
	}
	camera_.StopCapture();
	log[log::info] << "Disconnecting from camera";
	camera_.Disconnect();
	sleep(shutdown_delay_);
}
bool FlyCap::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(resolution_, 	"resolution")
			(fps_,			"fps")
			(serial_,		"serial")
			(index_, 		"index")
			(keep_format_,	"keep_format")
			.parsed<std::string>
				(format_, 	"format", core::raw_format::parse_format))
		return true;

	return core::IOThread::set_param(param);
}

} /* namespace flycap */
} /* namespace yuri */
