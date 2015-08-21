/*!
 * @file 		FlyCap.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		02.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "FlyCap.h"
#include "flycap_defs.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/utils/irange.h"
#include <iostream>
namespace yuri {
namespace flycap {


IOTHREAD_GENERATOR(FlyCap)

core::Parameters FlyCap::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("FlyCap");
	p["resolution"]["Standard capture resolution"]=resolution_t{1280, 960};
	p["geometry"]["Crop geometry (for custom modes)"]=geometry_t{0, 0, 0, 0};
	p["format"]["Capture format (Y8, Y16, RGB, YUV)"]="Y8";
	p["fps"]["Capture framerate"]=30;
	p["index"]["Index of camera to use"]=0;
	p["serial"]["Serial number of camera to user (overrides index)"]=0;
	p["keep_format"]["Keep currently format (skips setting of format)"]=false;
	p["embedded_framecounter"]["Use embedded frame counter"]=false;
	p["custom"]["Use custom mode (set to -1 to use standar modes)"]=-1;

	p["shutter"]["Shutter time (set to 0.0 for automatic value)"]=0.0f;
	p["gain"]["Gain values [dB] (set to negative value for automatic value)"]=-1.0f;
	p["brightness"]["Brightness value (set to negative value for automatic value)"]=-1.0f;
	p["gamma"]["Gamma value (set to negative value for automatic value)"]=-1.0f;
	p["exposure"]["Exposure value [EV] (set to -100 or less for automatic value)"]=-100.0f;

	p["trigger"]["Enable trigger"]=false;
	p["trigger_mode"]["Trigger mode"]=0;
	p["trigger_source"]["Source for trigger"]=0;
	p["gpio0"]["Direction for GPIO0 (0 for input, 1 for output)"]=0;
	p["gpio1"]["Direction for GPIO1 (0 for input, 1 for output)"]=1;
	p["gpio2"]["Direction for GPIO2 (0 for input, 1 for output)"]=0;
	p["gpio3"]["Direction for GPIO3 (0 for input, 1 for output)"]=0;

	p["strobe0"]["Enable strobe for GPIO 0"]=false;
	p["strobe0_polarity"]["Set polarity for GPIO 0 strobe (false/true)"]=false;
	p["strobe0_delay"]["Set delay for GPIO 0 strobe"]=0.0f;
	p["strobe0_duration"]["Set duration for GPIO 0 strobe"]=0.0f;

	p["strobe1"]["Enable strobe for GPIO 1"]=false;
	p["strobe1_polarity"]["Set polarity for GPIO 1 strobe (false/true)"]=false;
	p["strobe1_delay"]["Set delay for GPIO 1 strobe"]=0.0f;
	p["strobe1_duration"]["Set duration for GPIO 1 strobe"]=0.0f;

	p["strobe2"]["Enable strobe for GPIO 2"]=false;
	p["strobe2_polarity"]["Set polarity for GPIO 2 strobe (false/true)"]=false;
	p["strobe2_delay"]["Set delay for GPIO 2 strobe"]=0.0f;
	p["strobe2_duration"]["Set duration for GPIO 2 strobe"]=0.0f;

	p["strobe3"]["Enable strobe for GPIO 3"]=false;
	p["strobe3_polarity"]["Set polarity for GPIO 3 strobe (false/true)"]=false;
	p["strobe3_delay"]["Set delay for GPIO 3 strobe"]=0.0f;
	p["strobe3_duration"]["Set duration for GPIO 3 strobe"]=0.0f;

	return p;
}


namespace {
inline void flycap_init_fatal(fc2Error code, const std::string& msg)
{
	if (code != FC2_ERROR_OK) {
		throw exception::InitializationFailed(msg);
	}
}
inline void flycap_init_warn(fc2Error code, log::Log& log, const std::string& msg)
{
	if (code != FC2_ERROR_OK) {
		log[log::warning] << msg;
	}
}

inline void set_flycap_prop(flycap_camera_t& ctx, log::Log& log, const std::string& name, fc2PropertyType ptype, bool autovalue, float value = 0.0f)
{
	fc2Property prop;
	prop.type = ptype;
	flycap_init_warn(fc2GetProperty(ctx, &prop), log, "Failed to query "+name+" info");
	if (autovalue) {
		prop.absControl = true;
		prop.autoManualMode = false;
		prop.onOff = true;
		prop.absValue = value;
	} else {
		prop.autoManualMode = true;
	}
	flycap_init_warn(fc2SetProperty(ctx, &prop), log, "Failed to set " + name);

}
}

FlyCap::FlyCap(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("flycap")),resolution_(resolution_t{1280,960}),
format_(core::raw_format::y8),fps_(30),index_(0),serial_(0),keep_format_(false),
embedded_framecounter_(false),custom_(-1),shutter_time_(0.0f),gain_(-1.0f),brightness_(-1.0f),gamma_(-1.0f),
exposure_(-100.0f), 
#if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 7))
strobes_({false, false, false, false}),
polarities_({false, false, false, false}),delays_({0.0f, 0.0f, 0.0f, 0.0f}),
durations_({0.0f, 0.0f, 0.0f, 0.0f})
#endif
{
	IOTHREAD_INIT(parameters)

	unsigned int numCameras = 0;
	flycap_init_warn(fc2GetNumOfCameras(ctx_, &numCameras), log, "Failed to query available cameras");
	log[log::info] << "Number of cameras detected: " << numCameras;

	ctx_.connect(index_, serial_);
	auto cam_info = ctx_.get_camera_info();

	log[log::info] << "Connected to " << cam_info.modelName << ", from "
	<< cam_info.vendorName << ", serial number: " << cam_info.serialNumber;

	if (!keep_format_) {
		if (custom_ < 0) {
			const auto mode = get_mode(resolution_, format_);
			const auto fps = get_fps(fps_);
			if (mode == FC2_NUM_VIDEOMODES || fps == FC2_NUM_FRAMERATES) {
				throw exception::InitializationFailed("Unknown video format");
			}

			flycap_init_fatal(fc2SetVideoModeAndFrameRate(ctx_,
					mode, fps), "Failed to set resolution");

		} else {
			fc2Format7Info f7info;
			BOOL supported = false;;
			f7info.mode = static_cast<fc2Mode>(custom_);
			flycap_init_fatal(fc2GetFormat7Info(ctx_, &f7info, &supported), "Failed to query format7info");
			if (!supported) {
				throw exception::InitializationFailed("Custom mode " + std::to_string(custom_) + " not supported" );
			}
			fc2Format7ImageSettings f7cfg;
			if (!geometry_) {
				f7cfg.offsetX = 0;
				f7cfg.offsetY = 0;
				f7cfg.width = f7info.maxWidth;
				f7cfg.height= f7info.maxHeight;
			} else {
				f7cfg.offsetX = std::max(0L, geometry_.x);
				f7cfg.offsetY = std::max(0L, geometry_.y);
				f7cfg.width = std::min<unsigned>(f7info.maxWidth, geometry_.width);
				f7cfg.height= std::min<unsigned>(f7info.maxHeight, geometry_.height);
			}
			f7cfg.mode=static_cast<fc2Mode>(custom_);
			f7cfg.pixelFormat=get_fc_format(format_);
			flycap_init_fatal(fc2SetFormat7Configuration(ctx_, &f7cfg, 100),
					"Failed to set custom mode");

			set_flycap_prop(ctx_, log, "framerate", FC2_FRAME_RATE, fps_ > 0.0f, fps_);
			set_flycap_prop(ctx_, log, "shutter", FC2_SHUTTER, shutter_time_ > 0.0f, shutter_time_);
			set_flycap_prop(ctx_, log, "gain", FC2_GAIN, gain_ >= 0.0f, gain_);
			set_flycap_prop(ctx_, log, "gain", FC2_BRIGHTNESS, brightness_ >= 0.0f, brightness_);
			set_flycap_prop(ctx_, log, "gain", FC2_GAMMA, gamma_ >= 0.0f, gamma_);
			set_flycap_prop(ctx_, log, "gain", FC2_AUTO_EXPOSURE, exposure_ > -100.0f, exposure_);


		}
		fc2TriggerMode trig;
		flycap_init_warn(fc2GetTriggerMode(ctx_, &trig), log, "Failed to query trigger mode");
		if (trigger_) {
			trig.onOff=true;
			trig.mode=trigger_mode_;
			trig.source=trigger_source_;
			flycap_init_warn(fc2SetTriggerMode(ctx_, &trig), log, "Failed to set trigger mode");
			sleep(10_ms);
			trig.onOff=false;
			flycap_init_warn(fc2SetTriggerMode(ctx_, &trig), log, "Failed to set trigger mode");
			trig.onOff=true;
			sleep(10_ms);
		} else {
			trig.onOff=false;
		}
		flycap_init_warn(fc2SetTriggerMode(ctx_, &trig), log, "Failed to set trigger mode");

		for (auto i: irange(0,4)) {
			flycap_init_warn(fc2SetGPIOPinDirection(ctx_, i, gpio_directions_[i]), log, "Failed to set GPIO direction for GPIO"+std::to_string(i));
		}

		for (auto i: irange(0,4)) {
			fc2StrobeControl strobe;
			strobe.source=i;
			fc2GetStrobe(ctx_, &strobe);
			strobe.onOff = strobes_[i]?1:0;
			strobe.delay = delays_[i];
			strobe.polarity = polarities_[i]?1:0;
			strobe.duration = durations_[i];
			flycap_init_warn(fc2SetStrobe(ctx_, &strobe), log, "Failed to set strobe for pin GPIO"+std::to_string(i));
		}
	} else {
		log[log::info] << "Keeping current format";
	}
	
	fc2EmbeddedImageInfo einfo;
	einfo.timestamp.onOff=false;
	einfo.timestamp.onOff=false;
	einfo.gain.onOff=false;
	einfo.shutter.onOff=false;
	einfo.brightness.onOff=false;
	einfo.exposure.onOff=false;
	einfo.whiteBalance.onOff=false;
	einfo.frameCounter.onOff=false;
	einfo.strobePattern.onOff=false;
	einfo.GPIOPinState.onOff=false;
	einfo.ROIPosition.onOff=false;
	if (embedded_framecounter_) {
		einfo.frameCounter.onOff=true;
	}
	flycap_init_warn(fc2SetEmbeddedImageInfo(ctx_, &einfo), log, "Failed to set embedded image info");

	ctx_.start();

}

FlyCap::~FlyCap() noexcept
{
}

void FlyCap::run()
{
	fc2Image image;
	flycap_init_fatal(fc2CreateImage(&image), "Failed to create buffer for captured image");
	while(still_running()) {
		fc2RetrieveBuffer(ctx_, &image);
		auto res = resolution_t{ image.cols, image.rows };
		auto fmt = get_yuri_format(image.format );
		if (!fmt) {
			log[log::warning] << "Unsupported format received";
			continue;
		}

//		log[log::info] << "bytes: " << (int)(image.pData[0]&0xFF) << ", " <<
//				(int)(image.pData[1]&0xFF) << ", " <<
//				(int)(image.pData[2]&0xFF) <<  ", " <<
//				(int)(image.pData[3]&0xFF);
		auto frame = core::RawVideoFrame::create_empty(fmt, res, image.pData, image.dataSize);
		if (embedded_framecounter_) {
			int32_t fc = (image.pData[0]&0xFF) << 24 |
								 (image.pData[1]&0xFF) << 16 |
								 (image.pData[2]&0xFF) <<  8 |
								 (image.pData[3]&0xFF) <<  0;
			frame->set_index(fc);
		}
		push_frame(0, std::move(frame));
	}
}

bool FlyCap::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(resolution_, 	"resolution")
			(geometry_, 	"geometry")
			(fps_,			"fps")
			(serial_,		"serial")
			(index_, 		"index")
			(keep_format_,	"keep_format")
			(embedded_framecounter_,
							"embedded_framecounter")
			(trigger_,		"trigger")
			(trigger_mode_,	"trigger_mode")
			(trigger_source_,
							"trigger_source")
			(gpio_directions_[0],
							"gpio0")
			(gpio_directions_[1],
							"gpio1")
			(gpio_directions_[2],
							"gpio2")
			(gpio_directions_[3],
							"gpio3")
			(strobes_[0],	"strobe0")
			(polarities_[0],"strobe0_polarity")
			(delays_[0],	"strobe0_delay")
			(durations_[0],	"strobe0_duration")

			(strobes_[1],	"strobe1")
			(polarities_[1],"strobe1_polarity")
			(delays_[1],	"strobe1_delay")
			(durations_[1],	"strobe1_duration")

			(strobes_[2],	"strobe2")
			(polarities_[2],"strobe2_polarity")
			(delays_[2],	"strobe2_delay")
			(durations_[2],	"strobe2_duration")

			(strobes_[3],	"strobe3")
			(polarities_[3],"strobe3_polarity")
			(delays_[3],	"strobe3_delay")
			(durations_[3],	"strobe3_duration")
			(shutter_time_,	"shutter")
			(gain_,			"gain")
			(brightness_,	"brightness")
			(gamma_,		"gamma")
			(exposure_,		"exposure")
			(custom_,		"custom")
			.parsed<std::string>
				(format_, 	"format", core::raw_format::parse_format))
		return true;

	return core::IOThread::set_param(param);
}

} /* namespace flycap */
} /* namespace yuri */
