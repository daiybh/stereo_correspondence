/*!
 * @file 		GPhoto.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		29.01.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "GPhoto.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/frame/compressed_frame_types.h"
namespace yuri {
namespace gphoto {


IOTHREAD_GENERATOR(GPhoto)

MODULE_REGISTRATION_BEGIN("gphoto")
		REGISTER_IOTHREAD("gphoto",GPhoto)
MODULE_REGISTRATION_END()

core::Parameters GPhoto::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Captures preview from a camera (Canon and possibly others)");
	p["timeout"]["Number of frames to wait before reconnecting"]=5;
	return p;
}

namespace {

int lookup_widget(CameraWidget* widget, const std::string& key, CameraWidget **child)
{
	int ret = 0;
	ret = gp_widget_get_child_by_name (widget, key.c_str(), child);
	if (ret < GP_OK)
		ret = gp_widget_get_child_by_label (widget, key.c_str(), child);
	return ret;
}

void print_status(GPContext* /* context */, const char* str, void* logp)
{
	auto log = *reinterpret_cast<log::Log*>(logp);
	log[log::debug] << str;
}

void print_err(GPContext* /* context */, const char* str, void* logp)
{
	auto log = *reinterpret_cast<log::Log*>(logp);
	log[log::error] << str;
}

}

GPhoto::GPhoto(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,0,1,std::string("gphoto")),opened_(false),
fail_count_(0),timeout_(5)
{
	set_latency(250_ms);
	IOTHREAD_INIT(parameters)
	context_ = gp_context_new();
	gp_context_set_error_func (context_, print_status, &log);
	gp_context_set_status_func (context_, print_err, &log);
}

GPhoto::~GPhoto() noexcept
{
}

bool GPhoto::open_camera()
{
	gp_camera_new(&camera_);
	if (gp_camera_init(camera_, context_) != GP_OK) {
		gp_camera_free(camera_);
		camera_ = nullptr;
		return false;
	}
	if (!enable_capture())
		return false;
	opened_ = true;
	fail_count_ = 0;
	return true;
}

bool GPhoto::enable_capture()
{
	std::unique_ptr<CameraWidget,std::function<void(CameraWidget*)>> widget
			(nullptr, [](CameraWidget*w){gp_widget_free(w);});

	CameraWidget * widget_tmp = nullptr;
	CameraWidget * child = nullptr;
	CameraWidgetType type;
	int ret = 0;
	ret = gp_camera_get_config(camera_, &widget_tmp, context_);
	if (ret != GP_OK) return false;
	widget.reset(widget_tmp);
	ret = lookup_widget(widget_tmp, "capture", &child);
	if (ret != GP_OK) {
		return false;
	}
	ret = gp_widget_get_type(child,&type);
	if (ret != GP_OK || type != GP_WIDGET_TOGGLE) {
		gp_widget_free(widget_tmp);
		return false;
	}
	int value = 1;
	ret = gp_widget_set_value(child, &value);
	if (ret != GP_OK) {
		return false;
	}
	ret = gp_camera_set_config (camera_, widget_tmp, context_);
	return ret == GP_OK;
}

bool GPhoto::close_camera()
{

	if (camera_) {
		gp_camera_exit(camera_, context_);
		gp_camera_free(camera_);
		camera_ = nullptr;
	}
	opened_ = false;
	return true;
}
void GPhoto::run()
{
	log[log::info] << "Waiting for connection to camera";
	while(still_running()) {
		if (!opened_) {
			if (open_camera()) {
				log[log::info] << "Camera connected.";
			} else {
				sleep(get_latency());
				continue;
			}
		}
		CameraFile *file = nullptr;
		gp_file_new(&file);
		auto ret = gp_camera_capture_preview(camera_, file, context_);
		if (ret != GP_OK) {
			if (!fail_count_) log[log::warning] << "Failed to capture " <<ret;
			if (fail_count_++ > timeout_) {
				log[log::error] << "Connection to camera lost.";
				close_camera();
			}
			gp_file_unref(file);
			continue;
		}
		fail_count_=0;
		const char* data;
		unsigned long int size;
		gp_file_get_data_and_size(file, &data, &size);
		auto frame = core::CompressedVideoFrame::create_empty(
							core::compressed_frame::jpeg,
							resolution_t{0,0},
							reinterpret_cast<const uint8_t*>(data),
							size);
		push_frame(0,frame);
		gp_file_unref(file);

	}
	close_camera();
}

bool GPhoto::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(timeout_, "timeout"))
		return true;
	return core::IOThread::set_param(param);
}

} /* namespace gphoto */
} /* namespace yuri */

