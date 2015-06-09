/*!
 * @file 		Frei0rWrapper.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		05.06.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Frei0rWrapper.h"
#include "yuri/core/Module.h"

#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/utils/DirectoryBrowser.h"
#include "yuri/core/utils/irange.h"
#include "yuri/core/utils/global_time.h"
#include <iostream>

namespace yuri {
namespace frei0r {


IOTHREAD_GENERATOR(Frei0rWrapper)

core::Parameters Frei0rWrapper::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Frei0rWrapper");
	return p;
}

Frei0rWrapper::Frei0rWrapper(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
base_type(log_,parent,std::string("frei0r")),Frei0rBase(log,parameters)
{
	IOTHREAD_INIT(parameters)
	module_ = make_unique<frei0r_module_t>(path_);

	log[log::info] << "Opened module " << module_->info.name << " by " << module_->info.author;
	log[log::info] << "type: " << module_->info.plugin_type << ", color: " << module_->info.color_model;
	switch (module_->info.color_model) {
		case F0R_COLOR_MODEL_BGRA8888:
			set_supported_formats({core::raw_format::bgra32});
			break;
		case F0R_COLOR_MODEL_RGBA8888:
			set_supported_formats({core::raw_format::rgba32});
			break;
		case F0R_COLOR_MODEL_PACKED32:
			set_supported_formats({	core::raw_format::bgra32,
									core::raw_format::rgba32,
									core::raw_format::abgr32,
									core::raw_format::argb32,
									core::raw_format::yuva4444,
									core::raw_format::yuyv422,
									core::raw_format::yvyu422,
									core::raw_format::uyvy422,
									core::raw_format::vyuy422});
			break;
		default:
			throw exception::InitializationFailed("Unsupported color format");
	}


}

Frei0rWrapper::~Frei0rWrapper() noexcept
{
	if (instance_) {
		module_->destruct(instance_);
	}
}

core::pFrame Frei0rWrapper::do_special_single_step(core::pRawVideoFrame frame)
{
	const auto res = frame->get_resolution();
	if (!instance_ || last_res_ != res) {
		if (instance_) {
			module_->destruct(instance_);
		}
		instance_ = module_->construct(res.width, res.height);
		set_frei0r_params();
		last_res_ = res;
	}
	auto outframe = core::RawVideoFrame::create_empty(frame->get_format(), res);
	auto dur = frame->get_timestamp() - core::utils::get_global_start_time();
	module_->update(instance_, dur.value/1.0e6, reinterpret_cast<uint32_t*>(PLANE_RAW_DATA(frame,0)), reinterpret_cast<uint32_t*>(PLANE_RAW_DATA(outframe,0)));

	return outframe;
}

bool Frei0rWrapper::set_param(const core::Parameter& param)
{
	if (Frei0rBase::set_param(param)) {
		return true;
	}
//	if (assign_parameters(param)
//			(path_, "_frei0r_path"))
//		return true;
	return core::IOThread::set_param(param);
}

} /* namespace frei0r */
} /* namespace yuri */
