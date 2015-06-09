/*!
 * @file 		Frei0rMixer.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		08.06.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Frei0rMixer.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"


#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/utils/DirectoryBrowser.h"
#include "yuri/core/utils/irange.h"
#include "yuri/core/utils/global_time.h"
#include <iostream>

namespace yuri {
namespace frei0r {


IOTHREAD_GENERATOR(Frei0rMixer)

core::Parameters Frei0rMixer::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("Frei0rMixer");
	return p;
}

Frei0rMixer::Frei0rMixer(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
MultiIOFilter(log_,parent,1,1,std::string("frei0r")),Frei0rBase(log,parameters),format_(0),last_res_{0,0}
{
	IOTHREAD_INIT(parameters)
	module_ = make_unique<frei0r_module_t>(path_);
	if (!module_->update2) {
		throw exception::InitializationFailed("Module doesn't export update2 method!");
	}
	if (module_->info.plugin_type == F0R_PLUGIN_TYPE_MIXER2) {
		resize(2,1);
		log[log::info] << "Loaded plugin for 2 inputs";
	} else if (module_->info.plugin_type == F0R_PLUGIN_TYPE_MIXER3) {
		resize(3,1);
		log[log::info] << "Loaded plugin for 2 inputs";
	} else {
		log[log::warning] << "Loaded unexpected plugin type, assuming single input";
	}
	log[log::info] << "Opened module " << module_->info.name << " by " << module_->info.author;
	log[log::info] << "type: " << module_->info.plugin_type << ", color: " << module_->info.color_model;
	switch (module_->info.color_model) {
		case F0R_COLOR_MODEL_BGRA8888:
			format_ = core::raw_format::bgra32;
			break;
		case F0R_COLOR_MODEL_RGBA8888:
			format_ = core::raw_format::rgba32;
			break;
		case F0R_COLOR_MODEL_PACKED32:

//			set_supported_formats({	core::raw_format::bgra32,
//									core::raw_format::rgba32,
//									core::raw_format::abgr32,
//									core::raw_format::argb32,
//									core::raw_format::yuva4444,
//									core::raw_format::yuyv422,
//									core::raw_format::yvyu422,
//									core::raw_format::uyvy422,
//									core::raw_format::vyuy422});
//			break;
		default:
			throw exception::InitializationFailed("Unsupported color format");
	}


}

Frei0rMixer::~Frei0rMixer() noexcept
{
	if (instance_) {
		module_->destruct(instance_);
	}
}

std::vector<core::pFrame> Frei0rMixer::do_single_step(std::vector<core::pFrame> frames)
{
	const size_t inputs = get_no_in_ports();
	if (frames.size() != inputs) {
		return {};
	}
	converters_.resize(inputs);
	for (auto& c: converters_) {
		if (!c) c = std::make_shared<core::Convert>(log, get_this_ptr(), core::Convert::configure());
	}
	std::vector<core::pRawVideoFrame> rframes;
	rframes.resize(inputs);
	for (auto idx: irange(inputs)) {
		auto f = std::dynamic_pointer_cast<core::RawVideoFrame>(converters_[idx]->convert_frame(frames[idx], format_));
		if (!f) return {};
		rframes[idx]=std::move(f);
	}
	const auto res = rframes[0]->get_resolution();
	for (const auto& f: rframes) {
		if (f->get_resolution() != res) {
			log[log::warning] << "Bad resolution!";
			return {};
		}
	}
	if (!instance_ || last_res_ != res) {
		if (instance_) {
			module_->destruct(instance_);
		}
		instance_ = module_->construct(res.width, res.height);
		set_frei0r_params();
		last_res_ = res;
	}
	auto outframe = core::RawVideoFrame::create_empty(format_, res);
	auto dur = rframes[0]->get_timestamp() - core::utils::get_global_start_time();
	uint32_t* input2 = nullptr;
	if (inputs>1) input2= reinterpret_cast<uint32_t*>(PLANE_RAW_DATA(rframes[1],0));
	uint32_t* input3 = nullptr;
	if (inputs>2) input3 = reinterpret_cast<uint32_t*>(PLANE_RAW_DATA(rframes[2],0));
	module_->update2(instance_,
					dur.value/1.0e6,
					reinterpret_cast<uint32_t*>(PLANE_RAW_DATA(rframes[0],0)),
					input2,
					input3,
					reinterpret_cast<uint32_t*>(PLANE_RAW_DATA(outframe,0)));

	return {outframe};
}

bool Frei0rMixer::set_param(const core::Parameter& param)
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
