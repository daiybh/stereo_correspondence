/*!
 * @file 		Convert.cpp
 * @author 		<Your name>
 * @date		30.10.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed BSD License
 *
 */

#include "Convert.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/thread/ConvertUtils.h"
#include "yuri/core/thread/IOThreadGenerator.h"
#include <unordered_map>
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmismatched-tags"
#endif

namespace std {
template<>
struct hash<std::pair<std::string, yuri::core::converter_key>> {
	size_t operator()(const std::pair<std::string, yuri::core::converter_key>& key) const{
		return std::hash<std::string>()(key.first) ^ std::hash<yuri::core::converter_key>()(key.second);
	}
};
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace yuri {
namespace core {


IOTHREAD_GENERATOR(Convert)

MODULE_REGISTRATION_BEGIN("convert")
		REGISTER_IOTHREAD("convert",Convert)
MODULE_REGISTRATION_END()

core::Parameters Convert::configure()
{
	core::Parameters p = core::IOFilter::configure();
	p.set_description("Convert");
	p["format"]["Target format"]="YUV";
	return p;
}


struct Convert::convert_pimpl_ {
	convert_pimpl_(log::Log& log_):log(log_){}
	log::Log &log;



	// Returns already prepared covnerter thread of creates new and returns it.
	pConverterThread get_thread(const std::string& name, converter_key key)
	{
		auto it = stateless_threads.find(name);
		if (it!=stateless_threads.end()) return it->second;

		auto it2 = statefull_threads.find({name, key});
		if (it2!=statefull_threads.end()) return it2->second;

		const auto& gen = IOThreadGenerator::get_instance();
		if (!gen.is_registered(name)) return {};

		Parameters par = gen.configure(name);
		pIOThread iot = gen.generate(name, log, pwThreadBase{}, par);
//		log[log::info] << "converter " << name << " " << (iot?"OK":"failed");
		pConverterThread pct = dynamic_pointer_cast<ConverterThread>(iot);
//		log[log::info] << "pct" << name << " " << (pct?"OK":"failed");
		if (pct) {
			if (pct->converter_is_stateless()) {
				log[log::debug] << "Storing stateless converter " << name;
				stateless_threads[name]=pct;
			} else {
				if (!pct->initialize_converter(key.second)) {
					return {};
				}
				log[log::debug] << "Storing statefull converter " << name;
				statefull_threads[{name, key}]=pct;
			}
		}
		return pct;
	}

	pFrame convert_step(pFrame frame_in, const convert::convert_node_t& step) {
		pConverterThread pct = get_thread(step.name, {step.source_format, step.target_format});
		if (!pct) return {};
		return pct->convert_frame(frame_in, step.target_format);
	}


	std::unordered_map<std::string, pConverterThread> stateless_threads;
	std::unordered_map<std::pair<std::string, converter_key>, pConverterThread> statefull_threads;


};


Convert::Convert(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOFilter(log_,parent,std::string("convert"))
{
	IOTHREAD_INIT(parameters)
	pimpl_.reset(new convert_pimpl_(log));
}

Convert::~Convert() noexcept
{
}

pFrame Convert::do_convert_frame(pFrame frame_in, format_t target_format)
{
	if (!frame_in) return {};
	format_t source_format = frame_in->get_format();
	if (source_format == target_format) return frame_in;
	auto path = find_conversion(source_format, target_format);
	if (path.empty()) {
		log[log::warning] << "Conversion not supported";
		return {};
	}
//	log[log::info] << "Path length: " << path.size();
	pFrame result = frame_in;
	for (const auto& step: path) {
//		log[log::info] << "Stepping to " << step.name;
		result = pimpl_->convert_step(result, step);
		if (!result) return {};
	}
//	log[log::info] << "COnversion ok";
	return result;
}

pFrame Convert::do_simple_single_step(const pFrame& frame)
{
	return convert_frame(frame, format_);
}

bool Convert::set_param(const core::Parameter& param)
{
	if (param.get_name() == "format") {
		format_ = raw_format::parse_format(param.get<std::string>());
	} else return core::IOThread::set_param(param);
	return true;
}

} /* namespace convert */
} /* namespace yuri */
