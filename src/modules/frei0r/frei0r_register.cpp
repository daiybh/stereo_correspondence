/*!
 * @file 		frei0r_register.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		05.06.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Frei0rWrapper.h"
#include "Frei0rSource.h"
#include "yuri/core/Module.h"

#include "yuri/core/utils/DirectoryBrowser.h"
#include "yuri/core/utils/irange.h"
#include "yuri/core/utils/environment.h"
#include <algorithm>
namespace yuri {
namespace frei0r {

namespace {
void set_f0r_bool_param(core::Parameter& p, const f0r_param_t * value)
{
	const double d = *reinterpret_cast<const double*>(value);
	p = d > 0.5;
}
void set_f0r_double_param(core::Parameter& p, const f0r_param_t * value)
{
	const double d = *reinterpret_cast<const double*>(value);
	p = d;
}
void set_f0r_position_param_x(core::Parameter& p, const f0r_param_t * value)
{
	const auto& d = *reinterpret_cast<const f0r_param_position_t*>(value);
	p = d.x;
}
void set_f0r_position_param_y(core::Parameter& p, const f0r_param_t * value)
{
	const auto& d = *reinterpret_cast<const f0r_param_position_t*>(value);
	p = d.y;
}
void set_f0r_color_param_r(core::Parameter& p, const f0r_param_t * value)
{
	const auto& d = *reinterpret_cast<const f0r_param_color_t*>(value);
	p = d.r;
}
void set_f0r_color_param_g(core::Parameter& p, const f0r_param_t * value)
{
	const auto& d = *reinterpret_cast<const f0r_param_color_t*>(value);
	p = d.g;
}
void set_f0r_color_param_b(core::Parameter& p, const f0r_param_t * value)
{
	const auto& d = *reinterpret_cast<const f0r_param_color_t*>(value);
	p = d.b;
}
void set_f0r_string_param(core::Parameter& p, const f0r_param_t * value)
{
	const auto c = *reinterpret_cast<const char*const*>(value);
	p = c;
}
template<class T>
std::function<decltype(T::configure())()> generate_configure(const std::string& name, const frei0r_module_t& module)
{
	auto description = std::string("[Frei0r] ") + module.info.name+" " +
			std::to_string(module.info.major_version)+"."+std::to_string(module.info.minor_version) +
			" (" + module.info.author+")";
	if (module.info.explanation) {
		description+=std::string(" - ")+module.info.explanation;
	}
	using param_type = decltype(T::configure());
	core::Parameters p = T::configure();
	p.set_description(description);
	p["_frei0r_path"]=name;
	auto instance = module.construct(8,8);
	if (module.get_param_info) {
		f0r_param_info_t param;
		for (auto i: irange(module.info.num_params)) {
			module.get_param_info(&param, i);
			if (!param.name) continue;
			std::string desc = param.explanation?param.explanation:"";
			f0r_param_t fp;
			module.get_param_value(instance, &fp, i);
			if (module.get_param_value) {
				switch(param.type) {
					case F0R_PARAM_DOUBLE:
						set_f0r_double_param(p[param.name][desc], &fp);
						break;
					case F0R_PARAM_BOOL:
						set_f0r_bool_param(p[param.name][desc], &fp);
						break;
					case F0R_PARAM_POSITION:
						set_f0r_position_param_x(p[std::string(param.name)+"_x"][desc], &fp);
						set_f0r_position_param_y(p[std::string(param.name)+"_y"][desc], &fp);
						break;
					case F0R_PARAM_COLOR:
						set_f0r_color_param_r(p[std::string(param.name)+"_r"][desc], &fp);
						set_f0r_color_param_g(p[std::string(param.name)+"_g"][desc], &fp);
						set_f0r_color_param_b(p[std::string(param.name)+"_b"][desc], &fp);
						break;
					case F0R_PARAM_STRING:
						set_f0r_string_param(p[param.name][desc], &fp);
						break;
					default:
						p[param.name][desc]="UNKNOWN";
						break;
				}
			}
		}
	}
	module.destruct(instance);
	return [p]()->IOThreadGenerator::param_type {
		return p;
	};
}
std::vector<std::string> add_vendor_subdirectories(std::vector<std::string> dirs)
{
	std::vector<std::string> dir2;
	for(auto&d: dirs) {
		auto&& sdirs = core::filesystem::browse_directories(d);
		dir2.push_back(std::move(d));
		dir2.insert(dir2.end(), sdirs.begin(), sdirs.end());
	}
	return dir2;
}
std::vector<std::string> get_plugin_directories()
{
	auto dirs = core::utils::get_environment_path("FREI0R_PATH");
	if (dirs.empty()) {
#ifdef YURI_POSIX
		auto home = core::utils::get_environment_variable("HOME");
		if (!home.empty()) {
			dirs.push_back(home+"/.frei0r-1/lib/");
		}
		dirs.push_back("/usr/lib/frei0r-1/");
		dirs.push_back("/usr/local/lib/frei0r-1/");
#endif
	}
	return add_vendor_subdirectories(std::move(dirs));
}

std::vector<std::string> list_plugins()
{
	std::vector<std::string> files;
	for (const auto& dir: get_plugin_directories()) {
		auto&& f = core::filesystem::browse_files(dir, "", ".so");
		files.reserve(files.size() + f.size());
		files.insert(files.end(), f.begin(), f.end());
	}

	std::sort(files.begin(), files.end());
	files.erase(std::unique(files.begin(), files.end()), files.end());

	return files;
}


}


MODULE_REGISTRATION_BEGIN("frei0r")
		const auto files = list_plugins();
		for (const auto& f: files) {
			try {
				frei0r_module_t module(f);
				switch (module.info.plugin_type) {
					case F0R_PLUGIN_TYPE_FILTER: {
						REGISTER_IOTHREAD_FUNC("frei0r_"+core::filesystem::get_filename(f, false),
							Frei0rWrapper::generate,
							generate_configure<Frei0rWrapper>(f, module))
					} break;
					case F0R_PLUGIN_TYPE_SOURCE: {
						REGISTER_IOTHREAD_FUNC("frei0r_source_"+core::filesystem::get_filename(f, false),
							Frei0rSource::generate,
							generate_configure<Frei0rSource>(f, module))
					} break;
				}
			}
			catch (std::exception& e)
			{
				std::cout << "Failed to load " << f << " ("<<e.what() <<")"<<std::endl;
			}

		}
		//REGISTER_IOTHREAD("frei0r",Frei0rWrapper)
		//REGISTER_IOTHREAD("frei0r_source",Frei0rSource)

		// This is ugly... but it is necesary now
		core::module_loader::leak_module_handle();
MODULE_REGISTRATION_END()


}
}


