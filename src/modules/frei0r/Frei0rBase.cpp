/*!
 * @file 		Frei0rBase.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		06.06.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "Frei0rBase.h"
#include "yuri/core/utils/irange.h"
#include "yuri/core/Module.h"

namespace yuri {
namespace frei0r {

Frei0rBase::Frei0rBase(log::Log& log, core::Parameters params)
:logb(log),params(std::move(params)),instance_(nullptr)
{

}

void Frei0rBase::set_frei0r_params()
{
	if (!instance_) return;
	if (module_->get_param_info && module_->set_param_value) {
		f0r_param_info_t param;
		for (auto i: irange(module_->info.num_params)) {
			module_->get_param_info(&param, i);
			if (!param.name) continue;
			try {

				switch(param.type) {
					case F0R_PARAM_DOUBLE: {
						auto p = params.get_parameter(param.name);
						double d = p.get<double>();
						module_->set_param_value(instance_, reinterpret_cast<f0r_param_t*>(&d), i);
					} break;
					case F0R_PARAM_BOOL: {
						auto p = params.get_parameter(param.name);
						double d = p.get<bool>()?1.0:0.0;
						module_->set_param_value(instance_, reinterpret_cast<f0r_param_t*>(&d), i);
					} break;
					case F0R_PARAM_STRING: {
						auto p = params.get_parameter(param.name);
						const char* d = p.get<std::string>().c_str();
						char* c = const_cast<char*>(d);
						module_->set_param_value(instance_, reinterpret_cast<f0r_param_t*>(&c), i);
					} break;
					case F0R_PARAM_POSITION: {
						auto px = params.get_parameter(std::string(param.name)+"_x");
						auto py = params.get_parameter(std::string(param.name)+"_y");
						f0r_param_position_t pos;
						pos.x = px.get<double>();
						pos.y = py.get<double>();
						module_->set_param_value(instance_, reinterpret_cast<f0r_param_t*>(&pos), i);
					} break;
					case F0R_PARAM_COLOR: {
						auto pr = params.get_parameter(std::string(param.name)+"_r");
						auto pg = params.get_parameter(std::string(param.name)+"_g");
						auto pb = params.get_parameter(std::string(param.name)+"_b");
						f0r_param_color_t col;
						col.r = pr.get<double>();
						col.g = pg.get<double>();
						col.b = pb.get<double>();
						module_->set_param_value(instance_, reinterpret_cast<f0r_param_t*>(&col), i);
					} break;
					default:
						logb[log::warning] << "Parameter " << param.name << " has unsupported type";
						continue;

				}
			}
			catch (std::runtime_error&)
			{

			}
		}
	}

}

bool Frei0rBase::set_param(const core::Parameter& param)
{
	return assign_parameters(param)
			(path_, "_frei0r_path");
}


} /* namespace frei0r */
} /* namespace yuri */
