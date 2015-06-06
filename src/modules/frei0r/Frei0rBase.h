/*!
 * @file 		Frei0rBase.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		06.06.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FREI0RBASE_H_
#define FREI0RBASE_H_

#include "yuri/core/thread/IOThread.h"
#include "frei0r_module.h"

namespace yuri {
namespace frei0r {

class Frei0rBase
{
protected:
	Frei0rBase(log::Log& log, core::Parameters params);
	void set_frei0r_params();
	bool set_param(const core::Parameter& param);
	log::Log& logb;
	core::Parameters params;
	std::unique_ptr<frei0r_module_t> module_;
	f0r_instance_t instance_;
	std::string path_;
};


} /* namespace frei0r */
} /* namespace yuri */



#endif /* FREI0RBASE_H_ */
