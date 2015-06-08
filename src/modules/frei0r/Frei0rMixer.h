/*!
 * @file 		Frei0rMixer.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		08.06.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FREI0RMIXER_H_
#define FREI0RMIXER_H_

#include "yuri/core/thread/MultiIOFilter.h"
#include "yuri/core/thread/Convert.h"
#include "Frei0rBase.h"

namespace yuri {
namespace frei0r {

class Frei0rMixer: public core::MultiIOFilter, private Frei0rBase
{
	using base_type = core::MultiIOFilter;


public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	Frei0rMixer(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~Frei0rMixer() noexcept;
private:
	virtual std::vector<core::pFrame> do_single_step(std::vector<core::pFrame> frames) override;
	virtual bool set_param(const core::Parameter& param) override;


	format_t format_;
	resolution_t last_res_;
	std::vector<core::pConvert> converters_;
};

} /* namespace frei0r */
} /* namespace yuri */
#endif /* FREI0RMIXER_H_ */
