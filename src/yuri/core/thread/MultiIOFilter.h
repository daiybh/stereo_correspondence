/*!
 * @file 		MultiIOFilter.h
 * @author 		Zdenek Travnicek
 * @date 		14.9.2013
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef MULTIIOFILTER_H_
#define MULTIIOFILTER_H_
#include "yuri/core/thread/IOThread.h"
#include "yuri/core/pipe/Pipe.h"

namespace yuri {
namespace core {

class MultiIOFilter: public IOThread {
public:
	static Parameters		configure();
							MultiIOFilter(const log::Log &log_, pwThreadBase parent,
			position_t inp, position_t outp, const std::string& id = "FILTER");

	virtual 				~MultiIOFilter() noexcept;

	std::vector<pFrame> 	single_step(const std::vector<pFrame>& frames);
	virtual bool 			step();

	virtual bool 			set_param(const Parameter &parameter) override;
protected:
	virtual void 				resize(position_t inp, position_t outp) override;
private:
	virtual std::vector<pFrame> do_single_step(const std::vector<pFrame>& frames) = 0;
	std::vector<pFrame> 	stored_frames_;
//	bool 					realtime_;
	position_t				main_input_;
};

}
}
#endif /* MULTIIOFILTER_H_ */
