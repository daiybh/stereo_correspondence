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

/*
 * Main input policies:
 * 0 or higher 	- threat input with this index as main input
 * -1 			- Trigger processing only after there's a new frame on each input
 * -2			- Trigger processing when there's new frame on any input (other frames are kept)
 * -3			- Trigger processing when there's new frame on any input (frames are not kept)
 */

class MultiIOFilter: public IOThread {
public:
	EXPORT static Parameters
							configure();
	EXPORT 					MultiIOFilter(const log::Log &log_, pwThreadBase parent,
			position_t inp, position_t outp, const std::string& id = "FILTER");

	EXPORT virtual 			~MultiIOFilter() noexcept;

	EXPORT std::vector<pFrame> 	
							single_step(std::vector<pFrame> frames);
	EXPORT virtual bool 	step();

	EXPORT virtual bool 	set_param(const Parameter &parameter) override;
protected:
	EXPORT virtual void 	resize(position_t inp, position_t outp) override;
private:
	virtual std::vector<pFrame> do_single_step(std::vector<pFrame> frames) = 0;
	std::vector<pFrame> 	stored_frames_;
//	bool 					realtime_;
	position_t				main_input_;
};

}
}
#endif /* MULTIIOFILTER_H_ */
