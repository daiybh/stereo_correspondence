/*
 * BasicIOFilter.h
 *
 *  Created on: 30.6.2013
 *      Author: neneko
 */

#ifndef BASICIOFILTER_H_
#define BASICIOFILTER_H_

#include "BasicIOThread.h"
#include "yuri/core/BasicPipe.h"
namespace yuri {
namespace core {


class BasicMultiIOFilter: public BasicIOThread {
public:
							BasicMultiIOFilter(log::Log &log_, pwThreadBase parent,
					yuri::sint_t inp, yuri::sint_t outp, std::string id = "FILTER")
:BasicIOThread(log_, parent, inp, outp, id),stored_frames_(inp) {}

	virtual 				~BasicMultiIOFilter() {}

	std::vector<pBasicFrame> single_step(const std::vector<pBasicFrame>& frames)
	{
		return do_single_step(frames);
	};
	virtual bool step()
	{
		bool ready = true;
		assert(in_ports>0);
		assert(stored_frames_.size() == static_cast<size_t>(in_ports));
		for (sint_t i=0; i< in_ports; ++i) {
			if (!stored_frames_[i]) stored_frames_[i]=in[i]->pop_frame();
			if (!stored_frames_[i]) ready = false;
		}
		if (ready) {
			auto outframes = single_step(stored_frames_);
//			assert(outframes.size() == outputs);
			for (size_t i=0; i< std::min(static_cast<size_t>(out_ports), outframes.size()); ++i) {
				if (outframes[i]) push_raw_frame(i, outframes[i]);
			}
			for (auto& sf: stored_frames_) sf.reset();
		}
		return true;
	}
private:
	virtual std::vector<pBasicFrame> do_single_step(const std::vector<pBasicFrame>& frames) = 0;
	std::vector<pBasicFrame> stored_frames_;

};

class BasicIOFilter: public BasicMultiIOFilter
{
public:
							BasicIOFilter(log::Log &log_, pwThreadBase parent,
				std::string id = "FILTER");
	virtual 				~BasicIOFilter();

	pBasicFrame				simple_single_step(const pBasicFrame& frame);

private:
	virtual pBasicFrame		do_simple_single_step(const pBasicFrame& frame) = 0;
	virtual std::vector<pBasicFrame> do_single_step(const std::vector<pBasicFrame>& frames);

};




}
}



#endif /* BASICIOFILTER_H_ */
